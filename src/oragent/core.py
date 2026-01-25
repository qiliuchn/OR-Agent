# src/oragent/core.py
"""
# OR-Agent

## Overview
OR-Agent is an automated research system designed to discover optimal solutions to complex problems through multi-agent collaboration,
evolutionary algorithms, optimization-inspired reflection mechanisms.



## Key features include:
1. **Multi-Agent Research Framework**: A distributed architecture where multiple lead agents conduct parallel research processes, 
    with specialized experiment agents exploring problem environments and executing evaluation scripts for feedback.
2. **Evolutionary Ideation**: Incorporates evolutionary algorithm principles into the research ideation phase, enabling exploration of uncharted problem spaces 
    and generation of innovative hypotheses. A shared solution database allows lead agents to draw parent solutions for subsequent research rounds, 
    using MAP-Elite and Island-based selection methods.
3. **Tree-Search Workflow Management**: Employs a tree-search-based workflow controller that models the branching structure of human research processes, 
    supporting both divergent exploration and iterative refinement.
4. **Optimization-Inspired Reflection**: Implements reflection mechanisms inspired by classical optimization methods: 
    experiment reflection acts as a verbal gradient, long-term reflection functions as verbal momentum, and reflection compression uses an exponential-decay schedule to stabilize updates, 
    enabling efficient convergence to high-quality solutions.
 
 

## Architecture and Workflow
The OR-Agent system operates through a coordinated multi-agent architecture with the following components:
- **OR Agent** (`ORAgent` class): The entry point of OR-Agent; coordinates the solution database and manages lead agents. 
    Supports multiple lead agents conducting research concurrently, sharing insights and collaborating.
- **Solution Database** (`SolutionDatabase`): Central repository storing all generated solutions, serving as a shared "knowledge pool" for all agents.
- **Lead Agents** (`LeadAgent` class): Conduct research rounds by starting from parent solutions and generating/refining solutions through experiments.
- **Idea Agent** (`IdeaAgent` class): Generates solution ideas.
- **Code Agent** (`CodeAgent` class): Implements ideas as code and handles debugging.
- **Experiment Agent** (`ExperimentAgent` class): Conducts experiments like a research scientist. 
    Investigates solutions (with ideas and code), identifies issues, attempts improvements, and 
    repeats until no further improvements are possible without restructuring. 
    Writes reports summarizing issues and potential improvement directions.
- **Workflow Control** (`FlowGraph` class): Each lead agent manages research workflow as a principal investigator. 
    Organizes workflow as a tree structure (`FlowGraph`), where each node represents an intermediate research step 
    with revised ideas and code implementations.



### Solution Representation
## Solution representation and storage
`Solution` dataclass is defined in `utils.py`; `Solution` is used by all algorithms (OR-Agent, ReEvo, EoH, AEL and FunSearch); 
it has fields like ids, code, output, metrics, features, score, and summary. Check `utils.Solution` for more details.

Notes on `metrics`, `features`, and `score`:
`metrics` is used only as prompt context. In principle any metrics type is supported.
We recommend `metrics` to take one of two following forms:
    1. A nested dictionary mapping test identifiers to their respective metric dictionaries (test_name -> {metric_name -> float}); type: Dict[str, Dict[str, float]]
    2. An aggregated dictionary mapping metric names directly to values, e.g. average test metrics; type: Dict[str, float]
    
`features` is derived from `metrics` and serves as the feature vector for the MAP-Elites-based solution database. 
    Since MAP-Elites requires discrete indices, we require `features` to be integers. 
    Typically, this involves binning aggregated float metrics into categorical values. 
    Additional attributes not directly measuring performance—such as code length (categorized)—can also be included.
    Note that the bin granularity controls cluster size in the solution database! Adjust according to your needs in eval.py.

`score` is a float value representing the overall quality of the solution. It is calculated based on the `metrics` and `features` fields.
    Whether a higher score indicates a better solution or better quality depends on config['ob_type'].
Users should define how to generate `metrics`, `features`, and `score` in `eval.py` for tested problem.


### `Node`
`Node` and `FlowGraph` classes are defined in `flow_graph.py`. `Node` wraps solution with additional attributes for tree structure organization. `Node` is internal to `LeadAgent`.
So, A Node obj has attributes like solution, parent, children, etc. `FlowGraph` manages the tree structure.


### Solution storage: `SolutionDatabase` and `FlowGraph`
`SolutionDatabase` (defined in `SolutionDatabase.py`) is the database where solutions are stored and shared. There is only one solution database for the entire research process.
`FlowGraph` is the place where the intermediate solutions (wrapped as `Node`s) generated during one round of research are stored. Nodes are organized as a tree.
At the end of a research round, approximate local optimum solutions (leaf nodes of the flow graph) are collected and added to the solution database.



## LLM clients
- **Individual Clients**: Each specialized agent creates its own LLM client, allowing independent parameter settings (e.g., temperature)
- **Model Selection**: Specialized agents load LLM provider and model names from the configuration file



## About autosaving
`ORAgent` class is responsible for triggering autosaving.
All specialized agents should support `save(checkpoint)` and `load(checkpoint)` method. All agents should save the variables created by themselves.
For simplicity, autosaving condition is only checked when `LeadAgent.run()` finishes one round of research (see ORAgent.run() method).
We intended to support auto-saving at during one round of research, this requires us to coordinate `ORAgent` and `LeadAgent` objects to save at the same checkpoint folder at the same time.
We leave this functionality for future versions.



## What the script contains
 - `ORAgent`: the class that provide the entry point to the OR-Agent package.
"""
import os
import sys
from pathlib import Path
import yaml
import json
import time
from datetime import datetime
import shutil
from oragent.lead_agent import LeadAgent
from oragent.solution_database import SolutionDatabase
import oragent.utils as utils
from oragent.utils import Solution  # `Solution` dataclass is used as container to store solution idea, code, eval results, and other related information, passed between agents




class ORAgent:
    """Organizes all resources, including solution database and lead agents."""
    def __init__(self, checkpoint=None, config=None) -> None:
        # =====General configuration start (common for all agents)=====
        self.package_dir = Path(__file__).parent
        self.project_root = Path.cwd()
        # Problem data is stored in `<project_root>/problems`
        # Prompts are stored in `<project_root>/prompts`
        
        if checkpoint:  # if checkpoint specified, load checkpoint
            self.load(checkpoint=checkpoint)
        elif config:  # else, check if config is provided
            self.config = config
        else:  # else, use default config
            # Load built-in config if config is not provided
            with open(f'{self.package_dir}/config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
                
        self.algorithm = self.config['algorithm'].lower().strip()
        self.problem = self.config['problem'].lower().strip()  # the problem to solve
        # Load experiment config if not provided
        if 'experiment' not in self.config:
            experiment_config_path = self.project_root / "problems" / self.problem / "settings.yaml"
            with open(experiment_config_path, 'r') as f:
                experiment_config = yaml.safe_load(f)
            self.config['experiment'] = experiment_config
        # Experiment config
        self.function_to_evolve = self.config['experiment']['function_to_evolve']  # the name of the function to be evolved
        self.obj_type = self.config['experiment']['obj_type'].lower().strip()
        assert self.obj_type in ['max', 'min'], f"Invalid objective type: {self.obj_type}"
        
        # =====OR-Agent Settings=====
        self.init_pop_size = self.config['init_pop_size']
        self.num_lead_agents = self.config['num_lead_agents']
        self.max_research_rounds = self.config['max_research_rounds']
        self.num_parents = self.config['num_parents']
        self.autosave_interval_minutes = self.config['autosave_interval_minutes']
        self.max_evolutions = self.config['max_evolutions']
        # Print out some settings
        print("\n>>>[ORAgent] Settings:")
        print("Problem:", self.problem)
        print("Initial population size:", self.init_pop_size)
        print("Number of num_lead_agents:", self.num_lead_agents)
        print("Max research rounds:", self.max_research_rounds)
        
        # =====Problem data and prompts=====
        self.problem_dir = f"{self.project_root}/problems/{self.problem}"
        self.prompt_dir = f"{self.project_root}/prompts"
        self.output_dir = self.config['output_dir'] or f"{self.project_root}/outputs/{self.algorithm}/{self.problem}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/details", exist_ok=True)  # folder to store details
        
        # =====Create solution database=====
        # Create the program database that is shared by all research rounds; only one database instance are kept alive throughout the research rounds
        # Create new one if no checkpoint specified
        if not checkpoint:
            self.solution_database = SolutionDatabase(config=self.config)
        
        # =====Create lead agent=====
        # Create one lead agent (`run()` is single-threaded)
        if not checkpoint:
            self.lead_agent = LeadAgent(config=self.config)
        
        # =====Initialize results.json (common for all agents)=====
        # Create a new results.json if user starts a new session; and do this before generating any solution
        if not checkpoint:
            results_path = f"{self.output_dir}/results.json"
            utils.init_json_list(results_path)
            # results_detailed.json is more detailed version of results.json
            results_path = f"{self.output_dir}/results_detailed.json"
            utils.init_json_list(results_path)
                
        # =====Initialize population=====
        # Initialize population if checkpoint not specified
        if not checkpoint:
            self.init_population()
        print("\n>>>[ORAgent] ORAgent initialized")
        
        
    def init_population(self) -> None:
        # Load the seed solution
        seed_solution = None  # create solution with empty fields
        if Path(f"{self.problem_dir}/seed_solution_idea.txt").exists():
            seed_solution = Solution()
            seed_solution.idea = utils.file_to_string(f"{self.problem_dir}/seed_solution_idea.txt")
            if Path(f"{self.problem_dir}/seed_solution.py").exists():
                seed_solution.code = utils.file_to_string(f"{self.problem_dir}/seed_solution.py")
        
        # Initialize the database by creating the initial population
        init_pop = self.lead_agent.init_population(seed_solution)  # Note: seed_solution is possibly empty
        
        # Add initial population to the database
        self.solution_database.add(init_pop)
        
        # Visualize database
        print(f"\n>>>[ORAgent] Database initialized:")
        self.solution_database.visualize()
        # Log database
        file_name = f"{self.output_dir}/database_init.txt"
        with open(file_name, 'w') as file:
            self.solution_database.visualize(file=file)
        # Save database for webui real-time visualization
        with open(f"{self.output_dir}/database.txt", 'w') as file:
            self.solution_database.visualize(file=file)
    
    
    def save(self, checkpoint: str=None):
        """
        Save checkpoint. Saved files:
        - config.yaml
        - database.json

        Args:
            checkpoint (str): checkpoint name; default None

        Return:
            None.
        """
        checkpoint = checkpoint or datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{self.algorithm}_{self.problem}"  # default checkpoint name example: '2025-12-29_20-40-25'
        checkpoint_directory = f'{self.project_root}/checkpoints/{checkpoint}'
        os.makedirs(checkpoint_directory, exist_ok=True)

        # Save config
        config_path = os.path.join(checkpoint_directory, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # Save solution database
        self.solution_database.save(checkpoint=checkpoint)
        
        # Save lead agent
        self.lead_agent.save(checkpoint=checkpoint)
        
        # Save results.json
        # we need to save a copy of results.json instead of directly append to the existing one in output dir
        # since the existing one may be ahead of the checkpoint in terms of iterations
        results_source = f"{self.output_dir}/results.json"
        results_dest = os.path.join(checkpoint_directory, 'results.json')
        if os.path.exists(results_source):
            shutil.copy2(results_source, results_dest)
            
        print(f"\n>>>[ORAgent] Checkpoint saved to: {checkpoint_directory}")
    
    def load(self, checkpoint):
        """Load checkpoint."""
        checkpoint_directory = f'{self.project_root}/checkpoints/{checkpoint}'

        # Load config
        config_path = os.path.join(checkpoint_directory, 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Restore solution database
        self.solution_database = SolutionDatabase(checkpoint=checkpoint)
        
        # Restore lead agent
        self.lead_agent = LeadAgent(checkpoint=checkpoint)
        
        # Restore results.json
        results_source = os.path.join(checkpoint_directory, 'results.json')
        self.algorithm = self.config['algorithm'].lower().strip()
        self.problem = self.config['problem'].lower().strip()
        self.output_dir = self.config['output_dir'] or f"{self.project_root}/outputs/{self.algorithm}/{self.problem}"
        results_dest = f"{self.output_dir}/results.json"
        if os.path.exists(results_source):
            shutil.copy2(results_source, results_dest)
            
        print(f"\n>>>[ORAgent] Checkpoint loaded from: {checkpoint_directory}")
        
    
    def run(self):
        """Start rounds of research in single-thread."""  
        last_save_time = time.time()
              
        # Run research rounds
        while self.lead_agent.get_total_responses() <= self.max_evolutions or self.lead_agent.get_function_evals() <= self.max_evolutions:
            print(f"\n>>>[ORAgent] **Researcher {self.lead_agent.id} round {self.lead_agent.research_round} starts**")
            
            # =====Lead agent conduct one round of research=====
            # Previous, we let ORAgent to sample from solution database and pass parent solutions (`Union(List[Solution])`) to LeadAgent.run().
            # vs here we pass database directly; previous practice support multi-processing
            # but since ORAgent.run() is a single-threaded, it does matters much;
            # current practice gives lead agent more freedom to decide on itself how to sample from the database
            child_solutions = self.lead_agent.run(self.solution_database)
            
            # =====Add newly generated solutions to the database=====
            self.solution_database.add(child_solutions)
            
            # =====Log and visualize database=====
            print(f"\n>>>[ORAgent] Database updated:")
            # Visualize database
            self.solution_database.visualize()
            # Log database
            file_name = f"{self.output_dir}/details/database_lead{self.lead_agent.id}_round{self.lead_agent.research_round}.txt"
            with open(file_name, 'w') as file:
                self.solution_database.visualize(file=file)
                
            # Save database for webui real-time visualization
            with open(f"{self.output_dir}/database.txt", 'w') as file:
                self.solution_database.visualize(file=file)
            
            # =====Lead agent update iter=====
            print(f"\n>>> [ORAgent] **Researcher {self.lead_agent.id} round {self.lead_agent.research_round} completed**")
            self.lead_agent.update_iter()  # lead agent number of research rounds is tracked internally; will increment automatically
            
            # =====Autosaving=====
            if time.time() - last_save_time >= self.autosave_interval_minutes * 60:
                self.save()
                last_save_time = time.time()


    def run_parallel(self):
        """
        Parallel execution of the research process.

        In a simple implementation, this is a multiprocessing version of the run() method.
        For more complex implementations, multiple lead agents can conduct research concurrently,
        potentially sharing insights and collaborating with each other.
        """
        # TODO:
