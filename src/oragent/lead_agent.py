# src/oragent/lead_agent.py
"""
# Lead Agent module

## Overview
The lead agent orchestrates the research process by coordinating specialized agents:
- **Idea agent**: Generates research ideas and can produce multiple ideas simultaneously to form comprehensive research plans.
- **Code agent**: Implements code based on ideas and debugs generated code when errors occur.
- **Experiment agent**: Conducts experiments by investigating the environment through callbacks, identifying issues, and iteratively refining solutions. 
    Each experiment concludes with a report summarizing discovered issues, proposed solutions, and directions for future research.
New ideas are generated from experiment reports, leading to new code implementations and subsequent experiments in an iterative cycle. 
The lead agent coordinates this research process by managing the workflow, which is structured as a tree.


 
 
## Lead agent work process
The run() method of the lead agent executes a single round of research and returns the generated solutions. one solution is called "complete" if it has all fields.
The lead agent uses an instance of the FlowGraph class (from the flow_graph module) to manage the research workflow, which is structured as a tree. 
Each node in the tree is represented by a Node object that encapsulates a solution dictionary along with additional tree-related metadata. 
A node is marked as `processed=True` once its solution contains all three components: ideas, code, and experimental results/reports. 
Each node also includes a `done` attribute, initialized as False. A leaf node can be marked as `done=True` only when it is determined to be un-improvable—i.e., 
when it represents a local optimum.

### Lead agent workflow
The lead agent executes one research round through the following steps:
0. **Initialization**: Create a FlowGraph instance with root node containing parent solutions sampled from the database.
1. **Research start**: Extend the root node to generate up to `max_children` children.
   These represent evolutionary starting points in the solution space.
2. **Research loop** (while loop):
   2.1. **Select best leaf**: Among all unfinished leaf nodes (`is_done=False`), select the one with the best score (node `N_i`).
        Selection considers `obj_type`: maximize or minimize score.
   2.2. **Node extension**: Extend node `N_i` to generate up to `max_children` children through idea generation, code implementation, and experimentation.
   2.3. **Node truncation & Local optimality test**:
        - Keep only children with better scores than `N_i` (performance ascending direction)
        - If no children improve upon `N_i`, mark `N_i` as `done=True` (approximate local optimum)
   2.4. **Research finishing test**: If all leaf nodes are done, break; otherwise repeat from 2.1.
3. **Return**: Return all leaf nodes as the research round's output.

Notes:
- By saying "extending" a node, we mean following the process of idea generation, code generation, and experiments.
-  The lead agent begins each research round by randomly sampling parent programs from the database, analogous to the initialization step in evolutionary algorithms. 
This process naturally incorporates the mutation and crossover operations commonly used in LLM-based genetic algorithms (e.g., AEL, EoH, ReEvo). Crossover occurs when 
the lead agent selects two parent programs and combines them to produce new offspring, whereas mutation occurs when the lead agent is instructed to further 
extend a node and generate additional variants.
- Unlike prior LLM-based genetic algorithms, OR-Agent does not just rely on frequent mutation and crossover operations. It also performs extensive and 
systematic investigation around each starting point before moving on to the next. This workflow is more aligned with how human scientists actually conduct research: 
generating ideas through crossover and mutation alone is insufficient—rigorous refinement, iterative updates, and targeted experimentation are essential for 
developing high-quality scientific insights.
- OR-Agent uses a simple organization: only LeadAgent who orchestrates the workflow communicates with other agents (IdeaAgent, CodeAgent, ExperimentAgent); 
    for future studies we may adopt more complex workflow.

### Illustration of lead agent workflow
Lead agent workflow for conducting a single research round:
```
1. "Research start"    --->  2. "Research Loop"                                                                                                                                                                                          3) "Return" leaf nodes
(long term reflection initialized)      ↑      --->        2.1. "Select best leaf"            --->        2.2) "Node Extension"   --->   2.3) "Node truncation & Local optimality test"   --->   2.4) "Research finishing test"    --->
                                        |                   |                                ↑                                                                                                                                    |                                     
                                        |                   |                       Leaf finished processing                                                                                                                      |
                                        |                   |                   (long term reflection updated)                                                                                                                    |
                                        |                   ↓                                |                                                                                                                                    |
                                        |             Experiment loop   --→  experiment summary generated                                                                                                           Research round not finished yet
                                        |                   |     ↑                                                                                                                                                               |
                                        |                   ↓     |                                                                                                                                                               |
                                        |   Experiment result & experiment reflection                                                                                                                                             |
                                        |   generated and appended to context                                                                                                                                                     |
                                         -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```




## Hierarchical Reflections
The lead agent maintains hierarchical reflections throughout the research process:
- **Experiment reflections**: Generated after each experiment step and appended to context (ReAct-style)
- **Experiment summaries**: Generated at the end of solution experiments, summarizing outcomes and insights
- **Long-term reflections**: Maintained across the research round, updated when node summaries are generated

### Reflection Hierarchy
```
Experiment reflections → Experiment summary → Long-term reflection
```
Long-term reflections are initialized at research round start and updated when nodes complete processing.
We don't update long-term reflection after each evaluation; since there are lots of small things like fixing syntax error, adjust parameter; "experiment summary" will summarize the whole experiment.
And long term reflection only update with experiment summary.


Lead agent output json fenced block for long-term reflection update like this:
```json
{
  "reflection": An updated long-term reflection that synthesizes prior knowledge with new insights, maintaining a cumulative understanding of the research progress and guiding future investigations.
}
```




## Tree depth constraint
# The depth of a tree refers the length of the longest path from the root to a leaf.
Here's a simple example:
         A          depth 0 (root node)
       / |
      B  C          depth 1
     /   |
    D    E          depth 2
   /
  F                 depth 3

config['max_tree_depth'] is used to constrain the maximum depth of the research tree.
if None, then there is no depth constraint.



## Performance tracking
LeadAgent, IdeaAGent, CodeAgent, ExperimentAgent will keep track of the llm calls and evaluations that directly invoked by themselves!
Examples are like: `llm_client.chat()`, `evaluator.evaluate()`;
The LeadAgent is responsible for collecting the total number of calls and evaluations; see `get_total_responses()`, `get_function_evals()`, `get_valid_responses()` of `LeadAgent`.




## What the script contains
 - `LeadAgent`: The main class that manages the evolution process. It initializes the program database, evaluator, and agents, and coordinates the research process.
"""
import os
import sys
import yaml
import json
from pathlib import Path
import time
from datetime import datetime
import dataclasses
from typing import List, Tuple, Optional, Dict, Union
from oragent.idea_agent import IdeaAgent
from oragent.code_agent import CodeAgent
from oragent.experiment_agent import ExperimentAgent
from oragent.evaluator import Evaluator
from oragent.flow_graph import Node, FlowGraph
import oragent.utils as utils
from oragent.utils import Solution
from oragent.solution_database import SolutionDatabase




class LeadAgent:
    """Agent to orchestrate rounds of research."""
    _id_counter = 0  # Class variable shared by all instances; used to generate unique lead agent IDs
    
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
        
        # =====LeadAgent settings=====
        # Lead agent id
        LeadAgent._id_counter += 1
        self.init_pop_size = self.config['init_pop_size']  # initial population size 
        self.num_parents = self.config['num_parents']  # number of parents used to start a round of research
        self.num_children = self.config['num_children']  # number of children to generate for each parent node in the tree
        self.max_tree_depth = self.config['max_tree_depth']  # maximum depth of the research tree
        self.num_islands = self.config['database']['num_islands']  # number of islands in database; used in population initialization for even distribution
        self.elitist_as_root_period = self.config['elitist_as_root_period']  # the research rounds period for re-using elitist as root; "0" means never directly use elitist as root
        self.elitist_enlargement_factor = self.config['elitist_enlargement_factor']  # the enlargement factor for elitist when used as root; int(elitist_enlargement * num_children) many children will be generated for elitist
        self.reflection_compression = self.config['reflection_compression']  # the limit (number of words) to compress long-term memory between research rounds; None means no compression
        self.reflection_period = self.config['reflection_period']  # the batch size of experiment reflections used to update long-term memory
        self.reflection_clearance_period = self.config['reflection_clearance_period']  # the research rounds period for clearing long-term memory; None means no clearance

        # =====Create LLM client=====
        self.llm_provider = self.config['model']['lead_agent_llm_provider']
        self.model_name = self.config['model']['lead_agent_model_name']
        self.llm_client = utils.LLMClient(config=self.config, llm_provider=self.llm_provider, model_name=self.model_name)
        
        # =====Create specialized agents=====
        # those agents have state vars
        # if checkpoint is specified, load() method will create them.
        if not checkpoint:
            self.idea_agent = IdeaAgent(config=self.config)
            self.code_agent = CodeAgent(config=self.config)
            self.experiment_agent = ExperimentAgent(config=self.config)
        
        # =====Create evaluator=====
        self.evaluator = Evaluator(config=self.config)
        
        # =====Vars updated during agent running=====
        # If checkpoint not specified, we need to create them
        if not checkpoint:
            self.id = LeadAgent._id_counter
            #self.iteration = 0  # number of evolution rounds
            # Solution ids
            self.research_round = 1  # current research rounds; starts from 1
            self.solution_count = 0  # solution output count for one research round; starts from 1
            # performance indices
            self.total_responses = 0  # Number of total responses; this can be used to track the number of LLM calls
            self.function_evals = 0  # Number of function evaluations; this is also an important metric for complexity, especially for the case when evaluation is the bottleneck
            self.valid_responses = 0 # Number of valid responses, namely responses that were successfully executed
            self.elitist = None   # Best individual so far
            self.flow_graph = None
        
        # =====Load problem data and prompts=====
        # Problem and prompt directory
        self.problem_dir = f"{self.project_root}/problems/{self.problem}"
        self.prompt_dir = f"{self.project_root}/prompts"
        self.output_dir = self.config['output_dir'] or f"{self.project_root}/outputs/{self.algorithm}/{self.problem}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/details", exist_ok=True)  # folder to store details
        # Load Common prompts
        self.system_long_term_reflection_prompt = utils.file_to_string(f'{self.prompt_dir}/system_long_term_reflection_oragent.txt')  # system role prompt for reflection
        self.user_long_term_reflection_prompt = utils.file_to_string(f'{self.prompt_dir}/user_long_term_reflection_oragent.txt')  # user role prompt for long-term reflection
        #self.system_reflection_compression_prompt = utils.file_to_string(f'{self.prompt_dir}/system_reflection_compression_oragent.txt')  # system role prompt for memory compression
        #self.user_reflection_compression_prompt = utils.file_to_string(f'{self.prompt_dir}/user_reflection_compression_oragent.txt')  # user role prompt for memory compression
        # Load problem-specific prompts
        self.problem_description = utils.file_to_string(f'{self.problem_dir}/problem_description.txt')
        self.function_description = utils.file_to_string(f'{self.problem_dir}/function_description.txt')
        if os.path.exists(f'{self.problem_dir}/external_knowledge.txt'):
            self.external_knowledge = utils.file_to_string(f'{self.problem_dir}/external_knowledge.txt')
        else:
            self.external_knowledge = ""
        
        # =====Set long term reflection to external knowledge=====
        # if no checkpoint specified, we set long-term reflection to be external knowledge at initialization.
        if not checkpoint:
            self.long_term_reflection = self.external_knowledge  # long term reflection; re-initialized at the start of each research round
            self.experiment_summaries = []  # store experiment summaries for long-term reflection update
            
            # Log long_term_reflection
            with open(f"{self.output_dir}/long_term_reflection.txt", 'w') as file:
                file.write(self.long_term_reflection)
        
    def reset(self):
        """Reset lead agent."""
        self.research_round = 1  # number of research rounds starts from 1
        self.solution_count = 0
        self.long_term_reflection = self.external_knowledge
        self.flow_graph = None  # created at run()
        self.total_responses = 0  # Number of total responses; this can be used to track the number of LLM calls
        self.function_evals = 0  # Number of function evaluations; this is also an important metric for complexity, especially for the case when evaluation is the bottleneck
        self.valid_responses = 0 # Number of valid responses, namely responses that were successfully executed
        self.elitist = None   # Best individual so far
        self.flow_graph = None
        self.idea_agent.reset()
        self.code_agent.reset()
        self.experiment_agent.reset()
        print(f"\n>>>[LeadAgent] lead agent {self.id} reset finished.")


    def init_population(self, seed_solution: Solution=None) -> List[Solution]:
        """
        Initialize the population.
        Two sub-tasks:
        1. Complete the seed solution:
            The seed function may contains nothing, or only contains idea, or contains both idea and code;
            Complete the seed solution to make it contains idea, code, metrics, features, and score. The rest of the attributes can be left empty.
        2. `self.init_pop_size` solutions should be generated and returned.
        
        Args:
            seed_solution (Solution, optional): A seed solution. Defaults to None.
        
        Returns:
            A list of `Solution` objects.
        """            
        print("\n>>>[LeadAgent] Initializing population...")
        # we don't increase lead agent instance variable `research_round` and `solution_count`
        # but note that the total number of LLM responses and function evals are indeed increased during population initialization
        tmp_research_round = 0  # research round used for pop init
        tmp_solution_count = 0  # solution count used for pop init
        
        # =====Complete seed solution if needed=====
        if seed_solution is not None:
            # Solution idea must be present
            if seed_solution.idea is None:
                raise RuntimeError('seed solution idea is None.')
            seed_solution.lead_agent_id = self.id
            seed_solution.research_round = tmp_research_round
            seed_solution.solution_count = tmp_solution_count
            seed_solution.island_id = None
            tmp_solution_count += 1
            # If the seed solution does not contain code, complete it
            if seed_solution.code is None:
                seed_solution = self.code_agent.run(parent_solutions=None, 
                                                    long_term_reflection=self.long_term_reflection, 
                                                    solution=seed_solution,
                                                    elitist_as_root=True,  # to make use of long-term reflection
                                                    )
                # Note: code will be generated; and metrics, features, score will also be generated
            else:
                # If there is code, just evaluate
                raw_output, metrics, features, score = self.evaluator.run(seed_solution)
                self.function_evals += 1
                seed_solution.output = raw_output
                seed_solution.metrics = metrics
                seed_solution.features = features
                seed_solution.score = score
            if utils.is_valid(seed_solution):
                print("\n>>>[LeadAgent] Seed function successfully generated")

            # Update elitist
            self.elitist = seed_solution
            print(f"\n>>>[LeadAgent] Elitist updated: {self.elitist.id_str(self.algorithm)} | score:{self.elitist.score} | code path: {self.elitist.code_filepath}")
        
            # Log metrics for plot and analysis; update results.json (common for all agents)
            result_entry = {
                "iteration": tmp_research_round,
                "total_responses": self.get_total_responses(),
                "total_function_evals": self.get_function_evals(),
                "total_valid_responses": self.get_valid_responses(),
                "best_obj_overall": self.elitist.score,
                "metrics": self.elitist.metrics,
                "code_filepath": self.elitist.code_filepath,
                "output_filepath": self.elitist.output_filepath,
                #"code": self.elitist.code,
            }  # entry to add to results file
            utils.append_json_list(f"{self.output_dir}/results_detailed.json", result_entry)
            utils.append_json_list(f"{self.output_dir}/results.json", result_entry)
        
        # =====Generate `self.init_pop_size` many new solutions=====
        # Create new ideas
        print("\n>>>[LeadAgent] Generating initial population ideas...")
        ideas = self.idea_agent.run(parent_solutions=seed_solution,  # may be None if there is no seed solution
                                    long_term_reflection=self.long_term_reflection, 
                                    num_ideas=self.init_pop_size,
                                    elitist_as_root=True,  # must make use of long-term reflection
                                    )
        
        # Convert new ideas to solution instances
        # Note `island_id` is None for the initial population
        # so that each initial solution will be assigned added to ALL islands by the solution database
        # Distribute initial solutions evenly to islands
        # TODO: 
        # there are `self.num_islands` many islands in database; and there are `self.init_pop_size` many solutions;
        # we evenly distribute initial solutions to islands in a round-robin fashion
        # "how to distribute solutions over islands at database initialization" - we leave it for the user to decide and mark with a TODO flag
        solutions = []
        for i, idea in enumerate(ideas):
            island_id = i % self.num_islands  # Round-robin distribution
            solutions.append(Solution(
            lead_agent_id=self.id,
            research_round=tmp_research_round,
            solution_count=tmp_solution_count,
            idea=idea,
            island_id=island_id,
            ))
            tmp_solution_count += 1 
        
        # Generate code
        print("\n>>>[LeadAgent] Generating initial population code...")
        children_solutions = []
        for sol in solutions:
            sol = self.code_agent.run(parent_solutions=seed_solution,  # maybe None if there is no seed solution
                                    long_term_reflection=self.long_term_reflection, 
                                    solution=sol,
                                    elitist_as_root=True,  # must make use of long-term reflection
                                    )
            children_solutions.append(sol)
        
        if False:
            # Currently, we do NOT do experiments on population initialization
            # =====Conduct experiments on initial population solutions=====
            # so that each solution will be refined and summary will be generated; also, long-term reflection will be updated during this process
            print("\n>>>[LeadAgent] Conducting experiments on initial population...")
            children_solutions_updated = []
            for sol in solutions:
                sol = self.experiment_agent.run(sol, long_term_reflection=self.long_term_reflection)
                # Check solution validness
                if utils.is_valid(sol):
                    # Update long term reflection after each experiment
                    self.update_long_term_reflection(solution=sol)  
                    children_solutions_updated.append(sol)
                else:
                    print(f"\n>>>[LeadAgent] Invalid solution generated at population initialization: {sol.id_str(self.algorithm)}")
            
        # =====Prepare return=====
        # Note: include seed solution
        init_pop = [sol for sol in children_solutions if utils.is_valid(sol)]
        if seed_solution:
            init_pop.insert(0, seed_solution)
        print(f"\n>>>[LeadAgent] Population initialization finished; size of initial population: {len(init_pop)}")
        
        # =====Update elitist and update results.json for performance tracking purpose=====
        best_sol = max(init_pop, key=lambda sol: sol.score, default=None) if self.obj_type == 'max' else min(init_pop, key=lambda sol: sol.score, default=None)

        # Update elitist
        elitist_updated = False
        if (self.obj_type == 'max' and best_sol.score > self.elitist.score) or (self.obj_type == 'min' and best_sol.score < self.elitist.score):
            self.elitist = best_sol
            elitist_updated = True
            print(f"\n>>>[LeadAgent] Elitist updated: {self.elitist.id_str(self.algorithm)} | score:{self.elitist.score} | code path: {self.elitist.code_filepath}")
        
        # Log metrics for plot and analysis; update results.json (common for all agents)
        result_entry = {
            "iteration": tmp_research_round,
            "total_responses": self.get_total_responses(),
            "total_function_evals": self.get_function_evals(),
            "total_valid_responses": self.get_valid_responses(),
            "best_obj_overall": self.elitist.score,
            "metrics": self.elitist.metrics,
            "code_filepath": self.elitist.code_filepath,
            "output_filepath": self.elitist.output_filepath,
            #"code": self.elitist.code,
        }  # entry to add to results file
        utils.append_json_list(f"{self.output_dir}/results_detailed.json", result_entry)
        if elitist_updated:
            utils.append_json_list(f"{self.output_dir}/results.json", result_entry)
        
        print(f"\n>>>[LeadAgent] Initial population generation finished.")
        return init_pop

    def update_iter(self) -> None:
        """Update iteration."""
        self.research_round += 1  # increase the number of finished research rounds
        self.solution_count = 0  # count the number of solutions output for one research round
        
        # Clear long-term reflection if needed
        # We clear the long-term reflection accumulated during the previous research round so starting a research round is like hiring a new scientist to do the job
        # TODO: Optionally, we can also keep the long-term reflection accumulated during the previous research rounds
        # "should long-term reflection lasts over research rounds? should we compress it?" - we leave it for the user to decide and mark with a TODO flag
        if self.reflection_clearance_period is not None and self.research_round % self.reflection_clearance_period == 0:
            print(f"\n>>>[LeadAgent] Clearing long-term reflection as per reflection clearance period setting (every {self.reflection_clearance_period} rounds)...")
            self.long_term_reflection = self.external_knowledge  # clear long term reflection; set to be the external knowledge

        # Note: experiment summaries cache list is not cleared between research rounds
        
        # Clear flow graph
        self.flow_graph = None  # created at run()
        
        print(f"\n>>>[LeadAgent] lead agent {self.id} update_iter finished.")
    
    
    def get_total_responses(self) -> int:
        """Get the total number of responses from all agents."""
        return self.total_responses + self.idea_agent.total_responses + self.code_agent.total_responses + self.experiment_agent.total_responses
    
    def get_function_evals(self) -> int:
        """Get the total number of function evaluations from all agents."""
        return self.function_evals + self.idea_agent.function_evals + self.code_agent.function_evals + self.experiment_agent.function_evals
    
    def get_valid_responses(self) -> int:
        """Get the total number of valid responses from all agents."""
        return self.valid_responses + self.idea_agent.valid_responses + self.code_agent.valid_responses + self.experiment_agent.valid_responses

    
    def save(self, checkpoint: str):
        """
        Save checkpoint. Saved files:
        - lead_agent_state.json
        And files created by work flow, idea agent, code agent, and experiment agent.

        Args:
            checkpoint (str): checkpoint name; default None

        Return:
            None.
        """
        #checkpoint = checkpoint or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # default checkpoint name example: '2025-12-29_20-40-25'
        checkpoint_directory = f'{self.project_root}/checkpoints/{checkpoint}'
        os.makedirs(checkpoint_directory, exist_ok=True)

        # Save config
        # No need to save config file; it's already saved by ORAgent
        
        # Save state variables
        state = {
            'id': self.id,  # int
            'research_round': self.research_round,  # int
            'solution_count': self.solution_count,  # int
            'total_responses': self.total_responses,  # int
            'function_evals': self.function_evals,  # int
            'valid_responses': self.valid_responses,  # int
            'long_term_reflection': self.long_term_reflection,  # str
            'experiment_summaries': self.experiment_summaries,  # List[str]
            'elitist': dataclasses.asdict(self.elitist) if self.elitist else None,  # dict
            #'population': [dataclasses.asdict(sol) for sol in self.population]  # List[dict]
        }
        state_path = os.path.join(checkpoint_directory, 'lead_agent_state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=4, default=str)
        
        # Save flow graph data
        if self.flow_graph:
            self.flow_graph.save(checkpoint=checkpoint)
        
        # Save idea agent
        self.idea_agent.save(checkpoint=checkpoint)
        
        # Save code agent
        self.code_agent.save(checkpoint=checkpoint)
        
        # Save experiment agent
        self.experiment_agent.save(checkpoint=checkpoint)
        
        print(f"\n>>>[LeadAgent] Checkpoint saved to: {checkpoint_directory}")
    
    def load(self, checkpoint: str):
        """Load checkpoint."""
        checkpoint_directory = f'{self.project_root}/checkpoints/{checkpoint}'

        # Load config
        config_path = os.path.join(checkpoint_directory, 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load state variables
        state_path = os.path.join(checkpoint_directory, 'lead_agent_state.json')
        with open(state_path, 'r') as f:
            state = json.load(f)

        # Restore state variables
        self.id = state['id']
        self.research_round = state['research_round']
        self.solution_count = state['solution_count']
        self.total_responses = state['total_responses']
        self.function_evals = state['function_evals']
        self.valid_responses = state['valid_responses']
        self.long_term_reflection = state['long_term_reflection']
        self.experiment_summaries = state['experiment_summaries']
        self.elitist = Solution(**state['elitist']) if state['elitist'] else None
        
        # Load flow graph if flow graph file exists
        flow_graph_path = os.path.join(checkpoint_directory, 'flow_graph.json')
        if os.path.exists(flow_graph_path):
            self.flow_graph = FlowGraph(checkpoint=checkpoint)
        else:
            self.flow_graph = None
        
        # Load idea agent
        self.idea_agent = IdeaAgent(checkpoint=checkpoint)
        
        # Load code agent
        self.code_agent = CodeAgent(checkpoint=checkpoint)
        
        # Load experiment agent
        self.experiment_agent = ExperimentAgent(checkpoint=checkpoint)
        
        print(f"\n>>>[LeadAgent] Checkpoint loaded from: {checkpoint_directory}")
    
    
    def print_progress(self, file=sys.stdout) -> str:
        """Log the progress."""
        print(f"Number of research rounds: {self.research_round}", file=file)
        print(f"Solution count: {self.get_total_responses()}", file=file)
        print(f"Total number of LLM calls: {self.get_total_responses()}", file=file)
        print(f"Total number of valid responses: {self.get_valid_responses()}", file=file)
        print(f"Total number of function evaluations: {self.get_function_evals()}", file=file)
        print(f"Current best objective value: {self.elitist.score}", file=file)
        print(f"Current flow graph:", file=file)
        if self.flow_graph:
            self.flow_graph.visualize(file=file)
        else:
            print(f"None", file=file)
            
    
    def update_long_term_reflection(self, solution: Solution) -> None:
        """
        Update long-term reflection when enough experiment summaries are collected.
        
        Args:
            solution (Solution): A complete solution
            
        Returns:
            None. self.long_term_reflection is updated. 
        """
        # Append the experiment summary to the buffer
        if solution.summary:
            # Should we add more contest info? use which one?
            # - solution.summary: summary only
            # - solution.performance_summary(): performance metrics + summary
            # - solution.idea_performance_summary(): idea + performance metrics + summary
            # TODO: "add solution details for long-term reflection update?" - we leave it for the user to decide and mark with a TODO flag
            self.experiment_summaries.append(solution.performance_summary_str())
        else:
            print(f"\n>>>[LeadAgent] Warning: solution {solution.id_str(self.algorithm)} has no experiment summary; cannot be used for long-term reflection update.")
            return
        
        # Update long term reflection if enough experiment summaries are collected
        if len(self.experiment_summaries) >= self.reflection_period:
            print(f"\n>>>[LeadAgent] After solution {solution.id_str(self.algorithm)}, updating long-term reflection since {self.reflection_period} solutions have been collected...")
            
            # Concatenate experiment summaries
            experiment_summaries_str = ''
            for i, summary in enumerate(self.experiment_summaries):
                experiment_summaries_str += f"{'-'*40}\nExperiment #{i + 1}:\n{'-'*40}\n{summary}\n\n"
            experiment_summaries_str = experiment_summaries_str.strip()
            
            # Word limit            
            if self.reflection_compression is None:
                other_context = ""  # no compression; no other context needed
            else:
                other_context = f"""The newly generated long-term research reflection can have at most {self.reflection_compression} words."""
            
            # Construct prompt
            user = self.user_long_term_reflection_prompt.format(
                problem_description = self.problem_description,  # common
                function_to_evolve = self.function_to_evolve,  # common
                obj_type = self.obj_type,  # common
                function_description = self.function_description,  # common
                prior_reflection = self.long_term_reflection if self.long_term_reflection else "(empty)",  # common
                experiment_summaries_str = experiment_summaries_str,
                #new_solution = str(solution),  # str representation of the newly generated solution; its performance, metrics, and most importantly, the summary, can help the agent to accumulate long-term knowledge.
                other_context = other_context,  # context for compression if needed; say limit the length of new long-term reflection
                )
            messages = [{"role": "system", "content": self.system_long_term_reflection_prompt}, {"role": "user", "content": user}]
            
            response = self.llm_client.chat(messages)
            self.total_responses += 1
            
            response_extracted = utils.extract_json(response)
            if response_extracted and 'reflection' in response_extracted:
                new_reflection = response_extracted['reflection']
                self.long_term_reflection = new_reflection
                self.valid_responses += 1
                print(f"\n>>>[LeadAgent] After solution {solution.id_str(self.algorithm)}, long-term reflection updated to:\n{self.long_term_reflection}")
                
                # Log long_term_reflection for webui
                with open(f"{self.output_dir}/long_term_reflection.txt", 'w') as file:
                    file.write(self.long_term_reflection)
            else:
                # invalid response; long-term reflection is not updated
                print(f"\n>>>[LeadAgent] Failed to update long-term reflection for solution {solution.id_str(self.algorithm)};\nResponse: {response}")
        
            # Log long-term reflections
            file_name = f"{self.output_dir}/details/long_term_reflection_{solution.id_str(self.algorithm)}.txt"
            with open(file_name, 'w') as file:
                file.write(self.long_term_reflection)

            # Clear experiment summaries buffer
            self.experiment_summaries = []
        
    
    def run(self, solution_database: SolutionDatabase) -> List[Solution]:
        """
        Lead agent runs one research round.
        
        Args:
            solution_database (SolutionDatabase): Solution database to sample solutions from
            
        
        Returns:
            solutions (List[Solution]): list of new solutions generated.
        """
        print(f"\n>>>[LeadAgent] Lead agent {self.id} research round {self.research_round} starts...")
        
        # Update elitist
        self.elitist = solution_database.get_best()
        
        # ====1. Research Initialization====        
        # Sample solutions from solution database
        # `parent_solutions` is the starting point for the research round
        
        # -----Sample root solutions-----
        # Exploitation vs exploration tradeoff
        # How to make use of elitist?
        # For the first round of research, we use elitist for fast exploitation at the beginning of research
        # for later stages, we just invoke solution_database.sample()
        # TODO: should we exploit elitist more? say use elitist as root_solution every `elitist_adopt_cycle` rounds?
        # note one round of research takes hours and generate tens of solutions
        # "how to exploit elitist?" - we leave it for the user to decide and mark with a TODO flag
        # use elitist to start research every so many rounds
        # lead agent starts with round 1; so we test with (self.research_round - 1); 
        # hence the first round will always exploit elitist unless `elitist_as_root_period` is set to 0
        if self.elitist_as_root_period > 0 and (self.research_round - 1) % self.elitist_as_root_period == 0:
            elitist_as_root = True
            # self.config['elitist_as_root_period'] == 0 means never use elitist as root
            # here when we use elitist to start the research round
            # optionally, we can use elitist + (num_parents - 1) many other solutions
            # so that we can still have diversity at the root node
            # TODO: "should we always use elitist + other solutions as root?" - we leave it for the user to decide and mark with a TODO flag
            root_solutions = [self.elitist]
            # Alternative
            #root_solutions= []
            #root_solutions.extend(solution_database.sample(num_parents=self.num_parents - 1))
            #root_solutions.append(self.elitist)
            print(f"\n>>>[LeadAgent] Lead agent {self.id} using elitist to start research round {self.research_round}...")
        else:
            elitist_as_root = False
            root_solutions = solution_database.sample(num_parents=self.num_parents)
        
        if isinstance(root_solutions, Solution):
            root_solutions = [root_solutions]
        
        # Sort root_solutions; from worst to best
        if self.obj_type == 'max':
            root_solutions = sorted(root_solutions, key=lambda x: x.score)
        else:
            root_solutions = sorted(root_solutions, key=lambda x: x.score, reverse=True)
        
        # -----Create flow_graph by passing parent_solutions to create the root node-----
        # Only the root node can have a list of solutions; other node wraps a single solution
        self.flow_graph = FlowGraph(root_solution=root_solutions, config=self.config)
        
        # parent solutions' island id
        # we need to know the island id to know which island the solutions belong to so that we can insert newly generated child solutions back to the same island
        island_id = None
        if root_solutions:    
            island_id = root_solutions[0].island_id
        print(f"\n>>>[LeadAgent] Island id: {island_id}")
        
        # -----Visualize current flow graph-----
        print("\n>>>[LeadAgent] First flow graph:\n")
        self.flow_graph.visualize()
        # Log flow graph after best leaf obtained
        # and name the flow graph by the current best leaf that is being extended
        file_name = f"{self.output_dir}/details/flow_graph_lead{self.id}_round{self.research_round}_start.txt"
        with open(file_name, 'w') as file:
            self.flow_graph.visualize(file=file)
            
        # Log progress to progress.txt for webui
        with open(f"{self.output_dir}/progress.txt", 'w') as file:
            self.print_progress(file=file)
                
        
        # =====1. Research Start Process=====
        # While loop to generate child nodes for root node are generated; at least one valid child node should be generated for root node 
        attempt = 0          
        while True:
            attempt += 1
            # -----Generate ideas-----
            # altogether, ideas can form a comprehensive research plan, investigating different directions
            print(f"\n>>>[LeadAgent] Generating ideas at research round {self.research_round} start...")
            if elitist_as_root:
                # when elitist is used as root solution, we may generate more ideas to explore around elitist
                num_ideas = int(self.num_children * self.elitist_enlargement_factor)
                # This is what ReEvo does - do not use long-term reflection when doing short-term reflection and cross-overing on random parent pairs
                # long-term reflection is only used for extending elitist
                # TODO: "should we use long-term reflection when doing crossovering" - we leave it for the user to decide and mark with a TODO flag
            else:
                num_ideas = self.num_children
            
            print(f"\n>>>[LeadAgent] Generating {num_ideas} ideas...")
            ideas = self.idea_agent.run(parent_solutions=root_solutions, 
                                        long_term_reflection=self.long_term_reflection, 
                                        num_ideas=num_ideas,
                                        elitist_as_root=elitist_as_root)
            
            # Check number of ideas returned
            if len(ideas) != num_ideas:
                print(f"\n>>>[LeadAgent] Warning: idea agent returned {len(ideas)} ideas; expected {num_ideas} ideas.")
            num_ideas_valid = min(len(ideas), num_ideas)
            ideas = ideas[:num_ideas_valid]  # trim ideas if more than needed
            
            # Convert ideas to `Solution` instances
            solutions = []
            for idea in ideas:
                solutions.append(Solution(
                lead_agent_id=self.id,
                research_round=self.research_round,
                solution_count=self.solution_count,
                idea=idea,
                island_id=island_id  # Note: need to set island id
                ))
                self.solution_count += 1
            
            # -----Generate code-----
            print(f"\n>>>[LeadAgent] Generating child nodes for root at research round {self.research_round} start...")
            # iterate over ideas that have been generated, generate code for each idea
            solutions_updated = []
            for sol in solutions:
                sol = self.code_agent.run(parent_solutions=root_solutions, 
                                        long_term_reflection=self.long_term_reflection, 
                                        solution=sol)
                solutions_updated.append(sol)
                
            # -----Conduct experiments on each child node-----
            print(f"\n>>>[LeadAgent] Conducting experiments at research round {self.research_round} starting stage...")
            children_solutions = []
            for sol in solutions_updated:
                sol = self.experiment_agent.run(solution=sol, 
                                                long_term_reflection=self.long_term_reflection)

                # Here we add an additional solution debugging step
                if not sol.score:
                    sol = self.code_agent.debug(sol)
                
                # Check solution validness
                if utils.is_valid(sol):
                    children_solutions.append(sol)
                    # Update long term reflection after each experiment
                    self.update_long_term_reflection(solution=sol)
                    # Note we update long-term reflection after each experiment that generated valid solutions; 
                    # TODO: Questions: 1) should we also experiments that got invalid code but since they may still shed some lights?
                    # 2) how frequent should we update long-term reflection?
                    # (three places in this script we face the same questions
                    # - initialization
                    # - research round start
                    # - research round loop running)
                    # there are many options;
                    # "do failed experiments matter?" - we leave it for the user to decide and mark with a TODO flag
                else:
                    print(f"\n>>>[LeadAgent] Invalid solution generated at population initialization: {sol.id_str(self.algorithm)}")
            
            # -----Update elitist if a new best solution is found for performance tracking purpose-----
            best_child = max(children_solutions, key=lambda sol: sol.score, default=None) if self.obj_type == 'max' else min(children_solutions, key=lambda sol: sol.score, default=None)
            if best_child:
                # Update elitist
                elitist_updated = False
                if (self.obj_type == 'max' and best_child.score > self.elitist.score) or (self.obj_type == 'min' and best_child.score < self.elitist.score):
                    self.elitist = best_child
                    elitist_updated = True
                    print(f"\n>>>[LeadAgent] Elitist updated: {self.elitist.id_str(self.algorithm)} | score:{self.elitist.score} | code path: {self.elitist.code_filepath}")
                    
                # Log metrics for plot and analysis; update results.json
                result_entry = {
                    "iteration": self.research_round,
                    "total_responses": self.get_total_responses(),
                    "total_function_evals": self.get_function_evals(),
                    "total_valid_responses": self.get_valid_responses(),
                    "best_obj_overall": self.elitist.score,
                    "metrics": self.elitist.metrics,
                    "code_filepath": self.elitist.code_filepath,
                    "output_filepath": self.elitist.output_filepath,
                    #"code": self.elitist.code,
                }  # entry to add to results file
                utils.append_json_list(f"{self.output_dir}/results_detailed.json", result_entry)
                if elitist_updated:
                    utils.append_json_list(f"{self.output_dir}/results.json", result_entry)
            
            # -----Add children nodes to flow_graph as root's children-----
            # Convert `Solution`s to `Node`s
            children_nodes = [Node(solution=sol) for sol in children_solutions]       
            # Add nodes to flow_graph as children of root node
            # Note that those children nodes are not guaranteed to have performances better than the root node(s)!
            # we intended so because when we explore a new research direction, we allow temporary performance downgrades
            # TODO: of course, there are alternatives like controlling how bad the children could be at most
            # "how to define promising directions?" - we leave it for the user to decide and mark with a TODO flag
            if len(children_nodes) > 0:
                self.flow_graph.add(self.flow_graph.root, children_nodes)
                print(f"\n>>>[LeadAgent] lead agent {self.id} constructed flow graph; number of children of root: {len(children_nodes)}")
                break  # if at least one valid child is generated, then break out of loop; otherwise, retry
            else:
                if attempt >= 5:
                    raise RuntimeError(f">>>[LeadAgent] Max attempts reached for research start process")
                print(f"\n>>>[LeadAgent] lead agent {self.id} failed to generate valid children at research start; retrying...")
        
        
        # =====2. Research Loop=====
        while True:
            print(f"\n>>>[LeadAgent] Lead agent {self.id} research round {self.research_round} loop starts...")
            
            # -----Research finishing test-----
            if self.flow_graph.check_research_finished():
                print(f"\n>>>[LeadAgent] One research round finished.")
                break
            
            # -----Find the best leaf to extend-----
            # best leaf represent the most promising research direction; we will extend this leaf first
            # at the start of research, best_leaf is the root
            best_leaf = self.flow_graph.get_best_undone_leaf()
            
            # if there is constraint on max tree depth, and best leaf already reaches max depth, we stop extending this leaf
            if self.max_tree_depth and best_leaf.depth >= self.max_tree_depth:
                best_leaf.is_done = True
                print(f"\n>>>[LeadAgent] Best leaf {best_leaf.solution.id_str(self.algorithm)} reached max tree depth {self.max_tree_depth}; marked as done.")
                continue  # go to next iteration to find another best leaf
            
            # if there is a best undone leaf that has depth less than max depth, we extend it
            print(f"\n>>>[LeadAgent] Lead agent {self.id} research round {self.research_round} extends research tree | current best leaf: {best_leaf.solution.id_str(self.algorithm)} | score: {best_leaf.solution.score}")
            
            # -----Visualize current flow graph-----
            print("\n>>>[LeadAgent] Current flow graph:\n")
            self.flow_graph.visualize()
            # Log flow graph after best leaf obtained
            # and name the flow graph by the current best leaf that is being extended
            file_name = f"{self.output_dir}/details/flow_graph_{best_leaf.solution.id_str(self.algorithm)}.txt"
            with open(file_name, 'w') as file:
                self.flow_graph.visualize(file=file)
                
            # Log progress to progress.txt for webui
            with open(f"{self.output_dir}/progress.txt", 'w') as file:
                self.print_progress(file=file)
            
            # -----Generate children for the best leaf-----
            # altogether, ideas can form a comprehensive research plan, investigating different directions
            print(f"\n>>>[LeadAgent] Generating new ideas to extend the best leaf...")
            ideas = self.idea_agent.run(parent_solutions=best_leaf.solution, 
                                        long_term_reflection=self.long_term_reflection, 
                                        num_ideas=self.num_children)
            
            # Convert ideas to `Solution` instances
            solutions = []
            for idea in ideas:
                solutions.append(
                    Solution(idea=idea, 
                            lead_agent_id=self.id,
                            research_round=self.research_round,
                            solution_count=self.solution_count,
                            island_id=island_id  # Note: we need to set island id
                ))
                self.solution_count += 1
                
            # -----Generate code for ideas-----
            print(f"\n>>>[LeadAgent] Generating code for new ideas...")
            solutions_updated = []
            for sol in solutions:
                sol = self.code_agent.run(parent_solutions=best_leaf.solution, 
                                        long_term_reflection=self.long_term_reflection,
                                        solution=sol)
                solutions_updated.append(sol)
                
            # -----Conduct experiments for each child solution-----
            children_solutions = []
            for i, sol in enumerate(solutions_updated):
                print(f"\n>>>[LeadAgent] Conducting experiment for {i}-th solution ({sol.id_str(self.algorithm)})...")
                sol = self.experiment_agent.run(solution=sol, 
                                                long_term_reflection=self.long_term_reflection)
                
                # Check solution validness
                if utils.is_valid(sol):
                    # Update long term reflection after each experiment
                    self.update_long_term_reflection(solution=sol)
                    children_solutions.append(sol)
                else:
                    print(f"\n>>>[LeadAgent] Invalid solution generated: {sol.id_str(self.algorithm)}")
            
            # -----Keep only children with score better than `best_leaf`-----
            # these remaining children represent the performance ascending directions; 
            # we will add these nodes to research tree so that we can explore these directions later
            # TODO: there are other options like keeping only the top N best solutions
            # "what kind of child solutions are kept for later extension?" - we leave it for the user to decide and mark with a TODO flag            
            if self.obj_type == 'max':
                children_solutions = [sol for sol in children_solutions if sol.score > best_leaf.solution.score]
            else:
                children_solutions = [sol for sol in children_solutions if sol.score < best_leaf.solution.score]
            
            # -----Update elitist if a new best solution is found for performance tracking purpose-----
            best_child = max(children_solutions, key=lambda sol: sol.score, default=None) if self.obj_type == 'max' else min(children_solutions, key=lambda sol: sol.score, default=None)
            if best_child:
                # Update elitist
                elitist_updated = False
                if (self.obj_type == 'max' and best_child.score > self.elitist.score) or (self.obj_type == 'min' and best_child.score < self.elitist.score):
                    self.elitist = best_child
                    elitist_updated = True
                    print(f"\n>>>[LeadAgent] Elitist updated: {self.elitist.id_str(self.algorithm)} | score:{self.elitist.score} | code path: {self.elitist.code_filepath}")
                    
                # Log metrics for plot and analysis; update results.json (common for all agents)
                result_entry = {
                    "iteration": self.research_round,
                    "total_responses": self.get_total_responses(),
                    "total_function_evals": self.get_function_evals(),
                    "total_valid_responses": self.get_valid_responses(),
                    "best_obj_overall": self.elitist.score,
                    "metrics": self.elitist.metrics,
                    "code_filepath": self.elitist.code_filepath,
                    "output_filepath": self.elitist.output_filepath,
                    #"code": self.elitist.code,
                }  # entry to add to results file
                utils.append_json_list(f"{self.output_dir}/results_detailed.json", result_entry)
                if elitist_updated:
                    utils.append_json_list(f"{self.output_dir}/results.json", result_entry)
                    
            # -----Check whether the leaf node is done-----
            if not children_solutions:
                # If all children are worse than `best_leaf`, mark `best_leaf` as done - approximate local optimum has been reached
                best_leaf.is_done = True  # Note: set `Node`` field not `Solution` field!
                print(f"\n>>>[LeadAgent] Best leaf is done: {best_leaf.solution.id_str(self.algorithm)}")
            else:
                # else, add children to tree - performance ascending directions have been identified
                children_nodes = [Node(solution=sol) for sol in children_solutions]  # solutions to nodes
                self.flow_graph.add(parent_node=best_leaf, children_nodes=children_nodes)
                children_str = ', '.join([str(child.solution.id) for child in children_nodes])
                print(f"\n>>>[LeadAgent] Best leaf is successfully extended: {children_str}")
                
        
        # =====3. Prepare Return=====
        # get list of all leaf nodes
        # these leaves are `done`, which means they are approximate local optimum
        # return them so that they will be added to the solution database
        leaves = self.flow_graph.get_leaves()
        # contract out `Solution` instances
        solutions = [leaf.solution for leaf in leaves]  
        # TODO: filtering mechanism could be added here
        # like, should we filter solutions inferior to root node's solutions here?
        # we didn't apply any filtering here
        # "what kind of child solutions are 'good' enough to be added to database?" - we leave it for the user to decide and mark with a TODO flag
        
        print(f"\n>>>[LeadAgent] Lead agent {self.id} research round finished | number of solutions to return: {len(solutions)} (IDs: {', '.join([str(sol.id) for sol in solutions])})")
        
        print(f"\n>>>[LeadAgent] Final flow graph:")
        self.flow_graph.visualize()
        # Log flow graph
        file_name = f"{self.output_dir}/details/flow_graph_lead{self.id}_round{self.research_round}_done.txt"
        with open(file_name, 'w') as file:
            self.flow_graph.visualize(file=file)
            
        # Log progress to progress.txt for webui
        with open(f"{self.output_dir}/progress.txt", 'w') as file:
            self.print_progress(file=file)
                
        return solutions


        


if __name__ == '__main__':
    # For debugging purposes
    # TODO:
    pass