# src/oragent/funsearch.py
"""
A single-threaded implementation of the FunSearch pipeline.

Original paper:
 - Romera-Paredes, B., Barekatain, M., Novikov, A., Balog, M., Kumar, M.P., Dupont, E., Ruiz, F.J., Ellenberg, J.S., Wang, P., Fawzi, O. and Kohli, P., 2024. Mathematical discoveries from program search with large language models
Original code:
 - https://github.com/google-deepmind/funsearch/tree/main
This algorithm has been re-implemented in this project for compatibility, but the core functionality remains the same.
PS: This implementation of FunSearch is extended to support evolving the whole module instead of just a single function as in the original implementation.



## Original FunSearch pipeline
main() function sets up the FunSearch pipeline by:
    1. Extracting the names of the function to evolve and the function to run from the specification.
    2. Parsing the specification code into a structured Program object.
    3. Initializing a ProgramsDatabase to store and manage different versions of the function being evolved.
    4. Setting up multiple evaluators to assess the quality of generated function implementations.
    5. Setting up multiple samplers to generate new function implementations.
    6. Starting the main evolution loops, where each sampler enters an infinite loop to generate
         and evaluate new function implementations.

Pseudo code (from AlphaEvolve paper) to help understand the process:
parent_program, inspirations = database.sample()
prompt = prompt_sampler.build(parent_program, inspirations)
diff = llm.generate(prompt)
child_program = apply_diff(parent_program, diff)
results = evaluator.execute(child_program)
database.add(child_program, results)
"""
import os
import sys
import json
from pathlib import Path
import time
from datetime import datetime
import shutil
import dataclasses
import yaml
from oragent.evaluator import Evaluator
from oragent.solution_database import SolutionDatabase
import oragent.utils as utils
from oragent.utils import Solution



class FunSearch:
    def __init__(self, checkpoint=None, config=None)-> None:
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
        
        # =====Create LLM client=====
        self.client = utils.LLMClient(config=self.config, 
                                    llm_provider=self.config['model']['llm_provider'], 
                                    model_name=self.config['model']['model_name'])
        self.llm_temperature = self.config['model']['temperature']
        
        # =====Create evaluator=====
        self.evaluator = Evaluator(config=self.config)
        # Initializes a ProgramsDatabase to store and manage different versions of the function being evolved.
        
        # =====Create solution database=====
        # create solution database if checkpoint not specified
        if not checkpoint:
            self.database = SolutionDatabase(config=self.config)
        
        # =====Vars updated during agent running=====
        # if no checkpoint specified, we need to create them
        if not checkpoint:  
            self.iteration = 0  # number of evolution rounds
            self.total_responses = 0  # Number of total responses; this can be used to track the number of LLM calls
            # Updated after `response_to_individual`, `responses_to_population` calls
            self.function_evals = 0  # Number of function evaluations; this is also an important metric for complexity, especially for the case when evaluation is the bottleneck
            # Updated after `evaluate_population` calls
            self.valid_responses = 0 # Number of valid responses, namely responses that were successfully executed
            # Updated after `evaluate_population` calls
            self.elitist = None   # Best individual so far; `Solution` instance
            #self.long_term_reflection_str = ""  # long term reflection
            #self.population = None  # population so far; List of `Solution` instances
        
        # =====FunSearch settings=====
        self.init_pop_size = self.config['init_pop_size']
        self.pop_size = self.config['pop_size']
        self.max_evolutions = self.config['max_evolutions']
        self.autosave_interval_minutes = self.config['autosave_interval_minutes']
        
        # =====Loading all text prompts=====
        self.problem_dir = f"{self.project_root}/problems/{self.problem}"
        self.prompt_dir = f"{self.project_root}/prompts"
        self.output_dir = self.config['output_dir'] or f"{self.project_root}/outputs/{self.algorithm}/{self.problem}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/details", exist_ok=True)  # folder to store details
        self.problem_description = utils.file_to_string(f'{self.problem_dir}/problem_description.txt')
        self.function_description = utils.file_to_string(f'{self.problem_dir}/function_description.txt')
        self.seed_function = utils.file_to_string(f'{self.problem_dir}/seed_solution.py')
        # Common prompts
        self.system_generator_prompt = utils.file_to_string(f'{self.prompt_dir}/system_generator.txt')
        self.crossover_prompt = utils.file_to_string(f'{self.prompt_dir}/crossover_funsearch.txt')
        self.init_with_seed_prompt = utils.file_to_string(f'{self.prompt_dir}/init_with_seed.txt')
        
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
        print("\n>>>[FunSearch] FunSearch initialized")

    
    def init_population(self) -> None:
        """ 
        1. Evaluate the seed function, and set it as Elite
        2. Create the initial population
        """     
        # =====Load and evaluate seed function=====
        # Evaluate the seed function, and set it as Elite
        print("\n>>>[FunSearch] Evaluating seed function...")
        #print("Seed function code: \n" + code)
        self.seed_ind = utils.response_to_individual(response=utils.wrap_python_code(self.seed_function), 
                                                     iteration=self.iteration, 
                                                     response_id=0, 
                                                     output_dir=self.output_dir)
        # by invoking self.evaluator.evaluate_population(), we will also save the seed function to `self.population`
        self.seed_ind = self.evaluator.evaluate_individual(self.seed_ind)  # Evaluate seed function
        self.function_evals += 1
        
        # If seed function is invalid, stop
        if not self.seed_ind.score:
            raise RuntimeError(f"Seed function is invalid. Please check the stdout file in {os.getcwd()}.")
        
        # Now the population has a single individual, which is the seed function
        self.database.add(self.seed_ind)
        
        self.update_iter()
        
        # =====Generate initial population=====
        system = self.system_generator_prompt
        # `user` is a prompt including system prompt, seed function, and long-term reflection;
        # it's asking llm to generate a new function
        user = self.init_with_seed_prompt.format(
            problem_description=self.problem_description,
            function_description=self.function_description,
            seed_solution=utils.individual_to_str(self.seed_ind),
            function_to_evolve=self.function_to_evolve,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        #print("Initial Population Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)

        # Generate initial population (num = self.cfg.init_pop_size)
        # Note: multiple (num = self.cfg.init_pop_size) samples are generated in parallel
        responses = self.client.multi_chat([messages], self.init_pop_size, temperature = self.llm_temperature + 0.3) # Increase the temperature for diverse initial population
        # extract function samples from responses;
        # `population` is a tmp list
        population = utils.responses_to_population(responses, self.iteration, self.output_dir)
        self.total_responses += len(population)
        
        # Run code and evaluate population
        # by invoking self.evaluator.evaluate_population(), we will get a dict
        population = self.evaluator.evaluate_population(population)
        self.function_evals += len(population)
        population = [ind for ind in population if ind.score]
        self.valid_responses += len(population)
        
        self.database.add(population)
        self.update_iter()
        print(f"\n>>>[FunSearch] Population initialization done. Population size: {len(population)}")
    
    
    def update_iter(self) -> None:
        """
        Update after each iteration, including:
        - update the best code and objective so far and save
        Note: common for ReEvo, EoH, AEL, FunSearch
        """
        print("\n>>>[FunSearch] Updating iteration...")
        best_ind = self.database.get_best()
        # newly generated solutions are directly added to database
        # hence database best sol is no worse than self.elitist
        # we need to check if the newly generated solution is better than the previous best solution

        # Update elitist
        elitist_updated = False
        if self.elitist is None or (self.obj_type == "min" and best_ind.score < self.elitist.score) or ( self.obj_type == "max" and best_ind.score > self.elitist.score):
            # Best sol has been updated
            # that means a better solution has been found
            self.elitist = best_ind
            elitist_updated = True
            print(f"\n>>>[FunSearch] Elitist updated: {self.elitist.id_str(self.algorithm)} | score:{self.elitist.score} | code path: {self.elitist.code_filepath}")
            
        # Log metrics for plot and analysis; update results.json (common for all agents)
        result_entry = {
            "iteration": self.iteration,
            "total_responses": self.total_responses,
            "total_function_evals": self.function_evals,
            "total_valid_responses": self.valid_responses,
            "best_obj_overall": self.elitist.score,
            "metrics": self.elitist.metrics,
            "code_filepath": self.elitist.code_filepath,
            "output_filepath": self.elitist.output_filepath,
            #"code": self.elitist.code,
        }  # entry to add to results file
        
        # record elitist for every iteration in results_detailed.json
        utils.append_json_list(f"{self.output_dir}/results_detailed.json", result_entry)
        # results.json only record changes
        if elitist_updated:
            utils.append_json_list(f"{self.output_dir}/results.json", result_entry)
            
        self.iteration += 1  # Note: self.iteration increase by 1 each time you invoke update_iter()!
        print(f"\n>>>[FunSearch] Iteration {self.iteration} finished...")
    
    def print_progress(self, file=sys.stdout) -> str:
        """Log the progress. Common for ReEvo, EoH, AEL, FunSearch"""
        print(f"Iteration: {self.iteration}", file=file)
        print(f"Total number of LLM calls: {self.total_responses}", file=file)
        print(f"Total number of valid responses: {self.valid_responses}", file=file)
        print(f"Total number of function evaluations: {self.function_evals}", file=file)
        print(f"Current best objective value: {self.elitist.score}", file=file)


    def save(self, checkpoint: str=None):
        """
        Save checkpoint. Saved files:
        - config.yaml
        - state.json
        - database.json

        Args:
            checkpoint (str): checkpoint name; default None

        Return:
            None.
        """
        checkpoint = checkpoint or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # default checkpoint name example: '2025-12-29_20-40-25'
        checkpoint_directory = f'{self.project_root}/checkpoints/{checkpoint}'
        os.makedirs(checkpoint_directory, exist_ok=True)

        # Save config
        config_path = os.path.join(checkpoint_directory, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # Save state variables
        state = {
            'iteration': self.iteration,  # int
            'total_responses': self.total_responses,  # int
            'function_evals': self.function_evals,  # int
            'valid_responses': self.valid_responses,  # int
            #'long_term_reflection_str': self.long_term_reflection_str,  # str
            'elitist': dataclasses.asdict(self.elitist) if self.elitist else None,  # dict
            #'population': [dataclasses.asdict(sol) for sol in self.population]  # List[dict]
        }
        state_path = os.path.join(checkpoint_directory, 'state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=4, default=str)
            
        # Save solution database
        self.database.save(checkpoint=checkpoint)
        
        # Save results.json
        # we need to save a copy of results.json instead of directly append to the existing one in output dir
        # since the existing one may be ahead of the checkpoint in terms of iterations
        results_source = f"{self.output_dir}/results.json"
        results_dest = os.path.join(checkpoint_directory, 'results.json')
        if os.path.exists(results_source):
            shutil.copy2(results_source, results_dest)
        
        print(f"\n>>>[FunSearch] Checkpoint saved to: {checkpoint_directory}")
    
    def load(self, checkpoint):
        """Load checkpoint."""
        checkpoint_directory = f'{self.project_root}/checkpoints/{checkpoint}'

        # Load config
        config_path = os.path.join(checkpoint_directory, 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load state variables
        state_path = os.path.join(checkpoint_directory, 'state.json')
        with open(state_path, 'r') as f:
            state = json.load(f)

        # Restore state variables
        #self.long_term_reflection_str = state['long_term_reflection_str']
        self.iteration = state['iteration']
        self.total_responses = state['total_responses']
        self.function_evals = state['function_evals']
        self.valid_responses = state['valid_responses']
        self.elitist = Solution(**state['elitist']) if state['elitist'] else None
        #self.population = [Solution(**sol_dict) for sol_dict in state['population']]

        # Restore solution database
        self.database = SolutionDatabase(checkpoint=checkpoint)
        
        # Restore results.json
        results_source = os.path.join(checkpoint_directory, 'results.json')
        self.algorithm = self.config['algorithm'].lower().strip()
        self.problem = self.config['problem'].lower().strip()
        self.output_dir = self.config['output_dir'] or f"{self.project_root}/outputs/{self.algorithm}/{self.problem}"
        results_dest = f"{self.output_dir}/results.json"
        if os.path.exists(results_source):
            shutil.copy2(results_source, results_dest)
            
        # print out checkpoint info
        print(f"algorithm:", self.config['algorithm'])
        print(f"problem:", self.config['problem'])
        print(f"Iteration:", self.iteration)
        print(f"function_evals:", self.function_evals)
        print(f"valid_responses:", self.valid_responses)
        print(f"elitst score:", self.elitist.score)
        #print(f"population size:", len(self.population))
        print(f"population size:", len(self.database))
        #print(f"long-term reflection:", self.long_term_reflection_str)
        print(f"\n>>>[FunSearch] Checkpoint loaded from: {checkpoint_directory}")
        
    
    def crossover(self, parents: list[Solution]) -> list[Solution]:
        ''' 
        Given a selected population, first generate pairs
        then for each pair of parents, perform crossover, and generate a child
        
        Args:
            population: list[dict]; selected population; size: 2*pop_size
            
        Returns:
            crossed_population: list[dict]; generated crossed population; size: pop_size
        '''        
        messages_lst = []
        parent_1 = parents[0]
        if len(parents) == 2:
            parent_2 = parents[1]
        else:
            parent_2 = None
            print(f"\n>>>[FunSearch] Warning: Only one parent is selected for crossover.")
        
        # construct crossover prompt
        user = self.crossover_prompt.format(
            #alg_desc1 = parent_1["description"],  # description is not needed now; they are added to the docstring
            #alg_desc2 = parent_2["description"],
            problem_description=self.problem_description,
            function_description=self.function_description,
            function_to_evolve=self.function_to_evolve,
            solution1=utils.individual_to_str(parent_1),
            solution2=utils.individual_to_str(parent_2) if parent_2 else "(empty)",
        )
        messages = [{"role": "user", "content": user}]
        messages_lst.append(messages)
            
        # Multi-processed chat completion
        # Asynchronously generate responses
        # Note: if len(messages_lst) is too large, LLM api may support unless we handle it in multi_chat()
        responses_lst = self.client.multi_chat(messages_lst)  
        # TODO: here we only generate one child for a pair of parents; we could generate multiple children for a pair of parents
        # "funsearch crossover generates multiple children" - This requires further exploration, analysis, and validation, and is marked with a TODO flag
        crossed_population = utils.responses_to_population(responses_lst, self.iteration, self.output_dir)
        self.total_responses += len(crossed_population)

        assert len(crossed_population) == 1
        return crossed_population
    
    
    def run(self):
        """
        Launches a FunSearch experiment.
        Continuously gets prompts, samples programs, sends them for analysis.
        """
        last_save_time = time.time()
        
        while self.total_responses <= self.max_evolutions or self.function_evals <= self.max_evolutions:
            # =====Sample parents from the database=====
            print("\n>>>[FunSearch] Selecting population...")
            parents = self.database.sample(num_parents=2)
            island_id = parents[0].island_id
            
            # =====Generate offsprings=====
            print("\n>>>[FunSearch] Crossovering...")
            crossed_population = self.crossover(parents)
            
            # Evaluate the generated crossover offsprings
            population = self.evaluator.evaluate_population(crossed_population)
            self.function_evals += len(population)
            # Filter out invalid programs
            population = [ind for ind in population if ind.score]
            self.valid_responses += len(population)
            
            # =====Add offsprings to database=====
            # Set island id for child individuals
            for idv in population:
                idv.island_id = island_id
            
            # Add to database
            if population:
                self.database.add(population)
                self.update_iter()
            
            # Log progress to progress.txt for webui real-time visualization
            with open(f"{self.output_dir}/progress.txt", 'w') as file:
                self.print_progress(file=file)
                print(f"Just finished one round of evolution", file=file)
                
            # =====Autosaving=====
            if time.time() - last_save_time >= self.autosave_interval_minutes * 60:
                self.save()
                last_save_time = time.time()