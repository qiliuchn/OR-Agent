# src/oragent/eoh.py
""" 
# EoH: Evolution of Heuristics

See paper:
 - Liu, F., Tong, X., Yuan, M., Lin, X., Luo, F., Wang, Z., Lu, Z. and Zhang, Q., 2024. Evolution of heuristics: Towards efficient automatic algorithm design using large language model
Original Github repo:
 - https://github.com/FeiLiu36/EoH
ReEvo implementation:
 - https://github.com/ai4co/reevo/tree/main/baselines/eoh
This algorithm has been re-implemented in this project for compatibility, but the core functionality remains the same.


## EOH process
 N: population size
 
 - Step 0: Initialization: Initialize the population
 
 - Step 1 Generation of Heuristics: If the stopping condition is not met, 
    **Five Evolution prompt strategies** (detailed in Section 3.4) are used simultaneously to generate 5*N new heuristics. 
    
    For each of the five prompt strategies, repeat the following process N times:
     • Step 1.1: Select parent heuristic(s) from the current population to construct a prompt for the strategy.
     • Step 1.2: Request LLM to generate a new heuristic as well as its corresponding code implementation.
     • Step 1.3: Evaluate the new heuristic on a set of evaluation instances to determine its fitness value.
     • Step 1.4: Add the new heuristic to the current population if the heuristic and code are feasible.

 - Step 2 Population Management: Select the N best individual heuristics from the current population to form a
population for the next generation. Go to Step 1.



## Five Prompt strategies
E1: Generate new heuristics that are as much different as
possible from parent heuristics. First, p heuristics are selected
from the current population. Then, LLM is prompted
to design a new heuristic that is different from these selected
heuristics as much as possible in order to explore new ideas.

E2: Explore new heuristics that share the same idea as the
selected parent heuristics. First, p heuristics are selected
from the current population. Then, LLM is instructed to
identify common ideas behind these heuristics. Then, a new
heuristic is designed that are based the common ideas but
are as much different as possible from the selected parents
by introducing new parts.

M1: Modify one heuristic for better performance. Firstly,
one heuristic is selected from the population. Then, LLM is
prompted to modify it to produce a new heuristic.

M2: Modify the parameters of one selected heuristic. First,
one heuristic is selected from the current population. Then,
LLM is prompted to try different parameters in the current
heuristic instead of designing a new one.

M3: Simplify heuristics by removing redundant components.
First, one heuristic is selected from the current population.
Then, LLM is prompted to analyze and identify
the main components in the selected heuristic and analyze
whether there are any redundant components. Finally, LLM
is prompted to simplify the code implementation of the
heuristic based on its analysis.
        


## Hyperparameters
In the original paper, the hyperparameters are:
 - Population size: 20 for online bin packing and 10 for TSP and FSSP
 - Number of generations: 20
 - Number of parents for E1 and E2: 5
 - LLM: GPT-3.5-turbo
 - experiment instance timeout: 60 seconds
""" 
import os
import sys
from pathlib import Path
import time
from datetime import datetime
import shutil
import dataclasses
import numpy as np
import heapq  # for population management
import json
import yaml
from oragent.evaluator import Evaluator
import oragent.utils as utils
from oragent.utils import Solution



class EoH:
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
        
        # =====Create LLM client=====
        self.client = utils.LLMClient(config=self.config, 
                                    llm_provider=self.config['model']['llm_provider'], 
                                    model_name=self.config['model']['model_name'])
        
        # =====Create evaluator=====
        self.evaluator = Evaluator(config=self.config)
        
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
            self.population = None  # population so far; List of `Solution` instances
               
        # =====EoH settings=====
        self.init_pop_size = self.config['init_pop_size']
        self.pop_size = self.config['pop_size']
        self.max_evolutions = self.config['max_evolutions']
        self.autosave_interval_minutes = self.config['autosave_interval_minutes']
        self.num_parents = 2  # number of parents for 'e1' and 'e2' operators, default = 2
        self.operators = ['e1', 'e2', 'm1', 'm2', 'm3']
        self.operator_weights = [1, 1, 1, 1, 1]  # weights for operators, i.e., the probability of use the operator in each iteration, default = [1, 1, 1, 1, 1]
        
        # =====Loading all text prompts=====
        self.problem_dir = f"{self.project_root}/problems/{self.problem}"
        self.prompt_dir = f"{self.project_root}/prompts"
        self.output_dir = self.config['output_dir'] or f"{self.project_root}/outputs/{self.algorithm}/{self.problem}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/details", exist_ok=True)  # folder to store details
        self.problem_description = utils.file_to_string(f'{self.problem_dir}/problem_description.txt')
        self.function_description = utils.file_to_string(f'{self.problem_dir}/function_description.txt')
        # Common prompts
        self.init_prompt = utils.file_to_string(f'{self.prompt_dir}/init.txt')
        self.evolution_prompts = {
            "e1": utils.file_to_string(f'{self.prompt_dir}/e1_eoh.txt'),
            "e2": utils.file_to_string(f'{self.prompt_dir}/e2_eoh.txt'),
            "m1": utils.file_to_string(f'{self.prompt_dir}/m1_eoh.txt'),
            "m2": utils.file_to_string(f'{self.prompt_dir}/m2_eoh.txt'),
            "m3": utils.file_to_string(f'{self.prompt_dir}/m3_eoh.txt'),
        }
        
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
        print("\n>>>[EoH] EoH initialized")
                      

    def init_population(self) -> None:
        # =====Generate initial population (num = self.init_pop_size)=====
        # Note: multiple (num = self.init_pop_size) samples are generated in parallel
        user = self.init_prompt.format(
                                    problem_description=self.problem_description,
                                    function_description=self.function_description,
                                    function_to_evolve=self.function_to_evolve, 
                                    )
        messages = [{"role": "user", "content": user}]
        #logging.info("Initial prompt: \nUser Prompt: \n" + self.init_prompt)
        responses = self.client.multi_chat(messages, self.init_pop_size)  # self.pop_size maybe large; use multi_chat
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

        # Update self.population to be the new `population`
        self.population = population
        self.update_iter()
        print(f"\n>>>[EoH] Population initialization done. Population size: {len(self.population)}")
        
        
    def population_management(self, pop=None, size=None) -> None:
        """ 
        Population management
        - Remove individuals with duplicated objective scores (Not applied here since float numbers barely duplicate)
        - Keep only `size` the best individuals
    
        Args:
            pop (list[dict]): Population of individuals; each individual is a dict with keys: 'objective', 'code', 'file_name', 'response_id'.
            size (int): Size of the population to keep.
            
        Returns:
            list[dict]: The population with the best individuals.
        
        """
        if pop is None:  # if pop is not provided, use self.population
            pop = self.population
            
        if size is None:  # if size is not provided, use self.pop_size
            size = self.pop_size
        
        # Remove individuals with exec_success=False
        pop = [individual for individual in pop if individual.score]  
        
        if size > len(pop):
            size = len(pop)
        
        '''deprecated
        # Remove individuals with duplicate objective scores
        unique_pop = [] 
        unique_objectives = []
        for individual in pop:
            if individual['objective'] not in unique_objectives:
                unique_pop.append(individual)
                unique_objectives.append(individual['objective'])
        '''

        # Delete the worst individuals; keep only `size` elements
        if self.obj_type == "min":
            pop_new = heapq.nsmallest(size, pop, key=lambda individual: individual.score)
        else:
            pop_new = heapq.nlargest(size, pop, key=lambda individual: individual.score)
        """ 
        The heapq module in Python provides an implementation of the heap queue algorithm, 
        also known as the priority queue algorithm. It uses a min-heap data structure, 
        meaning the smallest element is always at the root (index 0) of the heap.
        
        heapq.nsmallest(n, iterable, key=None)
         • Returns a list containing the n smallest elements from the iterable.
         • Uses a heap internally, making it more efficient than sorted() when n is much smaller than the total length of the iterable.
        """
        #return pop_new
        self.population = pop_new
            
    
    def update_iter(self) -> None:
        """
        Update after each iteration, including:
        - update the best code and objective so far and save
        Note: common for ReEvo, EoH, AEL, FunSearch
        """
        print("\n[EoH] Updating iteration...")
        print(f"\n>>>[EoH] population size: {len(self.population)}")
        population = [ind for ind in self.population if ind.score]  # filter out invalid individuals
        self.population = population
        objs = [ind.score for ind in population]
        print(f"\n>>>[EoH] valid population size: {len(self.population)}")
        
        if objs:
            if self.obj_type == "min":
                best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))
            else:
                best_obj, best_sample_idx = max(objs), np.argmax(np.array(objs))
        
            # Update elitist
            elitist_updated = False
            if self.elitist is None or (self.obj_type == "min" and best_obj < self.elitist.score) or (self.obj_type == "max" and best_obj > self.elitist.score):
                self.elitist = population[best_sample_idx]
                elitist_updated = True
                print(f"\n>>>[EoH] Elitist updated: {self.elitist.id_str(self.algorithm)} | score:{self.elitist.score} | code path: {self.elitist.code_filepath}")
                
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
        else:
            print(f"\n>>>[EoH] Warning: No valid solution found during this iteration!")
        
        self.iteration += 1  # Note: self.iteration increase by 1 each time you invoke update_iter()!
        print(f"\n>>>[EoH] Iteration {self.iteration} finished...")
        
    
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

        # Save state variables
        state = {
            'iteration': self.iteration,  # int
            'total_responses': self.total_responses,  # int
            'function_evals': self.function_evals,  # int
            'valid_responses': self.valid_responses,  # int
            #'long_term_reflection_str': self.long_term_reflection_str,  # str
            'elitist': dataclasses.asdict(self.elitist) if self.elitist else None,  # dict
            'population': [dataclasses.asdict(sol) for sol in self.population]  # List[dict]
        }

        state_path = os.path.join(checkpoint_directory, 'state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=4, default=str)

        # Save results.json
        # we need to save a copy of results.json instead of directly append to the existing one in output dir
        # since the existing one may be ahead of the checkpoint in terms of iterations
        results_source = f"{self.output_dir}/results.json"
        results_dest = os.path.join(checkpoint_directory, 'results.json')
        if os.path.exists(results_source):
            shutil.copy2(results_source, results_dest)

        print(f"\n>>>[EoH] Checkpoint saved to: {checkpoint_directory}")
    
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
        self.population = [Solution(**sol_dict) for sol_dict in state['population']]

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
        print(f"population size:", len(self.population))
        #print(f"long-term reflection:", self.long_term_reflection_str)
        print(f"\n>>>[EoH] Checkpoint loaded from: {checkpoint_directory}")


    def get_parents(self, pop: list[Solution], operator: str) -> list[list[Solution]]:
        """
        Get parents for a given operator.
         
        Args:
            pop (list): population
            operator (str): operator name
        
        Returns:
            parents_list: list of parents of length `pop_size`
        """
        parents_lst = []
        for _ in range(self.pop_size):  # Note: we select `pop_size` many parents
            if operator in ["e1", "e2"]:
                # sample two parents
                parents = utils.parent_selection(pop, self.obj_type, self.num_parents, denominator_expansion=True)
            elif operator in ["m1", "m2", "m3"]:
                # sample a single parent
                parents = utils.parent_selection(pop, self.obj_type, 1, denominator_expansion=True)
            else:
                raise RuntimeError(f"\n>>>[EoH] Error: Evolution operator [{operator}] has not been implemented!\n")
            parents_lst.append(parents)
            
        assert len(parents_lst) == self.pop_size
        return parents_lst


    def get_offsprings(self, parents_lst: list[list[Solution]], operator: str): 
        """
        Call LLM using parents and get the offsprings.
         
        Args:
            pop (list): population
            operator (str): operator name
        
        Returns:
            parents, offspring of length `pop_size`
        """
        # messages list var
        messages_lst = []
        
        for i in range(len(parents_lst)):
            parents = parents_lst[i]
            # Get parent(s) description
            if operator in ["e1", "e2"]:
                parents_description = ""
                for j in range(len(parents)):
                    parents_description += f"""
### Solution #{j + 1}
{utils.individual_to_str(parents[j])}
"""
            elif operator in ["m1", "m2", "m3"]:
                parents_description = f"""
### Parent Solution
{utils.individual_to_str(parents[0])}
"""
            else:
                raise ValueError(f"Invalid operator: {operator}")
            
            # get task prompt
            user = self.evolution_prompts[operator].format(problem_description=self.problem_description,
                                            function_description=self.function_description,
                                            function_to_evolve=self.function_to_evolve,
                                            parents_description=parents_description)
            # create messages
            messages = [{"role": "user", "content": user}]
            messages_lst.append(messages)
        
        # get responses
        responses_lst = self.client.multi_chat(messages_lst)
        
        # get individuals
        offsprings = utils.responses_to_population(responses_lst, self.iteration, self.output_dir)
        self.total_responses += len(offsprings)
        
        # evaluate individuals
        offsprings = self.evaluator.evaluate_population(offsprings)
        self.function_evals += len(offsprings)
        # filter offsprings
        offsprings = [ind for ind in offsprings if ind.score]
        self.valid_responses += len(offsprings)

        return offsprings


    def run(self):
        last_save_time = time.time()
        
        # five update_iter() updates per loop
        
        # population initial size: self.pop_size
        #for pop in range(n_start, self.n_pop):  # Loop over N, the size of the population  # deprecated
        while self.total_responses <= self.max_evolutions or self.function_evals <= self.max_evolutions:
            # Note:
            # `self.function_evals` is incremented for each individual evolution
            # the total number of individual evolutions are constrained by `max_fe times`
            
            # Current population size: self.pop_size from the previous iteration
            
            # =====Loop over operators (evolution operators: ['e1','e2','m1','m2']); and conduct operations on population=====
            for i in range(len(self.operators)):  
                op = self.operators[i]
                print(f"\n>>>[EoH] OP: {op}, [{i + 1} / {len(self.operators)}] ", end="|") 
                
                # Log progress to progress.txt for webui real-time visualization
                with open(f"{self.output_dir}/progress.txt", 'w') as file:
                    self.print_progress(file=file)
                    print(f"Current operator: {op}", file=file)
                
                op_w = self.operator_weights[i]
                if (np.random.rand() < op_w):
                    # if np.random.rand() >= op_w, this operator is skipped - not applied to the population
                    # by this way, we can control which operators are applied
                    # say, we can let only one operator be applied to the population
                    # by setting operator_weights like [0, 1, 0, 0, 0]
                    # In EoH paper, all operators are applied to the population
                    # So, here we have op_w = [1, 1, 1, 1, 1]
                    
                    # -----Sample parents-----
                    # arguments: current population, selected operator
                    print("\n>>>[EoH] Getting Parents...")
                    parents_lst = self.get_parents(self.population, op)
                    # `parents_lst` len: self.pop_size;
                    # Note: 
                    # if current operator need 2 parents, then parents_lst[i] is a list of two individuals;
                    # if current operator need only 1 parent, then parents_lst[i] is a list of single individuals
                    
                    # -----Generate offsprings-----
                    print("\n>>>[EoH] Getting Offsprings...")
                    offsprings = self.get_offsprings(parents_lst, op)  # `offsprings` len: self.pop_size
                
                # -----Add offsprings to population-----
                """deprecated
                # Check duplication, and add the new offspring
                self.add2pop(population, offsprings)  
                """
                self.population.extend(offsprings)  # `self.population` size increase by self.pop_size after each operator application
                
                for individual in offsprings:
                    print("\n>>>[EoH] Obj: ", individual.score)
                
                # Update the best code and best score after each operator application
                self.update_iter()
                
            # =====population management; keep the population size within the limit=====
            # invoked after one loop of all types of evolutions
            print("\n>>>[EoH] Population Management...")
            self.population_management()  # `self.population` size change: 5 * self.pop_size -> self.pop_size
            # Population size is controlled this way
            
            # Log progress to progress.txt for webui real-time visualization
            with open(f"{self.output_dir}/progress.txt", 'w') as file:
                self.print_progress(file=file)
                print(f"Just finished one round of evolution", file=file)

            # =====Autosaving=====
            if time.time() - last_save_time >= self.autosave_interval_minutes * 60:
                self.save()
                last_save_time = time.time()
        # return the best code
        #return self.best_code_overall, self.best_code_filepath_overall
