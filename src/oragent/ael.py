# src/oragent/ael.py
"""
# AEL: Algorithm evolution using large language model

See paper:
 - Liu, F., Tong, X., Yuan, M. and Zhang, Q., 2023. Algorithm evolution using large language model
The original implementation is not available.
ReEvo implementation:
 - https://github.com/ai4co/reevo/tree/main/baselines/ael
This algorithm has been re-implemented in this project for compatibility, but the core functionality remains the same.


## AEL Process
N_g: The number of iterations
N: Population size
s: The number of new individuals for crossover

Initialization:
for j = 1, . . . ,N do
    Algorithm Creation: create new individual a_j
    given the target problem using LLM; Evaluate a_j
    and get fitness value f(a_j);
    Construct initial population P = {a_1, . . . , a_N};

Evaluation:
for i = 1, . . . ,N_g do
    for j = 1, . . . ,N do
        Selection: random (uniform dist) select a subset of input individuals p_j = {a1, . . . , al};
        
        (Algorithm) Crossover with probability θ_1:
            create a individual set o_j = {a_1, ..., a_s} using LLM given the target problem and input subset p_j;
        
        for k = 1, . . . , s do
            (Algorithm) Mutation with probability θ_2:
                modify individual a_k from the newly created individual set using LLM;
            Evaluate a_k and get fitness value f(a_k);
        
    Population management: P = P U {o_1, ..., o_N}, manage population P to reduce the size from (s + 1) * N to N.



## Hyperparameters
In the original paper, the experimental settings for AEL are as follows:
 - Population size N: 10
 - Number of population Ng: 10
 - Probability for crossover: 1.0
 - Probability for mutation: 0.2
 - Number of parent individuals l: 2
 - Number of offspring individuals: 1
 - LLM: GPT-3.5-turbo and GPT-4
""" 
import os
import sys
import json
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import shutil
import dataclasses
import heapq
import yaml
from oragent.evaluator import Evaluator
import oragent.utils as utils
from oragent.utils import Solution



class AEL:
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
        
        # =====AEL settings=====
        self.init_pop_size = self.config['init_pop_size']
        self.pop_size = self.config['pop_size']
        self.max_evolutions = self.config['max_evolutions']
        self.autosave_interval_minutes = self.config['autosave_interval_minutes']
        self.mutation_rate = self.config['mutation_rate']  # probability for mutation
        
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
        self.crossover_prompt = utils.file_to_string(f'{self.prompt_dir}/crossover_ael.txt')
        self.mutation_prompt = utils.file_to_string(f'{self.prompt_dir}/mutation_ael.txt')
        
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
        print(">>>\n[AEL] AEL initialized.")

        
    def init_population(self) -> None:
        # Generate initial population (num = self.init_pop_size)
        # Same with EoH
        # Note: multiple (num = self.init_pop_size) samples are generated in parallel
        user = self.init_prompt.format(
                                    problem_description=self.problem_description,
                                    function_description=self.function_description,
                                    function_to_evolve=self.function_to_evolve, 
                                    )
        messages = [{"role": "user", "content": user}]
        #print("Initial prompt: \nUser Prompt: \n" + self.init_prompt)
        responses = self.client.multi_chat(messages, self.init_pop_size)  # self.pop_size maybe large; use multi_chat
        # extract function samples from responses;
        # `population` is a tmp list
        population = utils.responses_to_population(responses, self.iteration, self.output_dir)
        self.total_responses += len(population)

        # Run code and evaluate population
        # by invoking self.evaluator(), we will get a dict
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
        print("\n[AEL] Updating iteration...")
        print(f"\n>>>[AEL] population size: {len(self.population)}")
        population = [ind for ind in self.population if ind.score]  # filter out invalid individuals
        self.population = population
        objs = [ind.score for ind in population]
        print(f"\n>>>[AEL] valid population size: {len(self.population)}")
        
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
                print(f"\n>>>[AEL] Elitist updated: {self.elitist.id_str(self.algorithm)} | score:{self.elitist.score} | code path: {self.elitist.code_filepath}")
                
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
            print(f"\n>>>[AEL] Warning: No valid solution found during this iteration!")
        
        self.iteration += 1  # Note: self.iteration increase by 1 each time you invoke update_iter()!
        print(f"\n>>>[AEL] Iteration {self.iteration} finished...")


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
        
        print(f"\n>>>[AEL] Checkpoint saved to: {checkpoint_directory}")
    
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
        print(f"\n>>>[AEL] Checkpoint loaded from: {checkpoint_directory}")
        
        
    def crossover(self, population: list[Solution]) -> list[Solution]:
        ''' 
        Given a selected population, first generate pairs
        then for each pair of parents, perform crossover, and generate a child
        
        Args:
            population: list[dict]; selected population; size: 2*pop_size
            
        Returns:
            crossed_population: list[dict]; generated crossed population; size: pop_size
        '''
        crossed_population = []
        assert len(population) == self.pop_size * 2, f"Selected population size mismatch: len(population)={len(population)}, expected={self.pop_size * 2} (self.pop_size={self.pop_size})"
        
        messages_lst = []
        for i in range(0, len(population), 2):  
            # we pair up the parents; 
            # selected population of size 2*pop_size
            # so we have pop_size many "messages"
            # and pop_size many children will be generated
            parent_1 = population[i]
            parent_2 = population[i+1]
            
            # construct crossover prompt
            user = self.crossover_prompt.format(
                #alg_desc1 = parent_1["description"],  # description is not needed now; they are added to the docstring
                #alg_desc2 = parent_2["description"],
                problem_description=self.problem_description,
                function_description=self.function_description,
                function_to_evolve=self.function_to_evolve,
                solution1=utils.individual_to_str(parent_1),
                solution2=utils.individual_to_str(parent_2),
            )
            messages = [{"role": "user", "content": user}]
            messages_lst.append(messages)
        
        # Multi-processed chat completion
        # Asynchronously generate responses
        # Note: if len(messages_lst) is too large, LLM api may support unless we handle it in multi_chat()
        responses_lst = self.client.multi_chat(messages_lst)
        crossed_population = utils.responses_to_population(responses_lst, self.iteration, self.output_dir)
        self.total_responses += len(responses_lst)
        
        crossed_population = self.evaluator.evaluate_population(crossed_population)
        self.function_evals += len(crossed_population) 
        # original author does not evaluate the population here
        # we evaluate here; so more info is provided in mutation stage
        
        assert len(crossed_population) == self.pop_size
        return crossed_population


    def mutate(self, population: list[Solution]) -> list[Solution]:
        """ 
        Mutate the population.
        
        Args:
            population (list[dict]): population to mutate; previously crossovered population
        
        Returns:
            list[dict]: mutated population
        """
        messages_lst = []
        mutated_idx = [0] * len(population)
        for i in range(len(population)):
            individual = population[i]
            
            # Mutate
            if np.random.uniform() < self.mutation_rate:
                if individual.code is None:
                    continue
                
                # construct mutate prompt
                user = self.mutation_prompt.format(
                    problem_description=self.problem_description,
                    function_description=self.function_description,
                    function_to_evolve=self.function_to_evolve,
                    solution=utils.individual_to_str(individual),
                    )
                messages = [{"role": "user", "content": user}]
                messages_lst.append(messages)
                mutated_idx[i] = 1
                
        # Multi-processed chat completion
        if messages_lst:
            # Rare case: all individuals are not mutated
            responses_lst = self.client.multi_chat(messages_lst)
            mutated_population = utils.responses_to_population(responses_lst, self.iteration, self.output_dir)
            
            mutated_population = self.evaluator.evaluate_population(mutated_population)
            self.function_evals += len(mutated_population)
        else:
            responses_lst = []
            mutated_population = []

        print(f"\n[AEL] {len(mutated_population)} out of {len(population)} mutated")
        self.total_responses += len(responses_lst)
        
        # Replace original ind with mutated one
        j = 0
        for i in range(len(population)):
            if mutated_idx[i] == 1:
                population[i] = mutated_population[j]
                j += 1
        
        assert len(population) == self.pop_size, f"length of population: {len(population)}  self.pop_size:{self.pop_size}"
        
        return population


    def run(self):
        last_save_time = time.time()
        
        # One update_iter() updates per loop
        
        # population initial size: self.pop_size
        while self.total_responses <= self.max_evolutions or self.function_evals <= self.max_evolutions:
            # Note:
            # `self.function_evals` is incremented for each individual evolution
            # the total number of individual evolutions are constrained by `max_fe times`
            
            # Current population size: self.pop_size from the previous iteration
            
            # =====Select parents=====
            print("\n>>>[AEL] Selecting population...")
            selected_population = utils.rank_select(population=self.population, 
                                                    obj_type=self.obj_type, 
                                                    pop_size=self.pop_size)  # selected population size: 2 * self.pop_size
            
            # =====Crossover=====
            # crossover on selected population
            # for each pair of parents, generate a child
            print("\n>>>[AEL] Crossover population...")
            crossed_population = self.crossover(selected_population)  # crossed_population size: self.pop_size

            # =====Mutate=====
            # mutate on the crossover population
            # for each individual, generate a mutated version
            print("\n>>>[AEL] Mutating population...")
            population = self.mutate(crossed_population)  # mutated_population size: self.pop_size
            
            print("\n>>>[AEL] Evaluating population...")
            population = [ind for ind in population if ind.score]
            self.valid_responses += len(population)
            
            # =====Update=====
            # add new population to the old population
            self.population.extend(population)  # self.population size: 2 * self.pop_size
            
            # update the best code and best score
            self.update_iter()
            
            # =====population management; keep the population size within the limit=====
            # invoked after one loop of all types of evolutions
            print("\n>>>[AEL] Population Management...")
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
