# src/oragent/reevo.py
""" 
# ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution

See paper:
 - Ye, H., Wang, J., Cao, Z., Berto, F., Hua, C., Kim, H., Park, J. and Song, G., 2024. ReEvo- Large language models as hyper-heuristics with reflective evolution
Original code:
 - https://github.com/ai4co/reevo/tree/main
This algorithm has been re-implemented in this project for compatibility, but the core functionality remains the same.


## Overview
ReEvo emulates human experts by reflecting on the relative performance
of two heuristics and gathering insights across iterations. This reflection approach is analogous
to interpreting genetic cues and providing "verbal gradient" within search spaces.
The dual-level reflections are used (long-term & short-term reflections).


## Process
The paper described five iterative steps: 
 - selection
 - short-term reflection
 - crossover
 - long-term reflection
 - elitist mutation


## ReEvo class 
ReEvo.init() will invoke:
  - init_population() to initialize the population to given size
 
ReEvo.run() run the evolution process:
1. Random select from the population:
    -> do short-term reflection
    -> generate offsprings
    
2. Use the new short-term reflection + old long-term reflection:
    -> update long-term reflection
    
(ReEvo.mutate())
3. Select elite individuals from the population + updated long-term reflection:
    -> generate offsprings



## Hyperparameters
In the original paper, the hyperparameters are:
 - LLM (generator and reflector): gpt-3.5-turbo
 - LLM temperature (generator and reflector): 1
 - Population size: 10
 - Number of initial generation: 30
 - Maximum number of evaluations: 100
 - Crossover rate: 1
 - Mutation rate: 0.5
"""
import os
import sys
from pathlib import Path
import time
from datetime import datetime
import shutil
import yaml
import numpy as np
import json
import dataclasses
from oragent.evaluator import Evaluator
import oragent.utils as utils
from oragent.utils import Solution



class ReEvo:
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
        self.generator_llm = utils.LLMClient(config=self.config, 
                                             llm_provider=self.config['model']['reevo_generator_llm_provider'], 
                                             model_name=self.config['model']['reevo_generator_model_name'])
        self.generator_llm_temperature = self.config['model']['reevo_generator_temperature']
        self.short_reflector_llm = utils.LLMClient(config=self.config, 
                                                   llm_provider=self.config['model']['reevo_short_reflector_llm_provider'], 
                                                 model_name=self.config['model']['reevo_short_reflector_model_name'])
        self.long_reflector_llm = utils.LLMClient(config=self.config, 
                                                  llm_provider=self.config['model']['reevo_long_reflector_llm_provider'], 
                                                 model_name=self.config['model']['reevo_long_reflector_model_name'])
        self.crossover_llm = utils.LLMClient(config=self.config, 
                                             llm_provider=self.config['model']['reevo_crossover_llm_provider'], 
                                             model_name=self.config['model']['reevo_crossover_model_name'])
        self.mutation_llm = utils.LLMClient(config=self.config, 
                                            llm_provider=self.config['model']['reevo_mutation_llm_provider'], 
                                             model_name=self.config['model']['reevo_mutation_model_name'])
        
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
            self.long_term_reflection = ""  # long term reflection
            self.population = None  # population so far; List of `Solution` instances
        
        # =====ReEvo settings=====
        self.init_pop_size = self.config['init_pop_size']
        self.pop_size = self.config['pop_size']
        self.mutation_rate = self.config['mutation_rate']
        self.max_evolutions = self.config['max_evolutions']
        self.autosave_interval_minutes = self.config['autosave_interval_minutes']
        
        # =====Loading all text prompts=====
        # Problem and prompt directory
        self.problem_dir = f"{self.project_root}/problems/{self.problem}"
        self.prompt_dir = f"{self.project_root}/prompts"
        self.output_dir = self.config['output_dir'] or f"{self.project_root}/outputs/{self.algorithm}/{self.problem}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/details", exist_ok=True)  # folder to store details
        self.problem_description = utils.file_to_string(f'{self.problem_dir}/problem_description.txt')
        self.function_description = utils.file_to_string(f'{self.problem_dir}/function_description.txt')
        if os.path.exists(f'{self.problem_dir}/external_knowledge.txt'):
            self.external_knowledge = utils.file_to_string(f'{self.problem_dir}/external_knowledge.txt')
        else:
            self.external_knowledge = ""
        self.seed_function = utils.file_to_string(f'{self.problem_dir}/seed_solution.py')
        # Common prompts
        self.system_generator_prompt = utils.file_to_string(f'{self.prompt_dir}/system_generator.txt')
        self.system_reflector_prompt = utils.file_to_string(f'{self.prompt_dir}/system_reflector.txt')
        self.user_reflector_st_prompt = utils.file_to_string(f'{self.prompt_dir}/user_reflector_st_reevo.txt')  # shrot-term reflection
        self.user_reflector_lt_prompt = utils.file_to_string(f'{self.prompt_dir}/user_reflector_lt_reevo.txt')  # long-term reflection
        self.crossover_prompt = utils.file_to_string(f'{self.prompt_dir}/crossover_reevo.txt')
        self.mutation_prompt = utils.file_to_string(f'{self.prompt_dir}/mutation_reevo.txt')
        self.init_with_seed_prompt = utils.file_to_string(f'{self.prompt_dir}/init_with_seed.txt')
        
        # =====Set long term reflection to external knowledge=====
        # self.long_term_reflection is a string that accumulates and stores the long-term reflection knowledge throughout the evolutionary process.
        # it's initialized to be empty string or external_knowledge if there exists external_knowledge.txt.
        # if no checkpoint specified, we set long-term reflection to be external knowledge at initialization.
        if not checkpoint:
            self.long_term_reflection = self.external_knowledge
            
            # Log long_term_reflection
            with open(f"{self.output_dir}/long_term_reflection.txt", 'w') as file:
                file.write(self.long_term_reflection)
        
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
        print("\n>>>[ReEvo] ReEvo initialized")
        

    def init_population(self) -> None:
        """ 
        1. Evaluate the seed function, and set it as Elite
        2. Create the initial population
        """                
        # =====Load and evaluate seed function=====
        # Evaluate the seed function, and set it as Elite
        print( "\n>>>[ReEvo] Evaluating seed function...")
        #logging.info("Seed function code: \n" + code)
        self.seed_ind = utils.response_to_individual(response=utils.wrap_python_code(self.seed_function), 
                                                     iteration=self.iteration, 
                                                     response_id=0,
                                                     output_dir=self.output_dir)
        # by invoking self.evaluator.evaluate_population(), we will also save the seed function to `self.population`
        self.seed_ind = self.evaluator.evaluate_individual(self.seed_ind)  # Evaluate seed function
        self.function_evals += 1
        # Now the population has a single individual, which is the seed function
        
        # If seed function is invalid, stop
        if not self.seed_ind.score:
            raise RuntimeError(f"\n>>>[ReEvo] Seed function is invalid. Please check the stdout file in {os.getcwd()}.")
        
        self.population = [self.seed_ind]
        
        # Update iteration
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
        #logging.info("Initial Population Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)

        # Generate initial population (num = self.init_pop_size)
        # Note: multiple (num = self.init_pop_size) samples are generated in parallel
        responses = self.generator_llm.multi_chat([messages], self.init_pop_size, temperature = self.generator_llm_temperature + 0.3) # Increase the temperature for diverse initial population
        # extract function samples from responses;
        # `population` is a tmp list of `Solution`s
        population = utils.responses_to_population(responses, self.iteration, self.output_dir)
        self.total_responses += len(population)
        
        # Run code and evaluate population
        # by invoking self.evaluator.evaluate_population(), we will get a dict
        population = self.evaluator.evaluate_population(population=population)
        self.function_evals += len(population)
        population = [ind for ind in population if ind.score]
        self.valid_responses += len(population)
        
        # Update self.population to be the new `population`
        self.population = population
        self.update_iter()
        print(f"\n>>>[ReEvo] Population initialization done. Population size: {len(self.population)}")
        
        
    def update_iter(self) -> None:
        """
        Update after each iteration, including:
        - update the best code and objective so far and save
        Note: common for ReEvo, EoH, AEL, FunSearch
        """
        print("\n[ReEvo] Updating iteration...")
        print(f"\n>>>[ReEvo] population size: {len(self.population)}")
        population = [ind for ind in self.population if ind.score]  # filter out invalid individuals
        self.population = population
        objs = [ind.score for ind in population]
        print(f">>>[ReEvo] Valid population size: {len(self.population)}")
        
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
                print(f"\n>>>[ReEvo] Elitist updated: {self.elitist.id_str(self.algorithm)} | score:{self.elitist.score} | code path: {self.elitist.code_filepath}")
                
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
            print(f"\n>>>[ReEvo] Warning: No valid solution found during this iteration!")
        
        self.iteration += 1  # Note: self.iteration increase by 1 each time you invoke update_iter()!
        print(f"\n>>>[ReEvo] Iteration {self.iteration} finished...")
        
            
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
            'long_term_reflection': self.long_term_reflection,  # str
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

        print(f"\n>>>[ReEvo] Checkpoint saved to: {checkpoint_directory}")
    
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
        self.long_term_reflection = state['long_term_reflection']
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
        print(f"long-term reflection:", self.long_term_reflection)
        print(f"\n>>>[ReEvo] Checkpoint loaded from: {checkpoint_directory}")
    
    
    def gen_short_term_reflection_prompt(self, ind1: Solution, ind2: Solution) -> tuple[list[dict], str, str]:
        """
        Short-term reflection before crossovering two individuals.
        Generate prompts for LLM.
        
        Args:
            ind1: dict, first individual
            ind2: dict, second individual
        
        Returns:
            (prompt for LLM, worse individual code, better individual code) -> (tuple[list[dict], str, str)
        """
        # robust in rare cases where two individuals have the same objective value
        #if ind1.score == ind2.score:
        #    print(ind1["code"], ind2["code"])
        #    raise ValueError("Two individuals to crossover have the same objective value!")
        # Determine which individual is better or worse
        if self.obj_type == "min":
            if ind1.score <= ind2.score:
                better_ind, worse_ind = ind1, ind2
            else:
                better_ind, worse_ind = ind2, ind1
        else: 
            if ind1.score <= ind2.score:
                better_ind, worse_ind = ind2, ind1
            else:
                better_ind, worse_ind = ind1, ind2

        #worse_solution = filter_code(worse_ind["code"])
        worse_solution = utils.individual_to_str(worse_ind)
        #better_solution = filter_code(better_ind["code"])
        better_solution = utils.individual_to_str(better_ind)

        user = self.user_reflector_st_prompt.format(
            problem_description = self.problem_description,
            function_description = self.function_description,
            function_to_evolve = self.function_to_evolve,
            worse_solution=worse_solution,
            better_solution=better_solution
            )
        message = [{"role": "system", "content": self.system_reflector_prompt}, {"role": "user", "content": user}]
        
        return message, worse_solution, better_solution


    def short_term_reflect(self, population: list[dict]) -> tuple[list[list[dict]], list[str], list[str]]:
        """
        Short-term reflection before crossovering two individuals.
        Note:
            We need to return parents code so that we can do crossover at a later stage.
        
        Args:
            population (list[dict]): A list of individuals; the selected population
            
        Returns:
            response_lst: short-term reflections
            worse_solution_lst: list of worse code used to generate reflections
            better_solution_lst: list of better code used to generate reflections
        """
        # create messages list first; the number of parent pairs is: len(population) / 2 number of pairs
        # then call LLM concurrently (by invoking multi_chat)
        messages_lst = []
        worse_solution_lst = []
        better_solution_lst = []
        for i in range(0, len(population), 2):  # len(population) / 2 number of pairs
            # Select two individuals
            parent_1 = population[i]
            parent_2 = population[i + 1]
            # Short-term reflection prompt
            messages, worse_solution, better_solution = self.gen_short_term_reflection_prompt(parent_1, parent_2)
            messages_lst.append(messages)  # message is the st-reflection
            worse_solution_lst.append(worse_solution)
            better_solution_lst.append(better_solution)
        
        # Asynchronously generate responses
        # `response_lst` is a list of responses; shape: self.pop_size
        response_lst = self.short_reflector_llm.multi_chat(messages_lst)
        self.total_responses += len(response_lst)
        self.valid_responses += len(response_lst)
        
        return response_lst, worse_solution_lst, better_solution_lst
    
    
    def long_term_reflect(self, short_term_reflections: list[str]) -> None:
        """
        Long-term reflection before mutation.
        
        Args:
            short_term_reflections (list[str]): A list of short-term reflections; shape: self.pop_size
            
        Returns:
            None. self.long_term_reflection will be updated. 
            Note: we maintain a single long-term reflection string for all individuals.
        """
        user = self.user_reflector_lt_prompt.format(
            problem_description = self.problem_description,
            function_description = self.function_description,
            function_to_evolve = self.function_to_evolve,
            prior_reflection = self.long_term_reflection,
            new_reflection = "\n".join(short_term_reflections),  # Note: all short-term reflections are concatenated into one string! so lt-reflection will summarize all short-term reflections
            )
        messages = [{"role": "system", "content": self.system_reflector_prompt}, {"role": "user", "content": user}]
        
        # Invoke LLM to generate long-term reflection
        # then update self.long_term_reflection
        self.long_term_reflection = self.long_reflector_llm.multi_chat([messages])[0]
        self.total_responses += 1
        self.valid_responses += 1
        
        # Log reflections to file
        # short-term reflections
        file_name = f"{self.output_dir}/details/iter{self.iteration}_short_term_reflections.txt"
        with open(file_name, 'w') as file:
            file.writelines("\n".join(short_term_reflections) + '\n')
        # long-term reflection
        file_name = f"{self.output_dir}/details/iter{self.iteration}_long_term_reflection.txt"
        with open(file_name, 'w') as file:
            file.writelines(self.long_term_reflection + '\n')


    def crossover(self, short_term_reflection_tuple: tuple[list[str], list[str], list[str]]) -> list[Solution]:
        """ 
        Crossover over pairs of parents with previously generated short term reflections:
        worse_solution, better_solution, st-reflection -> offspring code
        
        Usage:
        ```
        short_term_reflection_tuple = self.short_term_reflect(selected_population) # (response_lst, worse_solution_lst, better_solution_lst)
        crossed_population = self.crossover(short_term_reflection_tuple)
        ```
        
        Args:
            short_term_reflection_tuple (tuple[list[str], list[str], list[str]]): A tuple containing:
            - response_lst: A list of responses from the LLM; each response is a list of dicts
            - worse_solution_lst: A list of worse code strings (worse parent)
            - better_solution_lst: A list of better code strings (better parent)
        
        Returns:
            list[dict]: A list of offspring individuals
        """
        # First generate messages list
        # then we can generate responses concurrently
        reflection_content_lst, worse_solution_lst, better_solution_lst = short_term_reflection_tuple
        messages_lst = []
        for reflection, worse_solution, better_solution in zip(reflection_content_lst, worse_solution_lst, better_solution_lst):
            # Crossover
            system = self.system_generator_prompt
            #func_signature0 = self.func_signature.format(version=0)
            #func_signature1 = self.func_signature.format(version=1)
            user = self.crossover_prompt.format(
                problem_description = self.problem_description,
                function_description = self.function_description,
                function_to_evolve = self.function_to_evolve,
                worse_solution = worse_solution,
                better_solution = better_solution,
                reflection = reflection,
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            messages_lst.append(messages)
            
        # Asynchronously generate responses
        response_lst = self.crossover_llm.multi_chat(messages_lst)
        crossed_population = utils.responses_to_population(response_lst, self.iteration, self.output_dir)
        self.total_responses += len(crossed_population)

        assert len(crossed_population) == self.pop_size
        return crossed_population


    def mutate(self) -> list[dict]:
        """
        Elitist-based mutation. We only mutate the best (single) individual to generate int(self.pop_size * self.mutation_rate) new individuals.
        Note: we only mutate the best individual! and return the newly generated population
        
        elite (the best individual) + lt-reflection -> offsprings
        
        Return:
            list[dict]: The generated mutation population; shape: int(self.pop_size * self.mutation_rate); self.mutation_rate is 0.5 by default.
        """
        system = self.system_generator_prompt
        #func_signature1 = self.func_signature.format(version=1) 
        user = self.mutation_prompt.format(
            problem_description = self.problem_description,
            function_description = self.function_description,
            function_to_evolve = self.function_to_evolve,
            reflection = self.long_term_reflection + self.external_knowledge,
            elitist_solution = utils.individual_to_str(self.elitist),
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        responses = self.mutation_llm.multi_chat([messages], int(self.pop_size * self.mutation_rate))
        population = utils.responses_to_population(responses, self.iteration, self.output_dir)
        self.total_responses += len(population)
        return population

    
    def run(self):
        last_save_time = time.time()
        
        # two update_iter() updates per loop
        
        # self.population now has size self.init_pop_size
        while self.total_responses <= self.max_evolutions or self.function_evals <= self.max_evolutions:
            # `self.iteration` will increase by 1 each time `self.update_iter()` is invoked;
            # two iterations happen in population initialization; and for each loop, `self.iteration` will increase by 2.
            
            # If all individuals are invalid, stop
            if all([not individual.score for individual in self.population]):
                raise RuntimeError(f"All individuals are invalid. Please check the stdout files in {os.getcwd()}.")
            
            # Current population size: self.population is of size self.pop_size + int(self.pop_size * self.mutation_rate) from last iteration
            # after following `selection`, we will have `selected_population` with size 2 * self.pop_size
            # Population size is controlled through this way
                            
            # =====Select (uniform distribution)=====
            # Add elitist to population for selection since elitist may not be in self.population after crossover and mutation
            print("\n>>>[ReEvo] Selecting population...")
            population_to_select = self.population if (self.elitist is None or self.elitist in self.population) else [self.elitist] + self.population 
            selected_population = utils.random_select(population_to_select, self.pop_size)  # `selected_population` size: 2 * self.pop_size
            if selected_population is None:
                raise RuntimeError("Selection failed. Please check the population.")
            
            # Log progress to progress.txt for webui
            with open(f"{self.output_dir}/progress.txt", 'w') as file:
                self.print_progress(file=file)
                print(f"\n>>>[ReEvo] Selected population size: {len(selected_population)}; next: short-term reflection and crossover", file=file)
                
            # =====Short-term reflection on the selected population=====
            # pair of parents -> short term reflection
            print("\n>>>[ReEvo] Short-term reflection...")
            short_term_reflection_tuple = self.short_term_reflect(selected_population) # `short_term_reflection_tuple` is (response_lst, worse_solution_lst, better_solution_lst); 
            # `short_term_reflection_tuple` size: self.pop_size
            # `worse_solution_lst` and `better_solution_lst` are used in crossover later; they are the parents actually
            
            # =====Crossover on the selected population=====
            # pair of parents + st-term reflection -> crossover offsprings
            print("\n>>>[ReEvo] Crossover...")
            crossed_population = self.crossover(short_term_reflection_tuple)  # `crossed_population` size: self.pop_size
            
            # Evaluate the generated crossover offsprings; and update self.population
            print("\n>>>[ReEvo] Evaluating crossover offsprings...")
            self.population = self.evaluator.evaluate_population(crossed_population)  # here `self.population` size is back to self.pop_size again
            self.function_evals += len(self.population)
            self.population = [ind for ind in self.population if ind.score]
            self.valid_responses += len(self.population)
            
            # Update; so the best individual and best objective are updated
            self.update_iter()  
            
            # Log progress to progress.txt for webui
            with open(f"{self.output_dir}/progress.txt", 'w') as file:
                self.print_progress(file=file)
                print(f"Just finished crossover; next: long-term reflection and mutation", file=file)
            
            # =====Long-term reflection=====
            # old lt-reflection + st-term reflection -> new lt-reflection
            # `short_term_reflection_tuple[0]` is the list of responses from short-term reflection, namely short-term reflections generated
            # PS: all short-reflections are concatenated into a single string to do long-term reflection;
            # then st-reflections are discarded; `self.long_term_reflection` is updated
            print("\n>>>[ReEvo] Long-term reflection...")
            self.long_term_reflect([response for response in short_term_reflection_tuple[0]])  # long-term reflection is a single string
            
            # Log long_term_reflection
            with open(f"{self.output_dir}/long_term_reflection.txt", 'w') as file:
                file.write(self.long_term_reflection)
                
            # =====Mutate on the best (single) individual; generate int(self.pop_size * self.mutation_rate) many offsprings=====
            # best individual + new lt-term reflection -> mutation offsprings
            print("\n>>>[ReEvo] Mutation...")
            mutated_population = self.mutate()  # newly generated mutation population (based on the best individual); `mutated_population` size: int(self.pop_size * self.mutation_rate)
            
            # Evaluate the mutation offsprings
            print("\n>>>[ReEvo] Evaluating mutation offsprings...")
            mutated_population = self.evaluator.evaluate_population(mutated_population)
            self.population.extend(mutated_population)  # add mutation offsprings to self.population; `self.population` size increased to: self.pop_size + int(self.pop_size * self.mutation_rate)
            self.function_evals += len(mutated_population)
            self.valid_responses += len([ind for ind in mutated_population if ind.score])
            
            # Update; so the best individual and best objective are updated
            self.update_iter()
            
            # Log progress to progress.txt for webui
            with open(f"{self.output_dir}/progress.txt", 'w') as file:
                self.print_progress(file=file)
                print(f"\n>>>[ReEvo] Just finished one round of evolution", file=file)
                
            # =====Autosaving=====
            if time.time() - last_save_time >= self.autosave_interval_minutes * 60:
                self.save()
                last_save_time = time.time()

        # return the best code
        #return self.best_code_overall, self.best_code_filepath_overall