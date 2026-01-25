# src/oragent/code_agent.py
""" 
# Code Agent

## Overview
Code agent is responsible for generating code based on ideas.
Code agent is also responsible for debugging code when the generated code has errors.


## Steps
1. Generate code based on the idea provided by the idea agent.
2. Evaluate the generated code using the evaluator.
3. If there are errors in the code, debug the code using the LLM and re-evaluate until the code runs successfully or maximum debug rounds is reached.


## Code Generation Specification
LLM should generate a python code fenced bock. Example:
```python
def add_two_numbers(a, b):
    return a + b
```

## Code Update Specification
Code modifications is specified by conflict markers style.
LLM should return a string with fenced block ```diff ... ``` inside that contains the code modifications.
Example:
```diff
<<<<<<< SEARCH
import os
=======
import os
import math
>>>>>>> REPLACE
<<<<<<< SEARCH
    if not hasattr(driving_actions, 'vehicle_history'):
        driving_actions, 'vehicle_history') = {}
=======
    if not hasattr(driving_actions, 'vehicle_history'):
        driving_actions.vehicle_history = {}
>>>>>>> REPLACE
```

We also ask LLM to generate the thoughts about debugging at the same time, as this can improve the debugging performance (see Chain of Thought).
LLM client should output a JSON fenced block with exactly one field:
```json
{
    "thoughts": "thoughts on how to revise the code",
}
```

If score is successfully generated or max debugging rounds reached, then code agent will stop debugging and return the solution.
"""
import sys
import os
from pathlib import Path
import yaml
import json
import time
from datetime import datetime
import dataclasses
from typing import List, Tuple, Optional, Dict, Union
from oragent.evaluator import Evaluator
import oragent.utils as utils
from oragent.utils import Solution



class CodeAgent:
    """Agent for generating and debugging code."""
    def __init__(self, checkpoint=None, config=None):
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
        self.llm_provider = self.config['model']['code_agent_llm_provider']
        self.model_name = self.config['model']['code_agent_model_name']
        self.llm_client = utils.LLMClient(config=self.config, llm_provider=self.llm_provider, model_name=self.model_name)

        # =====Vars updated during agent running=====
        if not checkpoint:
            self.total_responses = 0  # Number of total responses; this can be used to track the number of LLM calls
            self.function_evals = 0  # Number of function evaluations; this is also an important metric for complexity, especially for the case when evaluation is the bottleneck
            self.valid_responses = 0  # Number of valid responses, namely responses that were successfully executed
        
        # =====Create evaluator=====
        self.evaluator = Evaluator(config=self.config)
        
        # =====CodeAgent settings=====
        self.max_debug_rounds = self.config['max_debug_rounds']
        self.reflection_disabled_for_crossover = self.config['reflection_disabled_for_crossover']  # whether long-term reflection is disabled  when doing crossover
        self.evaluation_description_disabled = self.config['evaluation_description_disabled']
        self.verbose = self.config['verbose']
        
        # =====Load problem data and prompts=====
        # Problem and prompt directory
        self.problem_dir = f"{self.project_root}/problems/{self.problem}"
        self.prompt_dir = f"{self.project_root}/prompts"
        self.output_dir = self.config['output_dir'] or f"{self.project_root}/outputs/{self.algorithm}/{self.problem}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/details", exist_ok=True)  # folder to store details
        # Load common prompts
        self.system_code_generation_prompt = utils.file_to_string(f'{self.prompt_dir}/system_code_generation_oragent.txt')  # system role prompt for code generation
        self.user_code_generation_prompt = utils.file_to_string(f'{self.prompt_dir}/user_code_generation_oragent.txt')  # user role prompt for code generation
        self.system_code_debugging_prompt = utils.file_to_string(f'{self.prompt_dir}/system_code_debugging_oragent.txt')  # system role prompt for code debugging
        self.user_code_debugging_prompt = utils.file_to_string(f'{self.prompt_dir}/user_code_debugging_oragent.txt')  # user role prompt for code debugging
        # Load problem-specific prompts
        self.problem_description = utils.file_to_string(f'{self.problem_dir}/problem_description.txt')
        # `self.evaluation_description_disabled` decides whether eval description is used as context
        if os.path.exists(f'{self.problem_dir}/evaluation_description.txt') and not self.evaluation_description_disabled:
            self.evaluation_description = utils.file_to_string(f'{self.problem_dir}/evaluation_description.txt')
        else:
            self.evaluation_description = None
        self.function_description = utils.file_to_string(f'{self.problem_dir}/function_description.txt')
        

    def reset(self):
        """Reset code agent state variables."""
        self.total_responses = 0  # Number of total responses; this can be used to track the number of LLM calls
        self.function_evals = 0  # Number of function evaluations; this is also an important metric for complexity, especially for the case when evaluation is the bottleneck
        self.valid_responses = 0 # Number of valid responses, namely responses that were successfully executed
        print(f"\n>>>[CodeAgent] Code Agent reset.")
    
    def save(self, checkpoint: str):
        """
        Save checkpoint. Saved files:
        - code_agent_state.json

        Args:
            checkpoint (str): checkpoint name; default None

        Return:
            None.
        """
        #checkpoint = checkpoint or datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{self.algorithm}_{self.problem}"  # default checkpoint name example: '2025-12-29_20-40-25'
        checkpoint_directory = f'{self.project_root}/checkpoints/{checkpoint}'
        os.makedirs(checkpoint_directory, exist_ok=True)

        # Save config
        # No need to save config file; it's already saved by ORAgent

        # Save state variables
        state = {
            'total_responses': self.total_responses,  # int
            'function_evals': self.function_evals,  # int
            'valid_responses': self.valid_responses,  # int
        }
        state_path = os.path.join(checkpoint_directory, 'code_agent_state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=4, default=str)
            
        print(f"\n>>>[CodeAgent] Checkpoint saved to: {checkpoint_directory}")
        
    def load(self, checkpoint: str):
        """Load checkpoint."""
        checkpoint_directory = f'{self.project_root}/checkpoints/{checkpoint}'
        
        # Load config
        config_path = os.path.join(checkpoint_directory, 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load state variables
        state_path = os.path.join(checkpoint_directory, 'code_agent_state.json')
        with open(state_path, 'r') as f:
            state = json.load(f)

        # Restore state variables
        self.total_responses = state['total_responses']
        self.function_evals = state['function_evals']
        self.valid_responses = state['valid_responses']
        
        print(f"\n>>>[CodeAgent] Checkpoint loaded from: {checkpoint_directory}")


    def run(self, 
            parent_solutions: Union[Solution, List[Solution]], 
            long_term_reflection: str, 
            solution: Solution,
            elitist_parent: bool=False,
            ):
        """
        Generate ideas for improving ideas.

        Args:
            parent_solutions (Union[Solution, List[Solution]): parent solution(s).
            long_term_reflection (Str): long term reflection of the lead agent.
            solution (Solution): currently solution that only contains the idea.
            elitist_parent (bool): whether elitist is used as root solution; default False.
            
        Returns:
            solution (Solution): solution with code, metrics, features, score updated
        """
        print(f"\n>>>[CodeAgent] Starts to work on solution {solution.id_str(algorithm=self.algorithm)}...")
        
        if not isinstance(parent_solutions, List):
            parent_solutions = [parent_solutions]
        
        # =====1. Generate code based on the idea=====
        # convert parent solutions to string
        parent_solutions_str = utils.parents_to_str(parent_solutions)
        
        # Handle long-term reflection for crossover
        # for mutation on elitist, we always use long-term reflection
        # for crossover, we may not want to provide long-term reflection
        if not elitist_parent and self.reflection_disabled_for_crossover:
            long_term_reflection = "None"
            print("\n>>>[CodeAgent] Long-term reflection is not used for this generation.")
        else:
            print("\n>>>[CodeAgent] Long-term reflection is used for this generation.")
        
        # Construct prompt
        user = self.user_code_generation_prompt.format(
            problem_description = self.problem_description,  # common
            function_to_evolve = self.function_to_evolve,  # common
            obj_type = self.obj_type,  # common
            evaluation_description = self.evaluation_description if self.evaluation_description else "(empty)",  # evaluation description is added to help code agent to understand the problem better
            function_description = self.function_description,  # common
            parent_solutions = parent_solutions_str if parent_solutions_str else "(empty)",  # parent solutions are added to help code agent to generate crossover and mutated code
            long_term_reflection = long_term_reflection if long_term_reflection else "(empty)",  # common
            idea = solution.idea,  # current idea
        )
        messages = [{"role": "system", "content": self.system_code_generation_prompt}, {"role": "user", "content": user}]
        
        attempt = 0
        while True:
            attempt += 1
            
            # Invoke LLM to generate code
            response = self.llm_client.chat(messages)
            self.total_responses += 1
            
            # Parse response into code string
            response_extracted = utils.extract_python_code(response)
            if response_extracted:
                solution.code = response_extracted
                break
            else:
                print(f"\n>>>[CodeAgent] Failed to extract code from response (attempt {attempt}): {response}")
                if attempt >= 5:
                    raise RuntimeError(f"\n>>>[CodeAgent] Max attempts reached for generating code")
        
        # =====2. Evaluate the generated code=====
        raw_output, metrics, features, score = self.evaluator.run(solution=solution, callbacks=None)
        self.function_evals += 1
        solution.output = raw_output
        solution.metrics = metrics
        solution.features = features
        solution.score = score
        
        # =====3. If there are errors, debug the code=====
        debug_round = 0
        while solution.score is None and debug_round < self.max_debug_rounds:  # if self.max_debug_rounds is 0, then no debugging is performed
            # if score is generated, then will break out of the while loop
            # or if max debug rounds reached, it still will return with the problematic code
            debug_round += 1
            print(f"\n>>>[CodeAgent] Debugging round {debug_round} for solution {solution.id_str(self.algorithm)}...")
            if self.verbose:
                print(f"\n>>>[CodeAgent] Current evaluation raw output:\n{raw_output}")
                
            # Construct debug prompt
            user = self.user_code_debugging_prompt.format(
                problem_description = self.problem_description,  # common
                function_to_evolve = self.function_to_evolve,  # common
                evaluation_description = self.evaluation_description if self.evaluation_description else "(empty)",  # evaluation description added to help code agent to understand the problem better
                function_description = self.function_description,  # common
                idea = solution.idea,  # current idea
                code = solution.code,  # current code
                raw_output = raw_output,  # raw output is the output of the evaluator to help code agent to understand the problem
            )
            messages = [{"role": "system", "content": self.system_code_debugging_prompt}, {"role": "user", "content": user}]
            
            # Invoke LLM to debug code
            response = self.llm_client.chat(messages)
            self.total_responses += 1
            
            # Parse response into updated code string
            thoughts = utils.extract_json(response)  # a dict with key 'thoughts'; for user to inspect
            code_diff = utils.extract_diff_block(response)
            
            if self.verbose:
                print(f"\n>>>[CodeAgent] Thoughts:\n{thoughts}")
                print(f"\n>>>[CodeAgent] Code diff:\n{code_diff}")
            
            if not code_diff:
                print(f"\n>>>[CodeAgent] Warning: no code diff found in response (attempt {debug_round}): \n{response}")
                # each debugging round must have code diff to update the code
                # otherwise, there may be some problem with the response
            else:          
                try:
                    # Update code
                    solution.code = utils.update_code(curr_code=solution.code, code_diff=code_diff)

                    # Re-evaluate the updated code
                    raw_output, metrics, features, score = self.evaluator.run(solution=solution, callbacks=None)
                    self.function_evals += 1
                    self.output = raw_output
                    solution.metrics = metrics
                    solution.features = features
                    solution.score = score
                    # Note: raw_output is updated whatsoever
                except Exception as e:
                    print(f"\n>>>[CodeAgent] Error: {e}\nInvalid response: \n{response}")

            # If above try block is not successful, will re-enter the while loop for next debugging round
            
            if debug_round >= self.max_debug_rounds:
                print(f"\n>>>[CodeAgent] Warning: max debugging rounds reached for solution {solution.id_str(self.algorithm)}. Stopping debugging.")
                break

        self.valid_responses += 1   # only count valid responses when debugging is done
        return solution
    
    
    def debug(self, solution: Solution):
        """
        Debug mode of CodeAgent.
        This method is the debugging part of `CodeAgent.run()` method.
        """        
        # Initial execution
        # this initial execution is in fact not needed for OR-Agent; solution fields are all filled when this method is invoked
        # however, we add this step for robustness and generality so that it still works when users customize OR-Agent
        raw_output, metrics, features, score = self.evaluator.run(solution=solution, callbacks=None)
        solution.output = raw_output
        solution.metrics = metrics
        solution.features = features
        solution.score = score
        
        if utils.is_valid(solution):
            return solution
        
        debug_round = 0
        while solution.score is None and debug_round < self.max_debug_rounds:  # if self.max_debug_rounds is 0, then no debugging is performed
            # if score is generated, then will break out of the while loop
            # or if max debug rounds reached, it still will return with the problematic code
            debug_round += 1
            print(f"\n>>>[CodeAgent] Debugging round {debug_round} for solution {solution.id_str(self.algorithm)}...")
            if self.verbose:
                print(f"\n>>>[CodeAgent] Current evaluation raw output:\n{raw_output}")
                
            # Construct debug prompt
            user = self.user_code_debugging_prompt.format(
                problem_description = self.problem_description,  # common
                function_to_evolve = self.function_to_evolve,  # common
                evaluation_description = self.evaluation_description if self.evaluation_description else "(empty)",  # evaluation description added to help code agent to understand the problem better
                function_description = self.function_description,  # common
                idea = solution.idea,  # current idea
                code = solution.code,  # current code
                raw_output = raw_output,  # raw output is the output of the evaluator to help code agent to understand the problem
            )
            messages = [{"role": "system", "content": self.system_code_debugging_prompt}, {"role": "user", "content": user}]
            
            # Invoke LLM to debug code
            response = self.llm_client.chat(messages)
            self.total_responses += 1
            
            # Parse response into updated code string
            # here we do not require formatting of thoughts since we thoughts are for human inspection don't need to them for automatic code generation
            # TODO: optionally, we can still extract thoughts for better logging
            # 1) require thoughts wrapped in <thinking>...</thinking>
            # 2) require thoughts wrapped in json block
            # Option 1) is preferred since LLM tend to terminate generation after finishing generating json block hence missing code diff block!
            #thoughts = utils.extract_json(response)  # a dict with key 'thoughts'; for user to inspect
            #thinking = utils.extract_thinking(response)  # a string of thoughts wrapped in <thinking>...</thinking>
            code_diff = utils.extract_diff_block(response)
            
            if self.verbose:
                print(f"\n>>>[CodeAgent] Response:\n{response}")
                #print(f"\n>>>[CodeAgent] Thoughts:\n{thoughts}")
                print(f"\n>>>[CodeAgent] Code diff:\n{code_diff}")
            
            if not code_diff:
                print(f"\n>>>[CodeAgent] Warning: no code diff found in response (attempt {debug_round}): \n{response}")
                # each debugging round must have code diff to update the code
                # otherwise, there may be some problem with the response
            else:
                try:
                    # Update code
                    solution.code = utils.update_code(curr_code=solution.code, code_diff=code_diff)

                    # Re-evaluate the updated code
                    raw_output, metrics, features, score = self.evaluator.run(solution=solution, callbacks=None)
                    self.function_evals += 1
                    self.output = raw_output
                    solution.metrics = metrics
                    solution.features = features
                    solution.score = score
                    # Note: raw_output is updated whatsoever
                except Exception as e:
                    print(f"\n>>>[CodeAgent] Error: {e}\nInvalid response: \n{response}")

            # If above try block is not successful, will re-enter the while loop for next debugging round
            
            if debug_round >= self.max_debug_rounds:
                print(f"\n>>>[CodeAgent] Warning: max debugging rounds reached for solution {solution.id_str(self.algorithm)}. Stopping debugging.")
                break

        self.valid_responses += 1   # only count valid responses when debugging is done
        return solution





if __name__ == '__main__':
    # For debugging purposes
    code_agent = CodeAgent()
    # First create seed solution
    seed_solution = Solution()
    seed_solution.idea = utils.file_to_string(f"{code_agent.problem_dir}/seed_solution_idea.txt")
    seed_solution.code = utils.file_to_string(f"{code_agent.problem_dir}/seed_solution.py")
    from oragent.evaluator import Evaluator
    evaluator = Evaluator()
    raw_output, metrics, features, score = evaluator.run(seed_solution)
    seed_solution.metrics = metrics
    seed_solution.features = features
    seed_solution.score = score
    # Create a child solution with an idea
    child_solution = Solution()
    child_solution.idea = "Implement hierarchical cooperative coordination combining local platoon formation with global flow optimization through distributed consensus on lane usage priorities, reducing redundant gap-creation maneuvers and improving traffic smoothness"
    # Generate code for the child solution
    solution = code_agent.run(parent_solutions=seed_solution, long_term_reflection="", solution=child_solution)
    print(solution)