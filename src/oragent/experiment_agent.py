# src/oragent/experiment_agent.py
"""
# Experiment Agent

## Overview
The experiment agent is responsible for running experiments to evaluate and improve candidate solutions.
It handles complex problem environments by exploring them to identify issues with the current solution.



## Experiment Agent Work Process
The experiment agent executes solution code and receives feedback from the environment. During each experiment, the agent can perform the following actions:
- **update code**: Update solution code based on code diffs, representing minor changes made to existing code
- **update callbacks**: Generate new callbacks to explore the problem environment and diagnose issues
- **terminate**: End the experiment if the solution meets performance requirements, cannot be further improved, or underlying issues have been identified

The experiment automatically terminates when the predefined maximum number of experiments (`max_experiment_repeats`) is reached.
After each experiment, the agent generates reflections to inform its next decision.

### Experiment Agent Workflow
```
Step 0) Prepare the experiment
Step 1) Experiment loop:
    Step 1.1) Evaluate the code and extract outputs
    Step 1.2) Call LLM to reflect on outputs and decide next action
    Step 1.3) Branch based on action:
        If action is `terminate`:
            Terminate the experiment
        Else if action is `update_code`:
            Update solution code based on code diff
        Else if action is `update_callbacks`:
            Update callbacks to explore problem environment and identify issues
    Step 1.4) If `max_experiment_repeats` is reached:
        Terminate the experiment
    Step 1.5) If experiment not terminated, go to Step 1)
Step 2) Prepare the return and summarize all experiments, generate final summary, and update solution for return
```

**Notes:**
1. **Minor Changes Only**: The experiment agent is restricted to updating only solution code parameters or make minor changes to code by specifying code diffs. 
Major code revisions like code re-structuring should generate new ideas and solution nodes with their own experiments. 
When environment exploration reveals issues that cannot be resolved by minor changes to code, 
the agent documents these insights in an experiment reflection (similar to an experimental report) and 
terminates the current experiment. The final summary includes these insights for the lead agent to generate 
improved ideas when extending the solution node.

2. **Callback Support**: The experiment agent relies on callbacks to interact with and explore the problem environment. 
To support this mechanism, users should implement callback registration in their evaluation script (`eval.py`) and 
clearly document the callback API in `callbacks_description.txt`.




## Code Update Specification
Code modifications is specified by conflict markers style. Parameter changes can also be specified in this way.
LLM should return a string with fenced block ```diff ... ``` inside that contains the code modifications.
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



## Callback Specification
Callbacks (or hooks) are functions provided to a library/framework that get called back at specific points during execution. 
Instead of calling the function directly, the evaluator invokes it when certain events occur. OR-Agent uses callbacks to explore problem environments.
For problems with complex environments, we recommend users implement callback registration support in their evaluation script (`eval.py`). 
This enables the experiment agent to explore the problem environment by registering callback functions and analyzing the output feedback.
Callback updates are specified by python fenced blocks.

**Example:**
User-provided evaluation script (`eval.py`):
```python
def evaluate(config, callbacks=None):
    '''Run traffic simulation
    Args:
        config: Simulation configuration
        callbacks: List of callback objects or None
    '''
    callbacks = callbacks or []

    # Helper to trigger callbacks
    def trigger(event_name, **kwargs):
        for callback in callbacks:
            if hasattr(callback, event_name):
                getattr(callback, event_name)(**kwargs)

    # Simulation logic with callbacks
    trigger('on_simulation_start', config=config)

    for step in range(config.max_steps):
        trigger('on_step_begin', step=step)

        # Your simulation logic
        pass

        trigger('on_step_end', step=step, metrics=metrics)

    trigger('on_simulation_complete')

if __name__ == "__main__":
    import sys
    config = sys.argv[1]  # command line argument
    result = evaluate(config, callbacks=[Callbacks()])
    print(result)
```

LLM can generate callback classes in fenced blocks:
```python
class Callbacks:
    def on_step_end(self, step, metrics):
        print(f"Step {step}: {metrics}")
```

The callback definition is passed to the evaluator as a callback object. The evaluator concatenates these definitions with the 
user-provided evaluation script (`eval.py`) to form a complete simulation script in `ExperimentAgent.run()`. 
If no callbacks update is provided, the previous callback definition will be used. 
If no callback definitions has been provided yet, then evaluator uses default callback definitions:
```python
class Callbacks:
    pass
```
The complete simulation script is executed by the evaluator, which triggers the callback functions at appropriate points. 
These functions collect data from the simulation, which is then used to identify issues and generate reflections for solution improvement.




## LLM API
1. Analysis (Required)
Wrap your detailed analysis in thinking tags. Include your assessment of the latest experiment results, identified issues, patterns, and potential improvements:
<thinking>
Your detailed analysis of the latest experiment results...
</thinking>

2. Code Updates (Optional)
If modifications are needed, provide a fenced diff block using conflict marker style. You may include multiple SEARCH/REPLACE pairs in a single diff block:
```diff
<<<<<<< SEARCH
...original code to replace...
=======
...new replacement code...
>>>>>>> REPLACE
<<<<<<< SEARCH
...another original code to replace...
=======
...another new replacement code...
>>>>>>> REPLACE
```

3. Callback Updates (Optional)
If callback modifications are needed, provide a fenced Python block:
```python
class Callbacks:
    def on_step_end(self, step, metrics):
        print(f"Step {step}: {metrics}")
```

4. Experiment Termination Decision (Required)
Finally, provide a JSON block with your termination decision:
```json
{
    "termination": "yes" or "no"
}
```
Note: The termination field must be either "yes" or "no".


For experiment summary, LLM client should provide a JSON response like:
```json
{
  "summary": "Comprehensive analysis synthesizing all experiment results, including key findings, persistent issues, successful strategies, and actionable recommendations for future improvements"
}
```
"""
import sys
import os
from pathlib import Path
import time
import yaml
import json
from datetime import datetime
import dataclasses
from typing import Union, List
from oragent.evaluator import Evaluator
import oragent.utils as utils
from oragent.utils import Solution



# ===== ExperimentAgent class =====
class ExperimentAgent:
    """Agent for conducting experiments."""
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
        self.reflect_llm_provider = self.config['model']['experiment_agent_reflect_llm_provider']
        self.reflect_model_name = self.config['model']['experiment_agent_reflect_model_name']
        self.reflect_llm_client = utils.LLMClient(config=self.config, llm_provider=self.reflect_llm_provider, model_name=self.reflect_model_name)
        self.summarize_llm_provider = self.config['model']['experiment_agent_summarize_llm_provider']
        self.summarize_model_name = self.config['model']['experiment_agent_summarize_model_name']
        self.summarize_llm_client = utils.LLMClient(config=self.config, llm_provider=self.summarize_llm_provider, model_name=self.summarize_model_name)
       
        # =====Vars updated during agent running=====
        if not checkpoint:
            #self.iteration = 0  # number of evolution rounds
            self.total_responses = 0  # Number of total responses; this can be used to track the number of LLM calls
            self.function_evals = 0  # Number of function evaluations; this is also an important metric for complexity, especially for the case when evaluation is the bottleneck
            self.valid_responses = 0 # Number of valid responses, namely responses that were successfully executed
        
        # =====Create evaluator=====
        self.evaluator = Evaluator(config=self.config)
        
        # =====ExperimentAgent settings=====
        self.max_experiment_repeats = self.config['max_experiment_repeats']  # max number of times to repeat experiments for a solution
        self.elitist_experiment_factor = self.config['elitist_experiment_factor']  # factor to multiply when doing experiment on elitist
        self.evaluation_description_disabled = self.config['evaluation_description_disabled']
        self.fast_exploration_for_crossover = self.config['fast_exploration_for_crossover']  # whether to use fast exploration for crossover
        self.verbose = self.config['verbose']
        
        # =====Load problem data and prompts=====
        # Problem and prompt directory
        self.problem_dir = f"{self.project_root}/problems/{self.problem}"
        self.prompt_dir = f"{self.project_root}/prompts"
        self.output_dir = self.config['output_dir'] or f"{self.project_root}/outputs/{self.algorithm}/{self.problem}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/details", exist_ok=True)  # folder to store details
        # Load common prompts
        self.system_experiment_reflection_prompt = utils.file_to_string(f'{self.prompt_dir}/system_experiment_reflection_oragent.txt')  # system role prompt for experiment reflection
        self.user_experiment_reflection_prompt = utils.file_to_string(f'{self.prompt_dir}/user_experiment_reflection_oragent.txt')  # user role prompt for experiment reflection
        self.system_experiment_summary_prompt = utils.file_to_string(f'{self.prompt_dir}/system_experiment_summary_oragent.txt')  # system role prompt for experiment summary
        self.user_experiment_summary_prompt = utils.file_to_string(f'{self.prompt_dir}/user_experiment_summary_oragent.txt')  # user role prompt for experiment summary
        # Load problem-specific prompts
        self.problem_description = utils.file_to_string(f'{self.problem_dir}/problem_description.txt')
        # `self.evaluation_description_disabled` decides whether eval description is used as context
        if os.path.exists(f'{self.problem_dir}/evaluation_description.txt') and not self.evaluation_description_disabled:
            self.evaluation_description = utils.file_to_string(f'{self.problem_dir}/evaluation_description.txt')
        else:
            self.evaluation_description = None
        self.function_description = utils.file_to_string(f'{self.problem_dir}/function_description.txt')
        if os.path.exists(f'{self.problem_dir}/callbacks_description.txt'):
            self.callbacks_description = utils.file_to_string(f'{self.problem_dir}/callbacks_description.txt')
        else:
            self.callbacks_description = None
        print("\n>>>[ExperimentAgent] ExperimentAgent initialized.")
    
    
    def reset(self):
        """Reset experiment agent state variables."""
        self.total_responses = 0  # Number of total responses; this can be used to track the number of LLM calls
        self.function_evals = 0  # Number of function evaluations; this is also an important metric for complexity, especially for the case when evaluation is the bottleneck
        self.valid_responses = 0 # Number of valid responses, namely responses that were successfully executed
        print(f"\n>>>[ExperimentAgent] ExperimentAgent reset.")
    
    
    def save(self, checkpoint: str):
        """
        Save checkpoint. Saved files:
        - experiment_agent_state.json

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
        state_path = os.path.join(checkpoint_directory, 'experiment_agent_state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=4, default=str)
            
        print(f"\n>>>[ExperimentAgent] Checkpoint saved to: {checkpoint_directory}")
        
    def load(self, checkpoint: str):
        """Load checkpoint."""
        checkpoint_directory = f'{self.project_root}/checkpoints/{checkpoint}'
        
        # Load config
        config_path = os.path.join(checkpoint_directory, 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load state variables
        state_path = os.path.join(checkpoint_directory, 'experiment_agent_state.json')
        with open(state_path, 'r') as f:
            state = json.load(f)

        # Restore state variables
        self.total_responses = state['total_responses']
        self.function_evals = state['function_evals']
        self.valid_responses = state['valid_responses']
        
        print(f"\n>>>[ExperimentAgent] Checkpoint loaded from: {checkpoint_directory}")
    
    
    def reflect(self, 
                parent_solutions: Union[Solution, List[Solution]], 
                solution: Solution, 
                long_term_reflection: str,
                #other_context=None,
                ):
        """
        Reflect on the most recent experiment result and determine the next action.
        Analyzes the latest experiment outcome to identify issues and generate improvement ideas. 
        Based on this analysis, determines whether to continue experimentation or terminate the optimization loop.
        
        Args:
            parent_solutions (Union[Solution, List[Solution]): parent solution(s).
            solution (Solution): The solution object.
            other_context (str): other context to add; like this is the last experiment.
        
        Returns:
            tuple: A tuple containing:
                - response (str): LLM raw response.
                - thinking (str): String of thoughts generated by LLM.
                - code_diff (str): Code diff block for updating the solution code.
                - callbacks (str): Callback class definitions as a python code string.
                - termination_dict (dict): Dictionary with 'termination' field indicating whether to terminate the experiment.
        """
        # convert parent solutions to string
        parent_solutions_str = utils.parents_to_str(parent_solutions)
        
        # Generate experiment history string by concatenating intermediate experiment (excluding the latest experiment) results and reflections
        experiment_history = solution.experiment_history()
                    
        # Construct prompt
        user = self.user_experiment_reflection_prompt.format(
            problem_description = self.problem_description,  # common
            function_to_evolve = self.function_to_evolve,  # common
            obj_type = self.obj_type,  # common
            evaluation_description = self.evaluation_description if self.evaluation_description else "(empty)",  # evaluation description
            function_description = self.function_description,  # common
            callbacks_description = self.callbacks_description if self.callbacks_description else "(callbacks NOT supported)",  # callbacks description for env exploration
            long_term_reflection = long_term_reflection,  # common
            parent_solutions = parent_solutions_str if parent_solutions_str else "(empty)", 
            idea = solution.idea,  # current idea
            code = solution.code,  # current solution code
            experiment_history = experiment_history if experiment_history else "(empty)",  # experiment history
            latest_experiment_result = solution.output,  # latest experiment result (errors will be included; metrics, features, and score will also be included; truncated if too long)
            #other_context = other_context if other_context else "(empty)",  # other things you may want to remind agent of; like this is the last experiment
        ) # Note: current callbacks is included in the `experiment_history`
        messages = [{"role": "system", "content": self.system_experiment_reflection_prompt}, {"role": "user", "content": user}]
        
        response = self.reflect_llm_client.chat(messages)
        self.total_responses += 1
        
        # Extract thoughts, code diff and callbacks
        thinking = utils.extract_thinking(response)  # string of reflection
        code_diff = utils.extract_diff_block(response)  # string of code diff
        callbacks = utils.extract_python_code(response)  # string of callback class definition
        termination_dict = utils.extract_json(response)  # dict with 'termination' field
        
        self.valid_responses += 1
        
        return response, thinking, code_diff, callbacks, termination_dict
    
    
    def summarize(self, 
                  solution: Solution, 
                  long_term_reflection: str
                  ) -> Solution:
        """
        Summarize the entire experiment history and identify improvement directions.
        Synthesizes all experiment results to extract key findings, persistent issues, and actionable insights for improving the solution. 
        
        Args:
            solution (Solution): The solution instance containing complete experiment history.
        
        Returns:
            str: A comprehensive summary including:
                - Key findings from all experiments
                - Identified issues and patterns
                - Recommended directions for solution improvement
        
        Note:
            This method analyzes the entire experiment history using LLM-based reasoning
            to generate a holistic summary, unlike _reflect() which focuses on the most
            recent experiment result.
        """
        # Generate experiment history string by concatenating intermediate experiment results and reflections
        experiment_history = solution.experiment_history()
    
        user = self.user_experiment_summary_prompt.format(
            problem_description = self.problem_description,  # common
            obj_type = self.obj_type,  # common
            function_to_evolve = self.function_to_evolve,  # common
            function_description = self.function_description,  # common
            long_term_reflection = long_term_reflection,  # common
            idea = solution.idea,  # current solution idea
            original_code = solution.intermediate_codes[0] if self.max_experiment_repeats > 0 else solution.code,  # original code
            experiment_history = experiment_history if self.max_experiment_repeats > 0 else 'None',  # all experiments
            current_code = solution.code,  # current code
            current_code_output = solution.output,  # current code output
        )
        messages = [{"role": "system", "content": self.system_experiment_summary_prompt}, {"role": "user", "content": user}]
        
        attempt = 0
        while True:
            attempt += 1
            
            # Invoke LLM to summarize
            response = self.summarize_llm_client.chat(messages)
            self.total_responses += 1
        
            response_extracted = utils.extract_json(response)
            if response_extracted:
                try:
                    reflection = response_extracted['summary']
                    self.valid_responses += 1
                except KeyError:
                    print(f"\n>>>[ExperimentAgent] Warning: JSON response does not contain 'summary' field (attempt {attempt}):\n{response}")
                break
            else:
                print(f"\n>>>[ExperimentAgent] Warning: LLM response could not be parsed as JSON (attempt {attempt}):\n{response}")
                if attempt >= 5:
                    raise RuntimeError(f"\n>>>[ExperimentAgent] Max attempts reached for summarizing solution")
        
        return reflection
    

    def run(self,
            parent_solutions: Union[Solution, List[Solution]], 
            solution: Solution, 
            long_term_reflection: Union[str, None]=None,
            use_long_term_reflection: bool=False,
            is_elitist: bool=False
            ):
        """
        Conducts experiments on candidate solutions to evaluate their performance. 
        Uses callbacks to identify issues during execution and enables tuning of code parameters based on experimental results. 
        Both intermediate reflections and a final summary are generated through LLM client invocations.

        Args:
            parent_solutions (Union[Solution, List[Solution]): parent solution(s).
            solution (Solution): solution that only contains the idea and code yet
            long_term_reflection (str): long term reflection from previous experiments
            use_long_term_reflection (bool): whether elitist is used as root solution; default False.
            is_elitist (bool): whether current solution is elitist; default False.
            
        Returns:
            solution (Solution): the updated solution; updated fields include:
             - code parameters are possibly updated (revisions generated by `reflect()`);
             - intermediate results are added (generated by `evaluator.run()`); 
             - intermediate reflections are added (generated by `reflect()`);
             - metrics are added (generated by `evaluator.run()`);
             - features are added (generated by `evaluator.run()`);
             - score is added (generated by `evaluator.run()`);
             - summary is added (generated by `summarize()`).
        """
        print(f"\n>>>[ExperimentAgent] Starts to work on solution {solution.id_str(algorithm=self.algorithm)}...")
        
        # =====0. Experiment Preparation=====
        # Handle long-term reflection for crossover
        # for mutation on elitist, we always use long-term reflection
        # for crossover, we may not want to provide long-term reflection
        if not use_long_term_reflection:
            long_term_reflection = "None"
            print("\n>>>[ExperimentAgent] Long-term reflection is not used.")
        else:
            print("\n>>>[ExperimentAgent] Long-term reflection is used.")
        
        long_term_reflection = long_term_reflection if long_term_reflection else "None"
        
        # Clear intermediate results in solution
        solution.intermediate_codes = []  # code
        solution.intermediate_outputs = []  # eval std output (including error info)
        solution.intermediate_metrics = []  # metrics extracted from output
        solution.intermediate_features = []  # features extracted from output
        solution.intermediate_scores = []  # score extracted from output
        solution.intermediate_actions = []  # actions by experiment agent


        # =====1. Experiment loop=====
        # Variables to be updated during experiment loop
        curr_callbacks = None
        raw_output = None
        metrics = None
        features = None
        score = None
        termination = 'no'
        code_diff = None
        
        # we choose to spend on more time doing experiments for elitist solution
        # TODO: "how to fast explore in experiments?" - This requires further exploration, analysis, and validation, and is marked with a TODO flag
        if is_elitist:
            max_experiment_repeats = int(self.max_experiment_repeats * self.elitist_experiment_factor)
        else:
            max_experiment_repeats = self.max_experiment_repeats
        
        experiment_count = 0
        while experiment_count < max_experiment_repeats:  # when self.max_experiment_repeats is 0, no experiment is conducted, directly go to summarization step
            experiment_count += 1
            
            # -----1.1. Evaluate the code-----
            # output, metrics, features, score are updated
            raw_output, metrics, features, score = self.evaluator.run(solution=solution, 
                                                                      callbacks=curr_callbacks)
            if experiment_count != 1:  # the 1st evaluation is actually not needed; solutions are already evaluated when they're passed to experiment agent; 
                # we retain the evaluation here just for robustness when doing further development
                self.function_evals += 1
                
            processed_output = utils.truncate(raw_output)  # raw output could be very long, truncate it to avoid context window overflow
            print(f"\n>>>[ExperimentAgent] Experiment {experiment_count} output:\n{processed_output}\n")
            
            # Update solution fields
            solution.output = processed_output
            solution.metrics = metrics
            solution.features = features
            solution.score = score
            
            # -----1.2. Reflect on the evaluator output and invoke LLM client to generate the next action-----
            # Prepare `other_context`
            '''deprecated
            last_try = (experiment_count == self.max_experiment_repeats)
            if not last_try:
                other_context = f"Currently we're conducting experiment #{experiment_count}, namely the 'latest experiment' is experiment #{experiment_count}."
            else:
                other_context = f"""Important: this is the final attempt (experiment #{experiment_count}) to improve the solution!
You may want to revert back to the previous version of the solution code or parameter configurations if you are not satisfied with the latest attempt.
Give it your best shot and make sure that the last solution code is executable."""
            '''
            
            # Reflect
            response, thinking, code_diff, callbacks, termination_dict = self.reflect(
                                                                        parent_solutions=parent_solutions,
                                                                        solution=solution, 
                                                                        long_term_reflection=long_term_reflection, 
                                                                    ) 
            
            # -----1.3. Save intermediate vars before making changes-----
            solution.intermediate_codes.append(solution.code)
            solution.intermediate_outputs.append(solution.output)
            solution.intermediate_metrics.append(solution.metrics)
            solution.intermediate_features.append(solution.features)
            solution.intermediate_scores.append(solution.score)
            solution.intermediate_actions.append(response)
            # you may want to also store intermediate solution code 
            # TODO: in case the last try fails, you can revert back to the last working solution manually
            # "manually revert code back at the end of experiment" - This requires further exploration, analysis, and validation, and is marked with a TODO flag
            
            # -----1.4. Take actions-----
            if not thinking:
                print(f"\n>>>[ExperimentAgent] Warning: no valid thinking extracted. Response:\n{response}\n")
                # We don't raise error here; it may be handled by next round of experiment
            else:
                print(f"\n>>>[ExperimentAgent] Thinking:\n{thinking}\n")
                
            if termination_dict:
                if "termination" in termination_dict:
                    termination = termination_dict['termination']
                    print(f"\n>>>[ExperimentAgent] Termination: {termination}\n")
                else:
                    print(f"\n>>>[ExperimentAgent] Warning: no valid termination field in response: \n{response}\n")
            else:
                # We don't raise error here; it may be handled by next round of experiment
                # by default, we keep termination to 'no'
                print(f"\n>>>[ExperimentAgent] Warning: cannot extract termination field in response: \n{response}\n")
            
            if code_diff:
                tmp_code = utils.update_code(solution.code, code_diff)
                if tmp_code:
                    # Code changes updated to solution code
                    solution.code = tmp_code
                    # Log code diff
                    print(f"\n>>>[ExperimentAgent] action after experiment {experiment_count}: update code\n")
                    file_name = f"{self.output_dir}/details/{solution.id_str(self.algorithm)}_codediff{experiment_count}.txt"
                    with open(file_name, 'w') as file:
                        file.write(code_diff)
                    print(f"Code diff saved to: {file_name}")
                    if self.verbose:
                        print(">>>[ExperimentAgent] Code diff:")
                        print(code_diff)
                else:
                    print(f"\n>>>[ExperimentAgent] Warning: code update failed\ncode diff:\n{code_diff}\n")
            
            if callbacks:
                # if new callbacks are generated, we update them
                # `curr_callbacks` updated and will be executed in the next iteration
                curr_callbacks = callbacks 
                print(f"\n>>>[ExperimentAgent] action after experiment {experiment_count}: update callbacks\n")
                # Log callbacks to local file instead of printing out since it's possibly long
                file_name = f"{self.output_dir}/details/{solution.id_str(self.algorithm)}_callbacks{experiment_count}.txt"
                with open(file_name, 'w') as file:
                    file.write(callbacks) 
                print(f"New callbacks definition saved to: {file_name}")
                if self.verbose:
                    print(">>>[ExperimentAgent] Callbacks:")
                    print(callbacks)
            else:
                print("\n>>>[ExperimentAgent] Warning: no callbacks update")
                
            # If the experiment is not terminated, revise the solution code or add callbacks to the code, repeat Step 1); otherwise, break;
            if termination.lower().strip() == "yes":
                print(f"\n>>>[ExperimentAgent] action after experiment {experiment_count}: terminate\n")
                break
            
            # -----1.5. Check max_experiment_repeats-----   
            if experiment_count >= max_experiment_repeats:
                print(f"\n>>>[ExperimentAgent] Reached max_experiment_repeats ({max_experiment_repeats}). Terminating experiment.\n")
                break


        # =====2. Prepare the return and summarize=====
        # -----Evaluate again before return; since the code may have been updated before breaking the loop-----
        if code_diff:
            raw_output, metrics, features, score = self.evaluator.run(solution)
            self.function_evals += 1
            processed_output = utils.truncate(raw_output)  # raw output could be very long, truncate it to avoid context window overflow

            # Prepare return
            solution.output = processed_output
            solution.metrics = metrics
            solution.features = features
            solution.score = score
            
            solution.intermediate_codes.append(solution.code)
            solution.intermediate_outputs.append(solution.output)
            solution.intermediate_metrics.append(solution.metrics)
            solution.intermediate_features.append(solution.features)
            solution.intermediate_scores.append(solution.score)
            solution.intermediate_actions.append(None)  # the last evaluation has no response hence we append None; just to keep all intermediate vars having same length
                    
        # -----Additional debugging step if final solution is not valid-----
        # TODO: we could add an additional debugging step here
        # Instead, we moved this step to LeadAgent at location where ExperimentAgent just returns solution
        # so that the workflow is simpler - only central LeadAgent calls other agents
        # alternatives are possible
        # “additional debugging stage before experiment return” - This requires further exploration, analysis, and validation, and is marked with a TODO flag
        
        # -----Manually revert back to previous solution-----
        # here we revert back to previous best solutions
        # Find the index of the best score (maximum if self.obj_type == 'max', minimum if self.obj_type == 'min')
        # Handle None values by filtering them out
        valid_scores = [(i, score) for i, score in enumerate(solution.intermediate_scores) if score is not None]

        if not valid_scores:
            # If all scores are None, use the current solution; for example, experiment 0 times
            best_sol_idx = None
        elif self.obj_type == 'max':
            best_sol_idx = max(valid_scores, key=lambda x: x[1])[0]
        else:  # self.obj_type == 'min'
            best_sol_idx = min(valid_scores, key=lambda x: x[1])[0]

        # Update `solution` using this intermediate solution
        # Note: best_sol_idx may be 0, and 0 is valid index; don't use `if best_sol_idx`
        if (solution.score is None and best_sol_idx is not None) \
            or (best_sol_idx != None and self.obj_type == 'max' and solution.intermediate_scores[best_sol_idx] > solution.score) \
            or (best_sol_idx != None and self.obj_type == 'min' and solution.intermediate_scores[best_sol_idx] < solution.score):
            print(f"\n>>>[ExperimentAgent] Revert back to previous code version | score reverted from {solution.score} to {solution.intermediate_scores[best_sol_idx]}")
            solution.code = solution.intermediate_codes[best_sol_idx]
            solution.output = solution.intermediate_outputs[best_sol_idx]
            solution.metrics = solution.intermediate_metrics[best_sol_idx]
            solution.features = solution.intermediate_features[best_sol_idx]
            solution.score = solution.intermediate_scores[best_sol_idx]
            
            '''deprecated
            # for convenience of summary, we will abort intermediate experiment results afterwards
            # alternatively, we could have make use of the information in those failed experiments
            solution.intermediate_codes = solution.intermediate_codes[:best_sol_idx]
            solution.intermediate_outputs = solution.intermediate_outputs[:best_sol_idx]
            solution.intermediate_metrics = solution.intermediate_metrics[:best_sol_idx]
            solution.intermediate_features = solution.intermediate_features[:best_sol_idx]
            solution.intermediate_scores = solution.intermediate_scores[:best_sol_idx]
            solution.intermediate_actions = solution.intermediate_actions[:best_sol_idx]
            '''
        
        # -----Summary-----
        solution.summary = self.summarize(solution=solution, 
                                          long_term_reflection=long_term_reflection,
                                          )
        print(f"\n>>>[ExperimentAgent] Final summary after {experiment_count} experiments:\n{solution.summary}\n")
        
        # -----Clear intermediate vars after experiment to save space-----
        solution.intermediate_codes = []  # we recommend to clear intermediate code at least since they may consume too much space for complex problem; other intermediate variables can be cleared according to your needs
        #solution.intermediate_outputs = []
        #solution.intermediate_actions = []
        #solution.intermediate_metrics = []
        #solution.intermediate_features = []
        #solution.intermediate_scores = []
        
        return solution







if __name__ == '__main__':
    # For debugging purposes
    # Create seed solution and conduct experiment
    experiment_agent = ExperimentAgent()
    seed_solution = Solution()
    seed_solution.lead_agent_id = 0
    seed_solution.research_round = 0
    seed_solution.solution_count = 0
    seed_solution.idea = utils.file_to_string(f"{experiment_agent.problem_dir}/seed_solution_idea.txt")
    seed_solution.code = utils.file_to_string(f"{experiment_agent.problem_dir}/seed_solution.py")
    
    # Invoke experiment agent
    updated_solution = experiment_agent.run(seed_solution)
    print("Updated solution:")
    print(str(updated_solution))