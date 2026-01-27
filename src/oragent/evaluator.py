# src/oragent/evaluator.py
"""
# Evaluator 
Evaluator class is used for evaluating solution functions in the OR-Agent framework.

## Usage
1. Place problem data in `<project_root>/problems/<your_problem>/`
2. Provide an evaluation script at `<project_root>/problems/<your_problem>/eval.py`
3. Provide a default callbacks script at `<project_root>/problems/<your_problem>/default_callbacks.py`
The `eval.py` script imports the solution function ("function_to_evolve") and will print out the results.



## Implementation Details
The `run` method takes a `Solution` instance and optional `callbacks` string.
`solution.code` contains the code to evaluate.

Process:
    Step 0. Reads the evaluation script from `<project_root>/problems/<your_problem>/eval.py`
    Step 1. ("Prepare local scripts to run") Assumes `eval.py` imports the target function via `import seed_solution`; modifies `eval.py` to import the local solution module instead
        Stores the revised eval script and solution code in `<project_root>/outputs/oragent/<your_problem>/`
    Step 2. ("Run local scripts") Executes the revised eval file using `subprocess.run`
    Step 3. ("Parse stdout") Extracts results from stdout: raw_output, metrics, features, and score



## Eval script command line arguments requirements (same for all problems)
Command line arguments:
1. `root_dir`: the project root directory; knowing project root can help you to load data; default: current working directory (os.getcwd());
    Eval script need this to load dataset since eval script may be generated and stored in a different location to support parallelism;
2. `file_output_prefix`: the output file prefix: this prefix can be used to save output files during evaluation for inspection purposes; 
    we use prefix since you may want file like: `lead1_round0_count0_id0_file_collisions.xml` for better inspection.
    absolute path is recommended;
    default: '', which means just save to current working directory;
3. `mode`: train or val; default: val; not used by `Evaluator`;
4. `problem_size`; not used by `Evaluator`;

You can manually run the eval script this way:
```
python eval.py \
    --root_dir=<path_to_project_root> \
    --file_output_prefix=<path_to_output_file> \
    --mode=val \
    --problem_size=50
```

`Evaluator` class will run the eval script like this:
```
subprocess.run([
                'python', 'script.py',
                '--root_dir', '/path/to/project',
                '--file_output_prefix', '/path/to/outputs/exp1_',
            ], 
            text=True,
            timeout=self.timeout_seconds,  # timeout seconds
            cwd=os.getcwd(),
            env=env,  # python env
            stdout=f, 
            stderr=f
            )
```
Note:
Evaluator won't specify `mode` and `problem_size`; 
Since evaluator is intended for general purpose, we assume it does not know any problem detail.
This makes it easier for you to add new problems - you don't need to revise the `Evaluator` class.


## Eval script output requirements (same for all problems)
Eval script should print out `metrics`,`features`, and `score`;
1. `metrics`: `metrics` is used only as prompt context and user inspection. In principle any metrics type is supported.
    We recommend `metrics` to take one of two following forms:
    a dict that map test name (str) to metrics (Dict); (metric name -> value); type: Dict[str, Dict[str, float]]
    or an aggregated dictionary mapping metric names directly to values, e.g. average test metrics; type: Dict[str, float]
    It's optional but we strongly recommend you to prepare a detailed metrics dict for each problem,
    as this can help LLM to better understand the solution performance!
2. `features`: a tuple of ints that represents the features of the solution; 
    `features` tuple is used for solution storage in the solution database; 
    Features is generally generated from metrics, possibly with some added feature; 
    but Evaluator will not assume any conversion method; you need to specify it yourself.
    Features could be set to `None` if you don't want to specify feature; in that case MAP-Elite will be disabled;
3. `score`: a float that represents the score of the solution;
    `score` is used for as the fitness score.
    It's required. It's usually generated from metrics; but Evaluator will not assume any conversion method; you need to specify it yourself.

Example:
Assume the following variables are generated during eval script:
```
metrics = {
    "critical_ttc_count": 28, 
    "collisions": 0, 
    "emergencyStops": 0, 
    "emergencyBraking": 4, 
    "teleports": 0, 
    "avg_speed": 12.51, 
    "speed_variance": 16.22
}
features = (2, 0, 1, 4)
score = 12.34
```

Then eval script should print to stdout by:
```
...
__SANDBOX_RESULT__

__METRICS_START__
<print(repr(metrics))>
__METRICS_END__

__FEATURES_START__
<print(repr(features))>
__FEATURES_END__

__SCORE_START__
<print(repr(score))>
__SCORE_END__

__SANDBOX_SUCCESS__
```
Note: Evaluator support Python float and numpy float like:
np.float64(4.450265933717864)
for other types of number, you need to cast before repr().


## Eval script dynamic solution function loading
Solution function scripts will be generated on the fly and loaded dynamically.
To enable parallelism, we will save different solution script to different files. Hence Evaluator will need to load the solution script dynamically.
Keep the line below unchanged:
```
import seed_solution as solution_module
```
Say, one solution script is generated as `solution_0905.py`;
Then the above line will be replaced by:
```
import solution_0905 as solution_module
```
and the new eval script will be saved locally and got run.
"""
import sys
import os 
from pathlib import Path
import yaml
import socket
import getpass
import numpy as np
from typing import Any, Dict, Tuple, Union
import subprocess
import multiprocessing
import oragent.utils as utils
from oragent.utils import Solution


def to_python_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types.
    
    Handles:
    - numpy scalars (float64, int64, etc.) -> Python float/int
    - tuples -> tuple with converted elements
    - dicts -> dict with converted values
    - None -> None
    - already Python types -> unchanged
    
    Examples:
        >>> to_python_types(np.float64(7.5))
        7.5
        >>> to_python_types((np.int64(4), np.int64(7), np.int64(9)))
        (4, 7, 9)
        >>> to_python_types({'a': np.float64(1.5), 'b': np.float64(2.5)})
        {'a': 1.5, 'b': 2.5}
        >>> to_python_types({'test1': {'acc': np.float64(0.9)}, 'test2': {'acc': np.float64(0.8)}})
        {'test1': {'acc': 0.9}, 'test2': {'acc': 0.8}}
    """
    # Handle None
    if obj is None:
        return None
    
    # Handle numpy scalars
    if isinstance(obj, np.generic):
        return obj.item()
    
    # Handle tuples
    if isinstance(obj, tuple):
        return tuple(to_python_types(item) for item in obj)
    
    # Handle lists
    if isinstance(obj, list):
        return [to_python_types(item) for item in obj]
    
    # Handle dicts (recursively converts values)
    if isinstance(obj, dict):
        return {key: to_python_types(value) for key, value in obj.items()}
    
    # Return as-is for native Python types
    return obj


def _evaluate_solution_wrapper(evaluator, solution):
    """
    Module-level wrapper function for parallel evaluation.
    Must be at module level to be picklable for multiprocessing.
    """
    try:
        raw_output, metrics, features, score = evaluator.run(solution)
        return solution, raw_output, metrics, features, score
    except Exception as e:
        print(f"\n>>>[Evaluator] Error evaluating solution {solution.id_str(evaluator.algorithm)}: {e}")
        return solution, None, None, None, None


class Evaluator:
    """Evaluate solution."""

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
            
        # =====Experiment settings=====
        self.timeout_seconds = self.config['evaluation']['timeout_seconds']  # timeout seconds for 
        # Get username and hostname
        username = getpass.getuser()
        hostname = socket.gethostname()
        # Combine them
        #user_host = f"{username}@{hostname}"
        user_host = f"{username}"
        self.python_path = self.config['evaluation']['python_path'][user_host]  # python path
        self.eval_env_vars = self.config['evaluation']['eval_env_vars']  # python environment vars
        self.verbose = self.config['verbose']
        
        # =====Load problem scripts=====
        self.problem_dir = f"{self.project_root}/problems/{self.problem}"
        #self.prompt_dir = f"{self.project_root}/prompts"
        # output directory; Note: use relative path!
        #self.output_dir = self.config['output_dir'] or f"{self.project_root}/outputs/{self.algorithm}/{self.problem}"
        self.output_dir = self.config['output_dir'] or f"outputs/{self.algorithm}/{self.problem}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/details", exist_ok=True)  # folder to store details
        # Load eval file as string
        self.eval_code = utils.file_to_string(f'{self.problem_dir}/eval.py')
        # Load default callbacks as string
        if os.path.exists(f'{self.problem_dir}/default_callbacks.py'):
            self.default_callbacks = utils.file_to_string(f'{self.problem_dir}/default_callbacks.py')
        else:
            self.default_callbacks = None
        print("\n>>>[Evaluator] Evaluator initialized.")
        
        
    def save(self, checkpoint:str):
        raise NotImplementedError
        
    def load(self, checkpoint: str):
        raise NotImplementedError
        
            
    def run(self, solution: Solution, callbacks: str=None):
        """
        Evaluate code in parallel and computing objective values.
        Run code string and return result string.
        Invoke the `_run_code` method for each individual in the population.
        
        Args:
            code: the code of the function to evolve.
            callbacks: the callback class definition.
        
        Returns:
            raw_output (str): the raw string of the code execution output;
            metrics (Dict): metrics extracted from the raw_output;
            features (List[int]): features extracted from the raw_output;
            scores (float): score extracted from the raw_output.
        """
        # =====1. Prepare local scripts to run=====
        # Add imports to code for safe execution
        solution_code = utils.add_imports_to_code(solution.code)
        
        # -----Save the solution code to local file-----
        # Note: save the code into the same file for execution is not a good choice; Racing condition happens!
        # Instead, we let each process save its code to a separate file. In that case, we need to modify `eval.py` to import the correct module!
        individual_output_module = f"{solution.id_str(self.algorithm)}_solution"  # eval script will import this module
        individual_output_file =  f"{self.output_dir}/details/{solution.id_str(self.algorithm)}_solution.py"  # the file to save the code
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/details", exist_ok=True)  # folder to store details
        with open(individual_output_file, 'w') as file:
            file.writelines(solution_code)
        solution.code_filepath = individual_output_file  # Update the code path field of solution

        # -----Revise eval script to add callbacks and import the local solution code instead-----
        callbacks = callbacks or self.default_callbacks
        if not callbacks:
            callbacks = ""  # callbacks NOT supported
        eval_code = self.eval_code.replace("import seed_solution", "import " + individual_output_module)
        eval_code = callbacks + "\n\n" + eval_code
        
        # -----Create a temporary eval file to save the revised eval script and store at the same directory as the solution code-----
        revised_eval_file_path = f'{self.output_dir}/details/{solution.id_str(self.algorithm)}_eval.py'
        # Note: eval file should be the same with the "gpt" module file
        with open(revised_eval_file_path, 'w') as f:
            f.write(eval_code)
        # TODO: solution code and eval script are written to output directory in this version of OR-Agent
        # this makes it easier to debug; but indeed takes much storage
        # For space efficiency, users may rewrite the same file(s)
        # "devise storage saving technique" - This requires further exploration, analysis, and validation, and is marked with a TODO flag
        
        # =====2. Run local scripts=====
        # customize the running env for your machine
        # say, specify python executable path add env variables to env if needed
        env = os.environ.copy()
        if self.eval_env_vars:
            env.update(self.eval_env_vars)
        # In case eval script needs to output some files, we provide an `output_prefix` for better logging
        file_output_prefix = f"{self.output_dir}/details/{solution.id_str(self.algorithm)}_file_"  # use relative path!
        # Stdout during execution is also saved to local file
        output_filepath = f"{self.output_dir}/details/{solution.id_str(self.algorithm)}_stdout.txt"
        solution.output_filepath = output_filepath  # update solution field
        
        with open(output_filepath, 'w') as f:
            # run the python script 
            # redirect output to file
            # alternatively, you can captures the output internally by setting `capture_output=True` for subprocess.run
            # and access the stdout via process.stdout and process.stderr
            try:
                process = subprocess.run([self.python_path, '-u', revised_eval_file_path,  # the script to run; -u means unbuffered output
                                            '--root_dir', self.project_root,               # sys.argv[1]: self.project_root is the project root
                                            '--file_output_prefix', file_output_prefix,    # sys.argv[2]: if program need to write files locally, use this prefix
                                            ],
                                            text=True,
                                            timeout=self.timeout_seconds,  # timeout seconds
                                            cwd=os.getcwd(),
                                            env=env,  # python env
                                            stdout=f, 
                                            stderr=f
                                        )
            except Exception as e:
                f.write(f"{e}")

        # =====3. Parse stdout=====
        # Initialize vars to return
        raw_output = None
        metrics = None
        features = None
        score = None
        # Parse execution results
        try:
            stdout_str = utils.file_to_string(output_filepath)
            raw_output = stdout_str
            
            if self.verbose:
                print("\n>>>[Evaluator] stdout:")
                print(utils.truncate(stdout_str))
            
            # Parse the output
            start_marker = "__SANDBOX_RESULT__"
            end_marker = "__SANDBOX_SUCCESS__"
            if end_marker in stdout_str:
                # Extract the result
                if start_marker in stdout_str:
                    start_idx = stdout_str.find(start_marker) + len(start_marker)
                    end_idx = stdout_str.find(end_marker)
                    result_str = stdout_str[start_idx:end_idx].strip()
                    try:
                        # Parse metrics
                        tmp_start_marker = "__METRICS_START__"
                        tmp_end_marker = "__METRICS_END__"
                        tmp_start_idx = result_str.find(tmp_start_marker) + len(tmp_start_marker)
                        tmp_end_idx = result_str.find(tmp_end_marker)
                        tmp_str = result_str[tmp_start_idx:tmp_end_idx].strip()
                        metrics = eval(tmp_str)  # a dict or None
                        metrics = to_python_types(metrics)  # in case numpy is used, convert to python number type
                        
                        # Parse features
                        tmp_start_marker = "__FEATURES_START__"
                        tmp_end_marker = "__FEATURES_END__"
                        tmp_start_idx = result_str.find(tmp_start_marker) + len(tmp_start_marker)
                        tmp_end_idx = result_str.find(tmp_end_marker)
                        tmp_str = result_str[tmp_start_idx:tmp_end_idx].strip()
                        features = eval(tmp_str)  # a tuple of ints or None
                        features = to_python_types(features)  # in case numpy is used, convert to python number type
                        
                        # Parse score
                        tmp_start_marker = "__SCORE_START__"
                        tmp_end_marker = "__SCORE_END__"
                        tmp_start_idx = result_str.find(tmp_start_marker) + len(tmp_start_marker)
                        tmp_end_idx = result_str.find(tmp_end_marker)
                        tmp_str = result_str[tmp_start_idx:tmp_end_idx].strip()
                        score = eval(tmp_str)  # a float
                        score = to_python_types(score)  # in case numpy is used, convert to python number type
                        
                        print(f"\n>>>[Evaluator] Metrics extracted for solution (id): {solution.id_str(self.algorithm)}")
                        print("metrics:", metrics)
                        print("features:", features)
                        print("scores:", score)                              
                    except Exception as e:  # if not a valid Python expression
                        # If eval fails, return the string representation
                        error_string = f"\n>>>[Evaluator] {solution.id_str(self.algorithm)} - Fail to parse the result:\n Error:{e}" + "\n result_str:\n" + result_str
                        print(error_string)
                else:  # if start marker not found
                    error_string = f"\n>>>[Evaluator] {solution.id_str(self.algorithm)} - Error: no start marker found in the result:\n" + stdout_str
                    print(error_string)
            else:  # if end marker not found
                error_string = f"\n>>>[Evaluator] {solution.id_str(self.algorithm)} - Error: no end marker found in the result:\n" + stdout_str
                print(error_string)
        except Exception as e:
            print(f"\n>>>[Evaluator] {solution.id_str(self.algorithm)} - Error occurred: {e}")
                
        return raw_output, metrics, features, score


    # =====Method used by ReEvo, RoH, AEL, FunSearch=====
    def evaluate_individual(self, individual: Solution) -> Solution:
        """
        Evaluate an individual.
        
        Args:
            individual (Solution): The individual to evaluate
        
        Returns:
            returns a list of dictionaries (the updated population) where each individual dictionary now contains:
            - exec_success: Boolean indicating if code execution succeeded
            - obj: Float representing the objective value (minimized or maximized based on obj_type)
            - traceback_msg: String with error message if execution failed (only present on failure)
            - Original fields: output_filepath, code_filepath, code, response_id

            The method evaluates each individual by running their code, captures stdout/stderr, computes objective values, 
            and marks invalid individuals with infinite objective values.
        """
        if individual.code is None:
            print(f"\n>>>[Evaluator] Unable to evaluate solution: {individual.id_str(self.algorithm)}\nNo code.")
            return individual
        
        print(f"\n>>>[Evaluator] Evaluating solution: {individual.id_str(self.algorithm)}")
        
        raw_output, metrics, features, score = self.run(individual)
        
        if score is not None:
            individual.metrics = metrics
            individual.features = features
            individual.score = score
            individual.output = raw_output
            print(f"\n>>>[Evaluator] Evaluation successful: {individual.id_str(self.algorithm)}\nObj: {score}")
        else:
            print(f"\n>>>[Evaluator] Unable to evaluate solution: {individual.id_str(self.algorithm)}\nRaw output: \n{raw_output}")
                
        return individual
    
    # Kept for compatibility
    def evaluate_population(self, population: list[Solution]) -> list[Solution]:
        """
        Evaluate population by running code in parallel and computing objective values.
        
        Args:
            population (list[Solution]): The population to evaluate
        
        Returns:
            population
        """
        population_updated = []
        
        for ind in population:
            population_updated.append(self.evaluate_individual(ind))
                
        return population_updated
    
    
    def evaluate_population_parallel(self, population: list[Solution]) -> list[Solution]:
        """
        Multi-processing version of `evaluate_population` method.

        Args:
            population (list[Solution]): The population to evaluate

        Returns:
            list[Solution]: The updated population with evaluation results
        """
        # Filter out invalid solutions
        valid_population = [ind for ind in population if ind.code is not None]

        if not valid_population:
            return population

        print(f"\n>>>[Evaluator] Evaluating {len(valid_population)} solutions in parallel...")


        # Use multiprocessing Pool with partial function
        # uses Python's multiprocessing.Pool to execute the _evaluate_solution_wrapper function in parallel across multiple CPU cores.
        # multiprocessing.Pool() creates a pool of worker processes (defaults to number of CPU cores)
        # pool.map(func, valid_population) distributes the valid_population list across workers
        # Each worker calls _evaluate_solution_wrapper() with one solution from the list
        # Results are collected back in the same order as the input list
        with multiprocessing.Pool() as pool:
            # Use partial to bind self to the wrapper function
            from functools import partial
            func = partial(_evaluate_solution_wrapper, self)
            results = pool.map(func, valid_population)

        # Extract and update solutions from results
        # Note: multiprocessing pickles solution objects and sends copies to worker processes
        # When a Solution object is pickled and unpickled, it creates a new instance with the same field values.
        updated_solutions = []
        for solution, raw_output, metrics, features, score in results:
            if score is not None:
                solution.metrics = metrics
                solution.features = features
                solution.score = score
                solution.output = raw_output
                print(f"\n>>>[Evaluator] Evaluation successful: {solution.id_str(self.algorithm)}\nObj: {score}")
            else:
                print(f"\n>>>[Evaluator] Unable to evaluate solution: {solution.id_str(self.algorithm)}\nRaw output: \n{raw_output}")
            updated_solutions.append(solution)

        # Return updated solutions (invalid solutions remain unchanged)
        # Map back to original population order
        solution_map = {s.id: s for s in updated_solutions}
        return [solution_map.get(s.id, s) for s in population]






if __name__ == '__main__':
    # For debugging purposes
    print("========Test evaluator=========")
    project_root = Path(__file__).parent.parent  
    #code = utils.file_to_string(f"{project_root}/problems/driving/seed_solution.py")
    code = utils.file_to_string(f"{project_root}/problems/tsp_constructive/seed_solution.py")
    solution = Solution(code=code)
    
    evaluator = Evaluator()
    
    raw_output, metrics, features, score = evaluator.run(solution=solution)
    print("===results:===")
    print("-----raw_output:-----")
    print(raw_output)
    print("-----metrics:-----")
    print(metrics)
    print("-----features:-----")
    print(features)
    print("-----score:-----")
    print(score)
    
    
