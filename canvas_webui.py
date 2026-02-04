# canvas_webui.py
""" 
# Open Research Canvas
User can use Open Research Canvas to:
1) define research problem (in "Problem Definition Section"), 
2) start research agent (in "Agent Control Section") and 
3) monitor research progress (in "Agent Output Section").


## Layout illustration
+------------------------------------------------------------+
|      Open Research Canvas                                  |       Header Section
+------------------------------------------------------------+
|  Problem Description  [Load Pbm][Submit Btn][LLM-gen Btn]  |     
|  +--------+ +-----------------------+ +-----------------+  |
|  |Pbm Name| |Pbm Description        | | Ext knowledge   |  |       Problem Definition Section - Subsection 1) Problem Description
|  +--------+ +-----------------------+ +-----------------+  |
|                                                            |       
|  Evaluation Description        [Submit Btn] [LLM-gen Btn]  |             
|  +------------------+ +---------------------------------+  |
|  |Function Signature| |      Evaluation Script          |  |       Problem Definition Section - Subsection 2) Evaluation Description
|  +------------------+ +---------------------------------+  |
|                                                            |  
|  Test Dataset          [Submit Btn][LLM-gen Btn][Run Btn]  |  
|  +------------------+ +---------------------------------+  |
|  |Data Files Visual | |       Dataset Generation Script |  |       Problem Definition Section - Subsection 3) Test Dataset Generation
|  +------------------+ +---------------------------------+  |
|                                                            |  
|  Seed Solution          [Submit Btn][LLM-gen Btn][Run Btn] |
|  +-----------------+ +----------------------------------+  |
|  |Seed Sol Idea    | |      Seed Solution Script        |  |       Problem Definition Section - Subsection 4) Seed Solution Generation
|  +-----------------+ +----------------------------------+  | 
|                                                            |
|  Agent Control                                             |
|  +------------------------------------------------------+  |
|  |[Gen config][Select problem][Select alg][Start][Stop] |  |       Agent Control Section
|  +------------------------------------------------------+  |
|                                                            |     
|  Agent Output                                              |
|  +-------------+  +-------------------------------------+  |
|  |   Solution  |  |                                     |  |
|  | Performance |  |                                     |  |
|  |     Info    |  |            Real-time                |  |
|  +-------------+  |            System Messages          |  |       Agent Output Section - Subsection 1) Metrics, Config, Progress & Messages Display
|  +-------------+  |            Scroll Window            |  |
|  |   Research  |  |                                     |  |
|  |   Progress  |  |                                     |  |
|  +-------------+  +-------------------------------------+  |
|  +------------------------------------------------------+  |
|  |         Long-term reflections (optional)             |  |       Agent Output Section - Subsection 2) Long-Term Reflection Display
|  |         User feedback input                          |  |
|  +------------------------------------------------------+  |
|  +------------------------------------------------------+  |
|  |         Solution Database Info (optional)            |  |       Agent Output Section - Subsection 3) Solution Database Display
|  |                                                      |  |
|  +------------------------------------------------------+  |
+------------------------------------------------------------+
|        Developer | Terms of Use | Contact                  |       Footer section
+------------------------------------------------------------+



## Functionality

### Problem Definition Section
Problem Definition Section is the place where user and LLM collaborate to define the research problem, including:
 - problem description
 - evaluation description
 - test dataset generation
 - seed solution generation

**Problem Definition Section - Subsection 1) Problem Description**
This subsection has a drop-down list to select a problem from the list of existing problems
 - Drop-down list `Load Pbm`: user can select a problem from the list of existing problems; default None, which means don't load any problem;
  Check `problems/` folder for existing problems; the directory names are the existing problem names;
  the problem name selected is stored in variable `problem_name`;
  if user selects a problem, then `Pbm Name` text box will be updated with the problem name;
  other text boxes like `Pbm Description`, `Ext knowledge`, `Function Signature`, `Evaluation Script`,
  `Dataset Generation Script`, `Seed Sol Idea`, `Seed Solution Script` will also be updated by loading corresponding files from `problems/<problem_name>/` folder;
  See below subsection explanations for more details about which files should be loaded to update those text boxes.
  `Data Files Visual` display area will also be updated.

This subsection has three text boxes:
 - Text box `Pbm Name`: user input a problem name for creating a new problem, stored in variable `problem_name`; 
   this text box is disabled for user input if user selects a problem from the drop-down list `Load Pbm`;
 - Text box `Pbm Description`: user input problem description text, stored in variable `problem_description` (required)
 - Text box `Ext knowledge`: user input external knowledge text about how to solve the problem, stored in variable `external_knowledge` (optional); None if not exists

This subsection has three buttons:
 - Button `Submit Btn`: user click this button then create folder `problems/<problem_name>` and `problems/<problem_name>/dataset/` if not exists;
   And create a file `problems/<problem_name>/problem_description.txt` and write `problem_description` into it;
   if `external_knowledge` is not None, create a file `problems/<problem_name>/external_knowledge.txt` and write `external_knowledge` into it;
 - Button `LLM-gen Btn`: user click this button, 
    Invoke `orcanvas.problem_description.generate_problem_description` function to generate a polished problem description text,
    and update `problems/<problem_name>/problem_description.txt`;
    `Pbm Description` text box will also be updated with the updated problem description;
 
 
**Problem Definition Section - Subsection 2) Evaluation Description**
There are two text boxes:
 - Text box `Function Signature`: the function signature of the function to be optimized, user can input it here, or leave it empty (optional);
 - Text box `Evaluation Script`: user input evaluation script (optional);
 
And this subsection has two buttons:
 - Button `Submit Btn`: user click this button, store user input from `Function Signature` text box in variable `function_signature`; None if no user input; 
    if `function_signature` not None, write `function_signature` to file `problems/<problem_name>/function_description.txt`;
    Store text from "Evaluation Script" text box in variable `evaluation_script`; None if no user input; 
    if `evaluation_script` not None, write `evaluation_script` to file `problems/<problem_name>/eval.py`;
    
 - Button `LLM-gen Btn`: user click this button, invoke `orcanvas.evaluation_description.generate_evaluation_description` function to generate a polished evaluation script and function signature, as well as function to evolve and objective type
 and write them into `problems/<problem_name>/eval.py` and `problems/<problem_name>/function_description.txt` respectively;
 Also create `problems/<problem_name>/settings.yaml` file with content like:
 ```yaml
"function_to_evolve": "heuristics"
"obj_type": "min"
 ```
 
 
**Problem Definition Section - Subsection 3) Test Dataset Generation**
This subsection has an display area:
 - Area `Data files Visual`: displaying files in the folder `problems/<problem_name>/dataset/` if `problem_name` is not None;
  Use st.code;

This subsection also has a text box:
 - Text box `Dataset Generation Script`: user input test dataset generation script (optional);
 
And three buttons:
 - Button `Submit Btn`: user click this button, store user input from "Dataset Generation Script" text box in variable `dataset_generation_script`; None if no user input; 
    if `dataset_generation_script` not None, write `dataset_generation_script` to file `problems/<problem_name>/generate_dataset.py`;
 - Button `LLM-gen Btn`: user click this button, invoke `orcanvas.test_dataset_generation.generate_test_dataset` function to generate a polished test dataset generation script 
 and update `problems/<problem_name>/generate_dataset.py`;
 Text box `Dataset Generation Script` will also be updated with the updated test dataset generation script;
 - Button `Run Btn`: user click this button, run `problems/<problem_name>/generate_dataset.py` in a separate process to generate test dataset;
 Area `Data files Visual` will be updated to show the newly generated files;
 
 
**Problem Definition Section - Subsection 4) Seed Solution Description**
This subsection has two text boxes:
 - `Seed Solution Idea`: user input a seed solution idea (optional);
 - `Seed Solution Script`: user input seed solution script (optional);

This subsection has three buttons:
 - Button `Submit Btn`: user click this button, store user input from `Seed Solution Idea` text box in variable `seed_solution_idea`; None if no user input; 
    if `seed_solution_idea` not None, write `seed_solution_idea` to file `problems/<problem_name>/seed_solution_idea.txt`;
    Store text from `Seed Solution Script` text box in variable `seed_solution_script`; None if no user input;
    If `seed_solution_script` not None, write `seed_solution_script` to file `problems/<problem_name>/seed_solution.py`;
 - Button `LLM-gen Btn`: user click this button, invoke `orcanvas.seed_solution_generation.generate_seed_solution` function to generate a polished seed solution idea and script;
 and update `problems/<problem_name>/seed_solution_idea.py` and `problems/<problem_name>/seed_solution.py` respectively;
 Also update text box `Seed Solution Idea` and `Seed Solution Script` with the updated seed solution idea and script;
 - Button `Run Btn`: user click this button, run `problems/<problem_name>/eval.py` in a separate process to evaluate the seed solution;







### Agent Control section
Control section have the following items:
 - "Generate config" button: user click this, invoke `python src/oragent/cli.py --init-config` to generate a config.yaml file at current working directory.
 - "Select checkpoint" drop-down list: the drop-down list will include all checkpoints in `<project_root>/checkpoints` directory.
 - "Select problem" drop-down list: the drop-down list will include all problems in `<project_root>/problems` directory.
 - "Select algorithm" drop-down list: the drop-down list will include options: "ORAgent", "ReEvo", "EoH", "AEL", "FunSearch".
 - "Start" button: user click this, invoke `python src/oragent/cli.py --algorithm=<algorithm>` or `--problem=<problem>` to start the backend process.
 - "Stop" button: user click this, stop the backend process. 




### Agent Output Section
Webui will check the files in the output directory for update every few seconds.
WebUI need to display the following types of messages:
 - real-time system messages
 - solution performance info
 - research progress info
 - solution database info (optional)


**Real-time system messages display**
Messages are strings send to backend stdout.
">>>[OR-Agent] Make sure that your problem has already been defined in \"[CWD]/problems\" directory"
"\n>>>[FunSearch] FunSearch initialized"
"\n>>>[Evaluator]Evaluation successful: iter0_response0\nObj: 2.3"
""\n>>>[ExperimentAgent] action after experiment 5: terminate\"

As we can see, the entity that sends messages is signed by "[<name>]".
WebUI will start the backend process and capture the stdout for displaying.
WebUI will continuously read the messages from stdout and display them in the real-time system messages section; 
different colors for different entities;
when new messages come, old messages will scroll automatically.


**Solution performance info display**
WebUI will also display the progress of the research process in the research progress section.
Solution performance info is stored in `<project_root>/outputs/<algorithm>/<problem>/results.json`.
WebUI will display the solution performance curve; x-axis is "total_responses", y-axis is "best_obj_overall".

results.json is a list of dict with the following keys:
- "iteration": iteration number when the result is obtained
- "total_responses": total number of responses when the result is obtained
- "total_function_evals": total number of function evaluations when the result is obtained
- "total_valid_responses": total number of valid responses when the result is obtained
- "best_obj_overall": best objective value of the best solution
- "metrics": metrics of the best solution
- "code_filepath": code path of the best solution
- "output_filepath": stdout file path of the best solution

results.json is updated every time a new best solution is obtained.
If the file does not exist, display a message saying "No solution performance info available".


**Research progress info display**
Research progress info is a text file stored in `<project_root>/outputs/<algorithm>/<problem>/progress.txt`.
Example:
```
========================================
✓ Node 0 (0.80)
    ├──   Node 1 (0.90)
    │       └──   Node 3 (0.85)
    └──   Node 2 (0.70)
            └──   Node 4 (0.75)
========================================
✓ = done, empty = not done; (score)
```
WebUI will display the research progress info in the research progress section.
If the file does not exist, display a message saying "No research progress info available".


**Solution database info display (optional)**
Solution database info is a text file stored in `<project_root>/outputs/<algorithm>/<problem>/database.txt`.
Database has islands;
each island has many clusters;
each cluster has many solutions.

Example:
```
========================================
SOLUTION DATABASE VISUALIZATION
========================================
Total islands: 2
Objective type: max

ISLAND 0:
  Number of clusters: 2
  Total solutions: 3
  Best score in island: 0.9

  CLUSTER 0:
    Features: (1, 2, 3)
    Cluster score: 0.800000
    Number of solutions: 2
      Solution 0:
        Score: 0.700000
        Code length: 21
        ID: 1

      Solution 1:
        ...

  CLUSTER 1:
    ...

----------------------------------------
ISLAND 1:
    ...

----------------------------------------

SUMMARY STATISTICS:
Total solutions across all islands: 4
Total clusters across all islands: 3
Average cluster size: 1.33
Best overall score: 0.900000 (Island 0)
========================================
```

WebUI should load the text description; parse it in the right way; and display it in the solution database info section properly.
If the file does not exist, display a message saying "No solution database info available".
"""
import streamlit as st
import subprocess
import threading
import time
import json
import yaml
import os
from pathlib import Path
import queue
import re
import plotly.graph_objects as go
from datetime import datetime
import sys
import orcanvas



#=====Initialize session state=====
if 'process' not in st.session_state:
    st.session_state.process = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = {}
if 'message_queue' not in st.session_state:
    st.session_state.message_queue = queue.Queue()




#=====Helper functions=====
#-----Helper functions for OR-Canvas-----
# Helper function to load existing problems
def load_existing_problems():
    problems_dir = Path.cwd() / "problems"
    problems = []
    if problems_dir.exists():
        problems = [p.name for p in problems_dir.iterdir() if p.is_dir()]
    return problems

# Helper function to load problem data
def load_problem_data(problem_name):
    """Load all data for a given problem from files"""
    problem_dir = Path.cwd() / "problems" / problem_name
    data = {}

    print(f"\n=== Loading problem data for: {problem_name} ===")
    print(f"Problem directory: {problem_dir}")

    # Load problem description
    desc_file = problem_dir / "problem_description.txt"
    if desc_file.exists():
        print(f"✓ Found problem_description.txt")
        with open(desc_file, 'r') as f:
            data['problem_description'] = f.read()
    else:
        print(f"✗ Missing problem_description.txt")

    # Load external knowledge
    ext_file = problem_dir / "external_knowledge.txt"
    if ext_file.exists():
        print(f"✓ Found external_knowledge.txt")
        with open(ext_file, 'r') as f:
            data['external_knowledge'] = f.read()
    else:
        print(f"✗ Missing external_knowledge.txt")

    # Load function signature
    func_file = problem_dir / "function_description.txt"
    if func_file.exists():
        print(f"✓ Found function_description.txt")
        with open(func_file, 'r') as f:
            data['function_signature'] = f.read()
    else:
        print(f"✗ Missing function_description.txt")

    # Load evaluation script
    eval_file = problem_dir / "eval.py"
    if eval_file.exists():
        print(f"✓ Found eval.py")
        with open(eval_file, 'r') as f:
            data['evaluation_script'] = f.read()
    else:
        print(f"✗ Missing eval.py")

    # Load dataset generation script
    dataset_file = problem_dir / "generate_dataset.py"
    if dataset_file.exists():
        print(f"✓ Found generate_dataset.py")
        with open(dataset_file, 'r') as f:
            data['dataset_generation_script'] = f.read()
    else:
        print(f"✗ Missing generate_dataset.py")

    # Load seed solution idea
    seed_idea_file = problem_dir / "seed_solution_idea.txt"
    if seed_idea_file.exists():
        print(f"✓ Found seed_solution_idea.txt")
        with open(seed_idea_file, 'r') as f:
            data['seed_solution_idea'] = f.read()
    else:
        print(f"✗ Missing seed_solution_idea.txt")

    # Load seed solution script
    seed_script_file = problem_dir / "seed_solution.py"
    if seed_script_file.exists():
        print(f"✓ Found seed_solution.py")
        with open(seed_script_file, 'r') as f:
            data['seed_solution_script'] = f.read()
    else:
        print(f"✗ Missing seed_solution.py")

    print(f"=== Finished loading problem data ===\n")
    return data

# Helper function to save problem data
def save_problem_data(problem_name, data):
    """Save problem data to files"""
    problem_dir = Path.cwd() / "problems" / problem_name
    dataset_dir = problem_dir / "dataset"

    # Create directories if they don't exist
    problem_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save problem description
    if 'problem_description' in data and data['problem_description']:
        with open(problem_dir / "problem_description.txt", 'w') as f:
            f.write(data['problem_description'])

    # Save external knowledge
    if 'external_knowledge' in data and data['external_knowledge']:
        with open(problem_dir / "external_knowledge.txt", 'w') as f:
            f.write(data['external_knowledge'])

    # Save function signature
    if 'function_signature' in data and data['function_signature']:
        with open(problem_dir / "function_description.txt", 'w') as f:
            f.write(data['function_signature'])

    # Save evaluation script
    if 'evaluation_script' in data and data['evaluation_script']:
        with open(problem_dir / "eval.py", 'w') as f:
            f.write(data['evaluation_script'])

    # Save dataset generation script
    if 'dataset_generation_script' in data and data['dataset_generation_script']:
        with open(problem_dir / "generate_dataset.py", 'w') as f:
            f.write(data['dataset_generation_script'])

    # Save seed solution idea
    if 'seed_solution_idea' in data and data['seed_solution_idea']:
        with open(problem_dir / "seed_solution_idea.txt", 'w') as f:
            f.write(data['seed_solution_idea'])

    # Save seed solution script
    if 'seed_solution_script' in data and data['seed_solution_script']:
        with open(problem_dir / "seed_solution.py", 'w') as f:
            f.write(data['seed_solution_script'])

# Helper function to list dataset files
def list_dataset_files(problem_name):
    """List files in the dataset directory"""
    dataset_dir = Path.cwd() / "problems" / problem_name / "dataset"
    if dataset_dir.exists():
        files = [f.name for f in dataset_dir.iterdir() if f.is_file()]
        return files
    return []

# Helper function to run dataset generation script
def run_dataset_generation(problem_name):
    """Run the dataset generation script"""
    script_path = Path.cwd() / "problems" / problem_name / "generate_dataset.py"
    if script_path.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=script_path.parent
            )
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except Exception as e:
            return False, str(e)
    return False, "Dataset generation script not found"

# Helper function to run evaluation script
def run_evaluation(problem_name):
    """Run the evaluation script"""
    script_path = Path.cwd() / "problems" / problem_name / "eval.py"
    if script_path.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=script_path.parent
            )
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except Exception as e:
            return False, str(e)
    return False, "Evaluation script not found"


#-----Helper functions for OR-Agent-----
def load_experiment_config(problem=None):
    """Load experiment config from problems/<problem>/settings.yaml"""
    if not problem:
        # Try to get problem from session state
        if 'selected_problem' in st.session_state:
            problem = st.session_state.selected_problem
        else:
            return {}

    # Load from problems/<problem>/settings.yaml
    config_file = Path.cwd() / "problems" / problem / "settings.yaml"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                return config
        except Exception as e:
            st.session_state.messages.append(("System", f"Failed to load settings.yaml for {problem}: {str(e)}"))
            return {}
    return {}

def load_complete_config():
    """Load experiment config from problems/<problem>/settings.yaml"""
    if 'selected_problem' in st.session_state:
        problem = st.session_state.selected_problem
    else:
            return {}
        
    if 'selected_algorithm' in st.session_state:
        algorithm = st.session_state.selected_algorithm
    else:
            return {}
        
    # Load from problems/<problem>/settings.yaml
    config_file = Path.cwd() / "outputs"/ algorithm / problem / "config.yaml"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                return config
        except Exception as e:
            st.session_state.messages.append(("System", f"Failed to load config.yaml for algorithm {algorithm} problem {problem}: {str(e)}"))
            return {}
    return {}

# Color mapping for different entities
ENTITY_COLORS = {
    'or-agent': "#000000",
    'oragent': "#073B4C",
    'reevo': '#073B4C',
    'eoh': '#073B4C',
    'ael': "#073B4C",
    'funsearch': '#073B4C',
    'evaluator': "#AE1E1E",
    'experimentagent': "#C4A610",
    'ideaagent': "#B90BC5",
    'codeagent': "#179B1D",
    'leadagent': "#000000",
    'database': "#6C757D",
    'default': "#000000"
}  # OR-Agent, ReEvo, EoH, AEL, FunSearch share the same color since they don't run in the same process

def parse_message_line(line):
    """Parse a message line to extract entity and content"""
    # Look for pattern like "[Entity] message" with optional ">>>" prefix
    match = re.match(r'(?:>>>)?\[([^\]]+)\](.*)', line.strip())
    if match:
        entity = match.group(1).strip()
        content = match.group(2).strip()
        return entity, content
    return None, line.strip()

def read_stdout(process, message_queue):
    """Read stdout from process and put messages in queue"""
    try:
        # Use iter with readline for more reliable reading
        for line in iter(process.stdout.readline, b''):
            if line:
                decoded = line.decode('utf-8', errors='ignore')
                message_queue.put(decoded)
                # Debug: also add raw output to help diagnose
                # message_queue.put(f"[DEBUG stdout] {repr(decoded)}")

        # Process has ended, read any remaining output
        remaining = process.stdout.read()
        if remaining:
            decoded_remaining = remaining.decode('utf-8', errors='ignore')
            message_queue.put(decoded_remaining)
            # message_queue.put(f"[DEBUG stdout remaining] {repr(decoded_remaining)}")

    except Exception as e:
        # If there's an error reading, add it to the queue
        message_queue.put(f"[System] Error reading process output: {str(e)}")
        import traceback
        message_queue.put(f"[System] Traceback: {traceback.format_exc()}")

def read_stderr(process, message_queue):
    """Read stderr from process and put messages in queue"""
    try:
        # Use iter with readline for more reliable reading
        for line in iter(process.stderr.readline, b''):
            if line:
                decoded = line.decode('utf-8', errors='ignore')
                message_queue.put(f"[ERROR] {decoded}")

        # Process has ended, read any remaining error output
        remaining = process.stderr.read()
        if remaining:
            message_queue.put(f"[ERROR] {remaining.decode('utf-8', errors='ignore')}")

    except Exception as e:
        # If there's an error reading, add it to the queue
        message_queue.put(f"[System] Error reading process stderr: {str(e)}")

def start_backend(checkpoint, algorithm, problem):
    """Start the backend process with algorithm and problem arguments"""
    if st.session_state.process is not None:
        st.session_state.process.terminate()
        st.session_state.process = None

    # Try to start the backend process using the most robust method
    # Method 1: Try using the console script 'oragent' (if package is installed with entry point)
    # Method 2: Try using python -m oragent.cli (if package is installed)
    # Method 3: Try using python -m src.oragent.cli (if running from source)

    cmd = None

    # Build command arguments, only add if not None and not "None"
    args = []
    if checkpoint and checkpoint.lower().strip() != "none":
        args.extend(['--checkpoint', checkpoint])
    if algorithm and algorithm.lower().strip() != "none":
        args.extend(['--algorithm', algorithm])
    if problem and problem.lower().strip() != "none":
        args.extend(['--problem', problem])

    args.extend(['--web'])
    
    # Try python -m oragent.cli
    try:
        import importlib
        spec = importlib.util.find_spec("oragent")
        if spec is not None:
            # Package is installed, use python -m oragent.cli
            cmd = [sys.executable, "-m", "oragent.cli"] + args
    except:
        pass

    if cmd is None:
        # Try to use src.oragent.cli (running from source)
        cmd = [sys.executable, "-u", "-m", "src.oragent.cli"] + args

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=False,
        env={**os.environ, 'PYTHONUNBUFFERED': '1'}
    )

    # Start thread to read stdout
    st.session_state.messages.append(("System", f"Backend process started using command: {str(cmd)}"))
    # The code creates a thread that runs the read_stdout function, which reads the stdout from the process and puts the output into st.session_state.message_queue
    thread = threading.Thread(target=read_stdout, args=(process, st.session_state.message_queue))
    thread.daemon = True
    thread.start()

    # Start thread to read stderr
    stderr_thread = threading.Thread(target=read_stderr, args=(process, st.session_state.message_queue))
    stderr_thread.daemon = True
    stderr_thread.start()

    st.session_state.process = process

    return process

def stop_backend():
    """Stop the backend process"""
    if st.session_state.process is not None:
        st.session_state.process.terminate()
        st.session_state.process = None
        st.session_state.messages.append(("System", "Backend process stopped"))


def get_output_path(algorithm=None, problem=None):
    """Get output path for current algorithm and problem"""
    # Use provided algorithm and problem, or get from session state
    if not algorithm or not problem:
        # Try to get from session state or UI selection
        if 'selected_algorithm' in st.session_state and 'selected_problem' in st.session_state:
            algorithm = st.session_state.selected_algorithm
            problem = st.session_state.selected_problem
        else:
            return None

    # Convert algorithm to lowercase for directory path
    algorithm_lower = algorithm.lower()
    output_dir = Path.cwd() / "outputs" / algorithm_lower / problem
    return output_dir

def load_results_json():
    """Load results.json file"""
    output_dir = get_output_path()
    if not output_dir:
        return None

    results_file = output_dir / "results.json"
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def load_progress_txt():
    """Load progress.txt file"""
    output_dir = get_output_path()
    if not output_dir:
        return None

    progress_file = output_dir / "progress.txt"
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return f.read()
        except:
            return None
    return None

def load_long_term_reflection_txt():
    """Load long_term_reflection.txt file"""
    output_dir = get_output_path()
    if not output_dir:
        return None

    long_term_reflection_file = output_dir / "long_term_reflection.txt"
    if long_term_reflection_file.exists():
        try:
            with open(long_term_reflection_file, 'r') as f:
                return f.read()
        except:
            return None
    return None

def load_output_txt():
    """Load output.txt file; which is the saved stdout history of backend running."""
    output_dir = get_output_path()
    if not output_dir:
        return None

    output_file = output_dir / "output.txt"
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                return f.read()
        except:
            return None
    return None

def load_solution_database_txt():
    """Load database.txt file"""
    output_dir = get_output_path()
    if not output_dir:
        return None

    db_file = output_dir / "database.txt"
    if db_file.exists():
        try:
            with open(db_file, 'r') as f:
                return f.read()
        except:
            return None
    return None

def parse_solution_database(content):
    """Parse solution database content into structured format"""
    if not content:
        return None

    sections = content.split('----------------------------------------')
    result = {
        'islands': [],
        'summary': {}
    }

    current_island = None
    current_cluster = None

    for line in content.split('\n'):
        line = line.strip()

        # Parse summary statistics
        if line.startswith('Total solutions across all islands:'):
            result['summary']['total_solutions'] = line.split(':')[1].strip()
        elif line.startswith('Total clusters across all islands:'):
            result['summary']['total_clusters'] = line.split(':')[1].strip()
        elif line.startswith('Average cluster size:'):
            result['summary']['avg_cluster_size'] = line.split(':')[1].strip()
        elif line.startswith('Best overall score:'):
            result['summary']['best_score'] = line.split(':')[1].strip()

        # Parse island
        elif line.startswith('ISLAND'):
            island_num = int(line.split()[1].replace(':', ''))
            current_island = {
                'number': island_num,
                'clusters': []
            }
            result['islands'].append(current_island)
            current_cluster = None

        # Parse cluster
        elif line.startswith('CLUSTER'):
            cluster_num = int(line.split()[1].replace(':', ''))
            current_cluster = {
                'number': cluster_num,
                'solutions': []
            }
            if current_island:
                current_island['clusters'].append(current_cluster)

        # Parse solution
        elif line.startswith('Solution'):
            if current_cluster:
                sol_num = int(line.split()[1].replace(':', ''))
                current_cluster['solutions'].append({'number': sol_num})

    return result







#=====WebUI=====
# Streamlit UI
st.set_page_config(layout="wide", page_title="OR-Agent WebUI")



#-----Header Section-----
# Header section
st.markdown("""
<div style="background-color: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
    <h1 style="margin: 0; color: #003399;">Open Research Canvas</h1>
</div>
""", unsafe_allow_html=True)
#st.markdown("---")



#-----Problem Definition Section-----
# Initialize session state variables for problem definition
if 'problem_name' not in st.session_state:
    st.session_state.problem_name = ""
if 'problem_description' not in st.session_state:
    st.session_state.problem_description = ""
if 'external_knowledge' not in st.session_state:
    st.session_state.external_knowledge = ""
if 'function_signature' not in st.session_state:
    st.session_state.function_signature = ""
if 'evaluation_script' not in st.session_state:
    st.session_state.evaluation_script = ""
if 'dataset_generation_script' not in st.session_state:
    st.session_state.dataset_generation_script = ""
if 'seed_solution_idea' not in st.session_state:
    st.session_state.seed_solution_idea = ""
if 'seed_solution_script' not in st.session_state:
    st.session_state.seed_solution_script = ""
if 'last_loaded_problem' not in st.session_state:
    st.session_state.last_loaded_problem = ""
if 'previous_dropdown_selection' not in st.session_state:
    st.session_state.previous_dropdown_selection = "None"



#---Subsection 1: Problem Description---
st.markdown("#### Problem Description")
col1, col2, col3, col4, col5= st.columns([9, 1, 1, 1, 1])

with col1:
    # Placeholder - empty column for alignment
    st.empty()
    
with col2:
    # Load existing problems dropdown
    existing_problems = load_existing_problems()
    existing_problems.insert(0, "None")
    selected_existing = st.selectbox(
        "Load Problem",
        existing_problems,
        index=0,
        key="load_problem_select"
    )

    # If a problem is selected, load its data
    print(f"\n=== Problem selection debug ===")
    print(f"Selected existing: '{selected_existing}'")
    print(f"Previous dropdown selection: '{st.session_state.previous_dropdown_selection}'")
    print(f"Last loaded problem: '{st.session_state.last_loaded_problem}'")

    # Only load if dropdown selection changed (not on every rerun)
    if selected_existing != st.session_state.previous_dropdown_selection:
        print(f"Dropdown selection changed from '{st.session_state.previous_dropdown_selection}' to '{selected_existing}'")

        if selected_existing != "None":
            print(f"Loading problem data for: {selected_existing}")
            data = load_problem_data(selected_existing)
            st.session_state.problem_name = selected_existing
            st.session_state.problem_description = data.get('problem_description', '')
            st.session_state.external_knowledge = data.get('external_knowledge', '')
            st.session_state.function_signature = data.get('function_signature', '')
            st.session_state.evaluation_script = data.get('evaluation_script', '')
            st.session_state.dataset_generation_script = data.get('dataset_generation_script', '')
            st.session_state.seed_solution_idea = data.get('seed_solution_idea', '')
            st.session_state.seed_solution_script = data.get('seed_solution_script', '')
            st.session_state.last_loaded_problem = selected_existing

            # Also update the widget states directly since they have key parameters
            st.session_state["problem_name_input"] = selected_existing
            st.session_state["problem_description_input"] = data.get('problem_description', '')
            st.session_state["external_knowledge_input"] = data.get('external_knowledge', '')
            st.session_state["function_signature_input"] = data.get('function_signature', '')
            st.session_state["evaluation_script_input"] = data.get('evaluation_script', '')
            st.session_state["dataset_generation_script_input"] = data.get('dataset_generation_script', '')
            st.session_state["seed_solution_idea_input"] = data.get('seed_solution_idea', '')
            st.session_state["seed_solution_script_input"] = data.get('seed_solution_script', '')

            print(f"Updated session state and widget states. Calling st.rerun()")
        elif selected_existing == "None" and st.session_state.last_loaded_problem != "":
            # If selecting "None" after having loaded a problem, clear the session state
            print(f"Clearing session state - selected 'None'")
            st.session_state.problem_name = ""
            st.session_state.problem_description = ""
            st.session_state.external_knowledge = ""
            st.session_state.function_signature = ""
            st.session_state.evaluation_script = ""
            st.session_state.dataset_generation_script = ""
            st.session_state.seed_solution_idea = ""
            st.session_state.seed_solution_script = ""
            st.session_state.last_loaded_problem = ""

            # Also clear the widget states directly since they have key parameters
            st.session_state["problem_name_input"] = ""
            st.session_state["problem_description_input"] = ""
            st.session_state["external_knowledge_input"] = ""
            st.session_state["function_signature_input"] = ""
            st.session_state["evaluation_script_input"] = ""
            st.session_state["dataset_generation_script_input"] = ""
            st.session_state["seed_solution_idea_input"] = ""
            st.session_state["seed_solution_script_input"] = ""

            print(f"Cleared session state and widget states. Calling st.rerun()")
        else:
            print(f"No action needed - selected 'None' and no previously loaded problem")

        # Update previous dropdown selection
        st.session_state.previous_dropdown_selection = selected_existing
        st.rerun()
    else:
        print(f"Dropdown selection unchanged, skipping load")

with col3:
    # Problem name input (disabled if loading existing problem)
    problem_name_disabled = selected_existing != "None"
    print(f"Problem name text box - value: '{st.session_state.problem_name}', disabled: {problem_name_disabled}")
    st.text_input(
        "Problem Name",
        value=st.session_state.problem_name,
        disabled=problem_name_disabled,
        key="problem_name_input"
    )

with col4:
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 100%; margin-top: 28px;">
    """, unsafe_allow_html=True)
    submit_problem = st.button("Submit", type="secondary", use_container_width=True, key="submit_problem")
    st.markdown("</div>", unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 100%; margin-top: 28px;">
    """, unsafe_allow_html=True)
    llm_gen_problem = st.button("LLM Generate", type="secondary", use_container_width=True, key="llm_gen_problem")
    st.markdown("</div>", unsafe_allow_html=True)


# Problem description and external knowledge in columns
col1, col2 = st.columns([9, 4])

with col1:
    st.text_area(
        "Problem Description",
        value=st.session_state.problem_description,
        height=150,
        key="problem_description_input"
    )

with col2:
    st.text_area(
        "External Knowledge",
        value=st.session_state.external_knowledge,
        height=150,
        key="external_knowledge_input"
    )

# Handle submit button for problem description
if submit_problem:
    if not st.session_state.problem_name:
        st.error("Please enter a problem name")
    elif not st.session_state.problem_description:
        st.error("Please enter a problem description")
    else:
        data = {
            'problem_description': st.session_state.problem_description,
            'external_knowledge': st.session_state.external_knowledge if st.session_state.external_knowledge else None
        }
        save_problem_data(st.session_state.problem_name, data)
        st.success(f"Problem '{st.session_state.problem_name}' saved successfully!")
        st.session_state.messages.append(("System", f"Problem '{st.session_state.problem_name}' saved"))

# Handle LLM generate button for problem description
if llm_gen_problem:
    if not st.session_state.problem_name:
        st.error("Please enter a problem name first")
    elif not st.session_state.problem_description:
        st.error("Please enter a problem description first")
    else:
        try:
            # Call the LLM to generate polished problem description
            updated_description = orcanvas.generate_problem_description(st.session_state.problem_description)
            st.session_state.problem_description = updated_description

            # Save the updated description
            data = {
                'problem_description': updated_description,
                'external_knowledge': st.session_state.external_knowledge if st.session_state.external_knowledge else None
            }
            save_problem_data(st.session_state.problem_name, data)

            st.success("Problem description polished by LLM and saved!")
            st.session_state.messages.append(("System", f"Problem description for '{st.session_state.problem_name}' polished by LLM"))
            st.rerun()
        except Exception as e:
            st.error(f"Error generating problem description: {str(e)}")
            st.session_state.messages.append(("System", f"Error generating problem description: {str(e)}"))

st.markdown("---")



#---Subsection 2: Evaluation Description---
st.markdown("#### Evaluation Description")
col1, col2, col3 = st.columns([11, 1, 1])

with col1:
    # Placeholder - empty column for alignment
    st.empty()

with col2:
    submit_eval = st.button("Submit", type="secondary", use_container_width=True, key="submit_eval")

with col3:
    llm_gen_eval = st.button("LLM Generate", type="secondary", use_container_width=True, key="llm_gen_eval")


# Function signature and evaluation script in columns
col1, col2 = st.columns([3, 7])

with col1:
    st.text_area(
        "Function Signature",
        value=st.session_state.function_signature,
        height=300,
        key="function_signature_input"
    )

with col2:
    st.text_area(
        "Evaluation Script",
        value=st.session_state.evaluation_script,
        height=300,
        key="evaluation_script_input"
    )

# Handle submit button for evaluation description
if submit_eval:
    if not st.session_state.problem_name:
        st.error("Please define a problem first (Section 1)")
    else:
        data = {}
        if st.session_state.function_signature:
            data['function_signature'] = st.session_state.function_signature
        if st.session_state.evaluation_script:
            data['evaluation_script'] = st.session_state.evaluation_script

        save_problem_data(st.session_state.problem_name, data)
        st.success("Evaluation description saved successfully!")
        st.session_state.messages.append(("System", f"Evaluation description for '{st.session_state.problem_name}' saved"))

# Handle LLM generate button for evaluation description
if llm_gen_eval:
    if not st.session_state.problem_name:
        st.error("Please define a problem first (Section 1)")
    elif not st.session_state.problem_description:
        st.error("Please enter a problem description first (Section 1)")
    else:
        try:
            # Call the LLM to generate evaluation description
            result = orcanvas.generate_evaluation_description(
                problem_description=st.session_state.problem_description,
                function_signature=st.session_state.function_signature if st.session_state.function_signature else None,
                evaluation_script=st.session_state.evaluation_script if st.session_state.evaluation_script else None
            )

            function_signature_update, function_to_evolve, obj_type, evaluation_script_update = result

            # Update session state
            st.session_state.function_signature = function_signature_update
            st.session_state.evaluation_script = evaluation_script_update

            # Save the updated data
            data = {
                'function_signature': function_signature_update,
                'evaluation_script': evaluation_script_update
            }
            save_problem_data(st.session_state.problem_name, data)

            # Create settings.yaml
            settings_path = Path.cwd() / "problems" / st.session_state.problem_name / "settings.yaml"
            settings = {
                "function_to_evolve": function_to_evolve,
                "obj_type": obj_type
            }
            with open(settings_path, 'w') as f:
                yaml.dump(settings, f)

            st.success("Evaluation description generated by LLM and saved!")
            st.session_state.messages.append(("System", f"Evaluation description for '{st.session_state.problem_name}' generated by LLM"))
            st.rerun()
        except Exception as e:
            st.error(f"Error generating evaluation description: {str(e)}")
            st.session_state.messages.append(("System", f"Error generating evaluation description: {str(e)}"))

st.markdown("---")




#---Subsection 3: Test Dataset Generation---
st.markdown("#### Test Dataset")
col1, col2, col3, col4 = st.columns([10, 1, 1, 1])

with col1:
    # Placeholder - empty column for alignment
    st.empty()
    
with col2:
    submit_dataset = st.button("Submit", type="secondary", use_container_width=True, key="submit_dataset")

with col3:
    llm_gen_dataset = st.button("LLM Generate", type="secondary", use_container_width=True, key="llm_gen_dataset")

with col4:
    run_dataset = st.button("Run", type="secondary", use_container_width=True, key="run_dataset")
    
    
# Display dataset files and dataset generation script in columns
col1, col2 = st.columns([3, 7])
with col1:
    # Create a container with fixed height matching col2
        container = st.container(height=300)
        with container:
            st.markdown("**Dataset Files:**")
            # Display dataset files in fixed size container
            if st.session_state.problem_name:
                dataset_files = list_dataset_files(st.session_state.problem_name)
                if dataset_files:
                    for file in dataset_files:
                        st.code(file, language="text")
                else:
                    st.info("No file found.")
            else:
                st.info("Problem not specified.")
        

with col2:
    # Dataset generation script
    st.text_area(
        "Dataset Generation Script",
        value=st.session_state.dataset_generation_script,
        height=300,
        key="dataset_generation_script_input"
    )

# Handle submit button for dataset generation
if submit_dataset:
    if not st.session_state.problem_name:
        st.error("Please define a problem first (Section 1)")
    else:
        data = {}
        if st.session_state.dataset_generation_script:
            data['dataset_generation_script'] = st.session_state.dataset_generation_script

        save_problem_data(st.session_state.problem_name, data)
        st.success("Dataset generation script saved successfully!")
        st.session_state.messages.append(("System", f"Dataset generation script for '{st.session_state.problem_name}' saved"))

# Handle LLM generate button for dataset generation
if llm_gen_dataset:
    if not st.session_state.problem_name:
        st.error("Please define a problem first (Section 1)")
    elif not st.session_state.problem_description:
        st.error("Please enter a problem description first (Section 1)")
    elif not st.session_state.function_signature:
        st.error("Please define function signature first (Section 2)")
    elif not st.session_state.evaluation_script:
        st.error("Please define evaluation script first (Section 2)")
    else:
        try:
            # Call the LLM to generate dataset generation script
            updated_script = orcanvas.generate_test_dataset(
                problem_description=st.session_state.problem_description,
                function_signature=st.session_state.function_signature,
                evaluation_script=st.session_state.evaluation_script,
                dataset_generation_script=st.session_state.dataset_generation_script if st.session_state.dataset_generation_script else None
            )

            # Update session state
            st.session_state.dataset_generation_script = updated_script

            # Save the updated script
            data = {
                'dataset_generation_script': updated_script
            }
            save_problem_data(st.session_state.problem_name, data)

            st.success("Dataset generation script generated by LLM and saved!")
            st.session_state.messages.append(("System", f"Dataset generation script for '{st.session_state.problem_name}' generated by LLM"))
            st.rerun()
        except Exception as e:
            st.error(f"Error generating dataset generation script: {str(e)}")
            st.session_state.messages.append(("System", f"Error generating dataset generation script: {str(e)}"))

# Handle run button for dataset generation
if run_dataset:
    if not st.session_state.problem_name:
        st.error("Please define a problem first (Section 1)")
    else:
        success, output = run_dataset_generation(st.session_state.problem_name)
        if success:
            st.success("Dataset generation completed successfully!")
            st.session_state.messages.append(("System", f"Dataset generation for '{st.session_state.problem_name}' completed"))

            # Show output if any
            if output:
                with st.expander("Dataset Generation Output"):
                    st.code(output, language="text")
        else:
            st.error(f"Dataset generation failed: {output}")
            st.session_state.messages.append(("System", f"Dataset generation for '{st.session_state.problem_name}' failed: {output}"))

st.markdown("---")




#---Subsection 4: Seed Solution Generation---
st.markdown("#### Seed Solution")
col1, col2, col3, col4 = st.columns([10, 1, 1, 1])

with col1:
    # Placeholder - empty column for alignment
    st.empty()

with col2:
    submit_seed = st.button("Submit", type="secondary", use_container_width=True, key="submit_seed")

with col3:
    llm_gen_seed = st.button("LLM Generate", type="secondary", use_container_width=True, key="llm_gen_seed")

with col4:
    run_seed = st.button("Run", type="secondary", use_container_width=True, key="run_seed")

# Seed solution idea and script in columns
col1, col2 = st.columns([3, 7])

with col1:
    st.text_area(
        "Seed Solution Idea",
        value=st.session_state.seed_solution_idea,
        height=300,
        key="seed_solution_idea_input"
    )

with col2:
    st.text_area(
        "Seed Solution Script",
        value=st.session_state.seed_solution_script,
        height=300,
        key="seed_solution_script_input"
    )

# Handle submit button for seed solution
if submit_seed:
    if not st.session_state.problem_name:
        st.error("Please define a problem first (Section 1)")
    else:
        data = {}
        if st.session_state.seed_solution_idea:
            data['seed_solution_idea'] = st.session_state.seed_solution_idea
        if st.session_state.seed_solution_script:
            data['seed_solution_script'] = st.session_state.seed_solution_script

        save_problem_data(st.session_state.problem_name, data)
        st.success("Seed solution saved successfully!")
        st.session_state.messages.append(("System", f"Seed solution for '{st.session_state.problem_name}' saved"))

# Handle LLM generate button for seed solution
if llm_gen_seed:
    if not st.session_state.problem_name:
        st.error("Please define a problem first (Section 1)")
    elif not st.session_state.problem_description:
        st.error("Please enter a problem description first (Section 1)")
    elif not st.session_state.function_signature:
        st.error("Please define function signature first (Section 2)")
    elif not st.session_state.evaluation_script:
        st.error("Please define evaluation script first (Section 2)")
    else:
        try:
            # Call the LLM to generate seed solution
            seed_idea_update, seed_script_update = orcanvas.generate_seed_solution(
                problem_description=st.session_state.problem_description,
                function_signature=st.session_state.function_signature,
                evaluation_script=st.session_state.evaluation_script,
                seed_solution_idea=st.session_state.seed_solution_idea if st.session_state.seed_solution_idea else None,
                seed_solution_script=st.session_state.seed_solution_script if st.session_state.seed_solution_script else None
            )

            # Update session state
            st.session_state.seed_solution_idea = seed_idea_update
            st.session_state.seed_solution_script = seed_script_update

            # Save the updated data
            data = {
                'seed_solution_idea': seed_idea_update,
                'seed_solution_script': seed_script_update
            }
            save_problem_data(st.session_state.problem_name, data)

            st.success("Seed solution generated by LLM and saved!")
            st.session_state.messages.append(("System", f"Seed solution for '{st.session_state.problem_name}' generated by LLM"))
            st.rerun()
        except Exception as e:
            st.error(f"Error generating seed solution: {str(e)}")
            st.session_state.messages.append(("System", f"Error generating seed solution: {str(e)}"))

# Handle run button for seed solution evaluation
if run_seed:
    if not st.session_state.problem_name:
        st.error("Please define a problem first (Section 1)")
    else:
        success, output = run_evaluation(st.session_state.problem_name)
        if success:
            st.success("Seed solution evaluation completed successfully!")
            st.session_state.messages.append(("System", f"Seed solution evaluation for '{st.session_state.problem_name}' completed"))

            # Show output if any
            if output:
                with st.expander("Evaluation Output"):
                    st.code(output, language="text")
        else:
            st.error(f"Seed solution evaluation failed: {output}")
            st.session_state.messages.append(("System", f"Seed solution evaluation for '{st.session_state.problem_name}' failed: {output}"))

st.markdown("")
st.markdown("")




#-----Agent Control Section-----
st.markdown("""
<div style="background-color: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
    <h1 style="margin: 0; color: #003366;">Open Research Agent</h1>
</div>
""", unsafe_allow_html=True)

# Control section - 6 columns as per new layout
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    # "Gen config" button
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 100%; margin-top: 28px;">
    """, unsafe_allow_html=True)
    if st.button("Generate config file", type="secondary", use_container_width=True):
        # Run python src/oragent/cli.py --init-config
        import subprocess
        import importlib.util

        # Try to import oragent to check if it's installed as a package
        oragent_spec = importlib.util.find_spec("oragent")
        if oragent_spec is not None:
            # Package is installed, use the installed module
            module_name = "oragent.cli"
        else:
            # Fall back to local src/oragent module
            module_name = "src.oragent.cli"

        result = subprocess.run(
            [sys.executable, "-m", module_name, "--init-config"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            st.success("config.yaml generated successfully!")
            st.session_state.messages.append(("System", "config.yaml generated successfully"))
        else:
            st.error(f"Failed to generate config.yaml: {result.stderr}")
            st.session_state.messages.append(("System", f"Failed to generate config.yaml: {result.stderr}"))
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # "Select checkpoint" drop-down
    checkpoints_dir = Path.cwd() / "checkpoints"
    checkpoints = []
    if checkpoints_dir.exists():
        checkpoints = [p.name for p in checkpoints_dir.iterdir() if p.is_dir()]

    checkpoints.insert(0, "None")  # Add a None option
    selected_checkpoint = st.selectbox("Select checkpoint", checkpoints, index=0 if checkpoints else None, key="checkpoint_select")
    st.session_state.selected_checkpoint = selected_checkpoint
    
    # If user choose to use checkpoint, then we load the config to see what's the algorithm and problem used in the saved checkpoint
    if st.session_state.selected_checkpoint != 'None':
        config_file = checkpoints_dir / st.session_state.selected_checkpoint / "config.yaml"
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        checkpoint_selected_problem = config['problem'].lower().strip()
        checkpoint_selected_algorithm = config['algorithm'].lower().strip()
        print("\ncheckpoint problem:", checkpoint_selected_problem)
        print("checkpoint algorithm:", checkpoint_selected_algorithm)
    
with col3:
    # "Select problem" drop-down
    problems_dir = Path.cwd() / "problems"
    problems = []
    if problems_dir.exists():
        problems = [p.name for p in problems_dir.iterdir() if p.is_dir()]

    if st.session_state.selected_checkpoint != 'None':
        index = problems.index(checkpoint_selected_problem)
        disabled = True
        problem_label = f"Checkpoint problem: {checkpoint_selected_problem}"
    else:
        index = 0
        disabled = False
        problem_label = "Select problem"
    #print("problem index:", index)
    selected_problem = st.selectbox(problem_label, problems, index=index if problems else None, key="problem_select", disabled=disabled)
    st.session_state.selected_problem = selected_problem

with col4:
    # "Select alg" drop-down
    algorithms = ["ORAgent", "ReEvo", "EoH", "AEL", "FunSearch"]
    algorithms_lower = [x.lower() for x in algorithms]
    
    if st.session_state.selected_checkpoint != 'None':
        index = algorithms_lower.index(checkpoint_selected_algorithm)
        disabled = True
        algorithm_label = f"Checkpoint algorithm: {checkpoint_selected_algorithm}"
    else:
        index = 0
        disabled = False
        algorithm_label = "Select algorithm"
    #print("algorithm index:", index)
    selected_algorithm = st.selectbox(algorithm_label, algorithms, index=index if problems else None, key="algorithm_select", disabled=disabled)
    st.session_state.selected_algorithm = selected_algorithm

with col5:
    # "Select display" drop-down
    dislpay_modes = ["real-time messages", "history messages"]
    selected_display_mode = st.selectbox("Select display mode", dislpay_modes, index=0 if dislpay_modes else None, key="display_mode_select")
    st.session_state.selected_display_mode = selected_display_mode
    
with col6:
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 100%; margin-top: 28px;">
    """, unsafe_allow_html=True)
    # "Start" button
    if st.button("Start", type="primary", use_container_width=True):
        if selected_problem and selected_algorithm:
            start_backend(selected_checkpoint, selected_algorithm, selected_problem)
            st.session_state.messages.append(("System", f"Started {selected_algorithm} on {selected_problem}"))
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with col7:
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 100%; margin-top: 28px;">
    """, unsafe_allow_html=True)
    # "Stop" button
    if st.button("Stop", type="secondary", use_container_width=True):
        stop_backend()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")



#-----Agent Output Section-----
col_left, col_right = st.columns([1, 2])

with col_left:
    #----Agent Output Section - Solution Performance Info---
    # Solution Performance Info
    st.subheader("Solution Performance Info")

    results = load_results_json()
    if results:
        # Extract data for plotting
        total_responses = []
        total_function_evals = []
        best_obj = []

        for result in results:
            if 'total_responses' in result and 'best_obj_overall' in result:
                total_responses.append(result['total_responses'])
                best_obj.append(result['best_obj_overall'])
            if 'total_function_evals' in result:
                total_function_evals.append(result['total_function_evals'])
                
        if total_responses and best_obj:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=total_responses,
                y=best_obj,
                mode='lines+markers',
                name='Best Objective',
                line=dict(color='#FF6B6B')
            ))

            fig.update_layout(
                title="Solution Performance Curve",
                xaxis_title="Total Responses",
                yaxis_title="Best Objective Overall",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)
            
            # Add additional plot: Best Objective Overall (y) vs total function evals (x)
            if total_function_evals and best_obj and len(total_function_evals) == len(best_obj):
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=total_function_evals,
                    y=best_obj,
                    mode='lines+markers',
                    name='Best Objective',
                    line=dict(color='#4ECDC4')
                ))

                fig2.update_layout(
                    title="Best Objective vs Function Evaluations",
                    xaxis_title="Total Function Evaluations",
                    yaxis_title="Best Objective Overall",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )

                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Function evaluation data not available yet")

            # Show latest result
            latest = results[-1]
            st.markdown(f"**Latest Result:**")
            st.markdown(f"- Iteration: {latest.get('iteration', 'N/A')}")
            st.markdown(f"- Total Responses: {latest.get('total_responses', 'N/A')}")
            st.markdown(f"- Total Function Evaluations: {latest.get('total_function_evals', 'N/A')}")
            st.markdown(f"- Best Objective: {latest.get('best_obj_overall', 'N/A')}")
        else:
            st.info("No performance data available yet")
    else:
        st.info("No solution performance info available")

    st.markdown("---")



    #---Agent Output Section - Experiment Config Info---
    # Experiment Config Info
    st.subheader("Experiment Configuration")

    # Load experiment config
    #experiment_config = load_experiment_config()
    # Load complete config
    experiment_config = load_complete_config()
    # Display experiment config details
    if experiment_config:
        # Display experiment config details
        #function_to_evolve = experiment_config.get('function_to_evolve', 'Unknown')
        #obj_type = experiment_config.get('obj_type', 'Unknown')
        function_to_evolve = experiment_config['experiment']['function_to_evolve']
        obj_type = experiment_config['experiment']['obj_type']

        # Get problem name from session state
        problem_name = st.session_state.get('selected_problem', 'Unknown')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style="font-size: 14px; font-weight: bold; margin-bottom: 4px;">Problem</div>
            <div style="font-size: 16px;">{problem_name}</div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="font-size: 14px; font-weight: bold; margin-bottom: 4px;">Function to Evolve</div>
            <div style="font-size: 16px;">{function_to_evolve}</div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style="font-size: 14px; font-weight: bold; margin-bottom: 4px;">Objective Type</div>
            <div style="font-size: 16px;">{obj_type}</div>
            """, unsafe_allow_html=True)

        # Show full config in expander
        with st.expander("View Full Experiment Config"):
            st.json(experiment_config)
    else:
        st.info("No experiment configuration available")



with col_right:
    #---Agent Output Section - Real-time Messages---
    # Display Messages
    st.subheader("System Messages")

    # Process messages from queue (only for real-time mode)
    if st.session_state.selected_display_mode == "real-time messages":
        while not st.session_state.message_queue.empty():
            try:
                line = st.session_state.message_queue.get_nowait()
                entity, content = parse_message_line(line)
                if entity or content:
                    st.session_state.messages.append((entity or "System", content))
            except queue.Empty:
                break

    # Display messages based on selected mode
    messages_container = st.container(height=600)

    with messages_container:
        if st.session_state.selected_display_mode == "real-time messages":
            # Display real-time messages in reverse order (newest at top)
            for entity, content in reversed(st.session_state.messages[-100:]):  # Show last 100 messages
                # Convert entity to lowercase for case-insensitive lookup
                entity_lower = entity.lower() if entity else 'default'
                if entity_lower != 'default':
                    entity_str = f"[{entity}]"
                else:
                    entity_str = ''
                if entity_lower in ENTITY_COLORS:
                    color = ENTITY_COLORS[entity_lower]
                else:
                    color = ENTITY_COLORS['default']

                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 8px; margin: 4px 0; border-left: 4px solid {color}; border-radius: 4px;">
                    <span style="color: {color}; font-weight: bold;">{entity_str}</span> {content}
                </div>
                """, unsafe_allow_html=True)

        elif st.session_state.selected_display_mode == "history messages":
            # Load and display saved output.txt history
            output_content = load_output_txt()
            if output_content:
                # Split content into lines and display each line
                lines = output_content.split('\n')
                for line in lines:
                    if line.strip():  # Skip empty lines
                        entity, content = parse_message_line(line)
                        # Convert entity to lowercase for case-insensitive lookup
                        entity_lower = entity.lower() if entity else 'default'
                        if entity_lower != 'default':
                            entity_str = f"[{entity}]"
                        else:
                            entity_str = ''
                        if entity_lower in ENTITY_COLORS:
                            color = ENTITY_COLORS[entity_lower]
                        else:
                            color = ENTITY_COLORS['default']

                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 8px; margin: 4px 0; border-left: 4px solid {color}; border-radius: 4px;">
                            <span style="color: {color}; font-weight: bold;">{entity_str}</span> {content}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No saved history available. Run an experiment first to generate output.txt")
    
    st.markdown("---")



    #---Agent Output Section - Research progress Info---
    # Research Progress Info
    st.subheader("Research Progress Info")

    progress_content = load_progress_txt()
    if progress_content:
        # Display as monospace text
        st.code(progress_content, language="text")
    else:
        st.info("No research progress info available")

st.markdown("---")



#---Agent Output Section - Long-term Reflection---
# Display Long-term reflection if it exists
st.subheader("Long-term Reflection")

long_term_reflection_content = load_long_term_reflection_txt()
if long_term_reflection_content:
    # Display as monospace text
    st.code(long_term_reflection_content, language="text")
else:
    st.info("No long-term reflection info available")
        
st.markdown("---")

# User feedback section
st.subheader("User Feedback")

# Create a text input with default placeholder
user_feedback = st.text_area(
    "Share your thoughts or ideas with OR-Agent:",
    value="",
    placeholder="tell me what you think...",
    height=100
)

# Submit button
if st.button("Submit Feedback"):
    if user_feedback.strip():
        st.success("Thank you for your feedback!")
        # The user_feedback variable now contains the submitted text
        # It can be used later as needed
    else:
        st.warning("Please enter some feedback before submitting.")

# TODO: we intended to let user to able to interact with OR-Agent in real-time
# we leave this for future updates
# "real-time user feedback" - This requires further exploration, analysis, and validation, and is marked with a TODO flag

st.markdown("---")



#---Agent Output Section - Solution Database Info---
# Solution Database Info (optional)
st.subheader("Solution Database Info")

db_content = load_solution_database_txt()
if db_content:
    # Parse and display structured view
    parsed_db = parse_solution_database(db_content)

    if parsed_db and parsed_db['islands']:
        # Create tabs for each island
        tabs = st.tabs([f"Island {i['number']}" for i in parsed_db['islands']])

        for idx, (tab, island) in enumerate(zip(tabs, parsed_db['islands'])):
            with tab:
                st.markdown(f"**Island {island['number']}**")
                st.markdown(f"- Clusters: {len(island['clusters'])}")

                # Show clusters
                for cluster in island['clusters']:
                    with st.expander(f"Cluster {cluster['number']}"):
                        st.markdown(f"Solutions: {len(cluster['solutions'])}")

    # Show summary
    if parsed_db and parsed_db['summary']:
        st.markdown("**Summary Statistics:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div style="font-size: 18px; color: #666;">Total Solutions</div>
            <div style="font-size: 20px; font-weight: bold;">{parsed_db['summary'].get('total_solutions', 'N/A')}</div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="font-size: 18px; color: #666;">Total Clusters</div>
            <div style="font-size: 20px; font-weight: bold;">{parsed_db['summary'].get('total_clusters', 'N/A')}</div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style="font-size: 18px; color: #666;">Avg Cluster Size</div>
            <div style="font-size: 20px; font-weight: bold;">{parsed_db['summary'].get('avg_cluster_size', 'N/A')}</div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div style="font-size: 18px; color: #666;">Best Score</div>
            <div style="font-size: 20px; font-weight: bold;">{parsed_db['summary'].get('best_score', 'N/A')}</div>
            """, unsafe_allow_html=True)

    # Also show raw content in expander
    with st.expander("View Raw Database Content"):
        st.code(db_content, language="text")
else:
    st.info("No solution database info available")

st.markdown("---")



#-----Footer section-----
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Developer**: MAGIC Lab, Tongji Univ.")
with col2:
    st.markdown("**Terms of Use**")
with col3:
    st.markdown("**Contact**: liuqi_tj@hotmail.com")

# Auto-refresh
if st.session_state.process is not None:
    # Check if process has ended
    returncode = st.session_state.process.poll()
    if returncode is not None:
        # Process has ended
        if returncode != 0:
            st.session_state.messages.append(("System", f"Backend process crashed with exit code {returncode}"))
        else:
            st.session_state.messages.append(("System", "Backend process completed successfully"))
        st.session_state.process = None

    time.sleep(2)
    st.rerun()