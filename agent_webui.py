# agent.py
""" 
# WebUI for Open Research Agent
User can start OR-Agent research and monitor research progress using this web UI.
This web UI can also function as research result visualizer, like tensorboard.


## Layout illustration
+------------------------------------------------------------+
|      OR-Agent                                              |      Header section
+------------------------------------------------------------+
|  +------------------------------------------------------+  |
|  |[Gen config][Select problem][Select alg][Start][Stop] |  |      Control section
|  +------------------------------------------------------+  |
|  +-------------+  +-------------------------------------+  |
|  |   Solution  |  |                                     |  |
|  | Performance |  |                                     |  |
|  |     Info    |  |            Real-time                |  |
|  +-------------+  |            System Messages          |  |       Output section
|  +-------------+  |            Scroll Window            |  |
|  |   Research  |  |                                     |  |
|  |   Progress  |  |                                     |  |
|  +-------------+  +-------------------------------------+  |
|  +------------------------------------------------------+  |
|  |         Long-term reflections (optional)             |  |       Long-Term Reflection Section
|  |         User feedback input                          |  |
|  +------------------------------------------------------+  |
|  +------------------------------------------------------+  |
|  |         Solution Database Info (optional)            |  |       Solution Database Section
|  |                                                      |  |
|  +------------------------------------------------------+  |
+------------------------------------------------------------+
|        Developer | Terms of Use | Contact                  |       Footer section
+------------------------------------------------------------+



## Functionality
WebUI need to display the following types of messages:
 - real-time system messages
 - solution performance info
 - research progress info
 - solution database info (optional)
 

### Control section
Control section have the following items:
 - "Gen config" button: user click this, invoke `python src/oragent/cli.py --init-config` to generate a config.yaml file at current working directory.
 - "Select checkpoint" drop-down list: the drop-down list will include all checkpoints in `<project_root>/checkpoints` directory.
 - "Select problem" drop-down list: the drop-down list will include all problems in `<project_root>/problems` directory.
 - "Select alg" drop-down list: the drop-down list will include options: "ORAgent", "ReEvo", "EoH", "AEL", "FunSearch".
 - "Start" button: user click this, invoke `python src/oragent/cli.py --algorithm=<algorithm>` or `--problem=<problem>` to start the backend process.
 - "Stop" button: user click this, stop the backend process. 


### Real-time system messages display
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


### Solution performance info display
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


### Research progress info display
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


### Solution database info display (optional)
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



## Implementation
Use streamlit to build the webui.
Start the backend process and capture the stdout.
Check the files in the output directory for update every few seconds.


Check `src/oragent/cli.py` for the package command line interface.
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

# Initialize session state
if 'process' not in st.session_state:
    st.session_state.process = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = {}
if 'message_queue' not in st.session_state:
    st.session_state.message_queue = queue.Queue()

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

# Streamlit UI
st.set_page_config(layout="wide", page_title="OR-Agent WebUI")



# Header section
st.markdown("""
<div style="background-color: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
    <h1 style="margin: 0; color: #003366;">Open Research Agent</h1>
</div>
""", unsafe_allow_html=True)
#st.markdown("---")

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



# Output section
col_left, col_right = st.columns([1, 2])

with col_left:
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

    # Research Progress Info
    st.subheader("Research Progress Info")

    progress_content = load_progress_txt()
    if progress_content:
        # Display as monospace text
        st.code(progress_content, language="text")
    else:
        st.info("No research progress info available")

        
        
        
st.markdown("---")

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

# Footer section
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