# src/oragent/cli.py
""" 
Usage:
```bash
pip install oragent

oragent --init-config  # Create a template config.yaml in current directory; users can config OR-Agent by revising it
oragent  # run oragent by using config file in current working directory or built-in config
oragent --checkpoint <checkpoint_name>  # Run by loading checkpoint
oragent --config=config.yaml --algorithm <algorithm> --problem <problem>  # Run oragent by loading config from specified file
```
"""
import os
import sys
from pathlib import Path
import argparse
import yaml
from typing import NoReturn
import socket
import getpass


welcome = r"""
   ____  ____      ___                    __ 
  / __ \/ __ \    /   | ____ ____  ____  / /_
 / / / / /_/ /   / /| |/ __ `/ _ \/ __ \/ __/
/ /_/ / _, _/   / ___ / /_/ /  __/ / / / /_  
\____/_/ |_|   /_/  |_\__, /\___/_/ /_/\__/  
                     /____/                  
"""


def create_default_config(output_path="config.yaml"):
    """Create a default config.yaml file in the specified location."""
    package_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load default config
    with open(f"{package_dir}/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=4)
        
    print(f"\n>>>[OR-Agent] Created default config at: {output_path}")


def load_config(config_path=None):
    """
    Load configuration with fallback to defaults.
    
    Priority:
    1. Explicit path provided via --config
    2. config.yaml in current working directory
    3. Built-in defaults
    """
    # Option 1: Explicit path
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"\n>>>[OR-Agent] Config file not found: {config_path}")
        print(f"\n>>>[OR-Agent] Loading config from: {config_file}")
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    # Option 2: Current directory
    cwd_config = Path.cwd() / "config.yaml"
    if cwd_config.exists():
        print(f"\n>>>[OR-Agent] Loading config from: {cwd_config}")
        with open(cwd_config, 'r') as f:
            return yaml.safe_load(f)
    
    # Option 3: Use built-in defaults
    print("\n>>>[OR-Agent] No config.yaml found, using default configuration")
    print(f">>>[OR-Agent] Tip: Run 'python cli.py --init-config' to create a template {Path.cwd()}/config.yaml")
    print(">>>[OR-Agent] Now run with default configuration...")
    
    # Load default config
    package_dir = Path(__file__).parent
    with open(f"{package_dir}/config.yaml", 'r') as f:
        return yaml.safe_load(f)
            
    return config



def main():
    """
    Application entry point.
    Returns process exit code.
    """
    # =====Parse commandline arguments=====
    parser = argparse.ArgumentParser(description="OR Agent")
    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Create a default config.yaml in current directory"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="Checkpoint to load"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--algorithm", 
        type=str, 
        default=None,
        help="algorithm to use"
    )
    parser.add_argument(
        "--problem", 
        type=str, 
        default=None,
        help="problem to solve"
    )
    parser.add_argument(
        "--max-evolutions",
        type=int,
        default=None,
        help="Maximum number of evolutions"
    )
    parser.add_argument(
        "--init-pop-size",
        type=int,
        default=None,
        help="Initial population size"
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help="Population size (for ReEvo, EoH, AEL)"
    )
    parser.add_argument(
        "--num-children",
        type=int,
        default=None,
        help="Number of children to generate"
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=None,
        help="Timeout seconds for each solution evaluation"
    )
    parser.add_argument(
        "--fast-exploration-for-crossover",
        action="store_true",
        help="Use fast exploration mode for crossover"
    )
    parser.add_argument(
        "--max-debug-rounds",
        type=int,
        default=None,
        help="Maximum number of debug rounds for each solution code debugging"
    )
    parser.add_argument(
        "--max-experiment-repeats",
        type=int,
        default=None,
        help="Maximum number of experiment repeats for each solution"
    )
    parser.add_argument(
        "--elitist-as-root-period",
        type=int,
        default=None,
        help="Research round period that elitist is used as root"
    )
    parser.add_argument(
        "--elitist-enlargement-factor",
        type=float,
        default=None,
        help="Children enlargement factor for elitist as parent"
    )
    parser.add_argument(
        "--elitist-experiment-factor",
        type=float,
        default=None,
        help="Experiment enlargement factor for elitist solution"
    )
    parser.add_argument(
        "--reflection-compression",
        type=int,
        default=None,
        help="Timeout seconds for each solution evaluation"
    )
    parser.add_argument(
        "--reflection-period",
        type=int,
        default=None,
        help="Batch size of experiment reflections used to update long-term memory"
    )
    parser.add_argument(
        "--reflection-clearance-period",
        type=int,
        default=None,
        help="Research round period for clearing long-term memory"
    )
    parser.add_argument(
        "--reflection-disabled-for-crossover",
        action="store_true",
        help="Disable long-term reflection when doing crossover"
    )
    parser.add_argument(
        "--reflection-elitist-synchro",
        action="store_true",
        help="Synchronize long-term reflection update and elitist-as-root event"
    )
    parser.add_argument(
        "--evaluation-description-disabled",
        action="store_true",
        help="Disable using eval description"
    )
    parser.add_argument(
        "--ideas-coordinated-generation-disabled",
        action="store_true",
        help="Disable coordinated idea generation"
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Timeout seconds for each solution evaluation"
    )
    parser.add_argument(
        "--num-islands",
        type=int,
        default=None,
        help="Number of islands for solution database"
    )
    parser.add_argument(
        "--reset-period-minutes",
        type=int,
        default=None,
        help="Reset period for solution database"
    )
    parser.add_argument(
        "--cluster-sampling-temperature-init",
        type=int,
        default=None,
        help="Initial temperature for cluster sampling"
    )
    parser.add_argument(
        "--cluster-sampling-temperature-period",
        type=int,
        default=None,
        help="Period for cluster sampling temperature"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Output for WebUI"
    )
    args = parser.parse_args()
    
    # =====Welcome=====
    if not args.web:
        # terminal display
        print(welcome)
    else:
        # web display
        print("You're using WebUI of OR-Agent; stdout will be displayed in this message section.")
    print("\nWelcome to OR-Agent!")
    
    # =====Create config template and exit=====
    if args.init_config:
        create_default_config("config.yaml")
        print("\n>>>[OR-Agent] Edit config.yaml and then run")
        return
    
    # =====Load config=====
    if args.checkpoint:  
        # -----load config from checkpoint-----
        # if loading from checkpoint, algorithm and problem cannot be changed
        checkpoint_directory = f'checkpoints/{args.checkpoint}'
        print(f"\n>>>[OR-Agent] Loading checkpoint:", checkpoint_directory)
        config_path = os.path.join(checkpoint_directory, 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:  
        # -----start a new session config-----
        config = load_config(args.config)  # load from args.config or default config

        # -----Update config if user specified in command line-----
        # We should override settings in config
        # Note: you should not change any settings below if you run from checkpoint!
        if args.algorithm:
            config['algorithm'] = args.algorithm
            
        if args.problem:
            config['problem'] = args.problem
            # update experiment config
            with open(f"{Path.cwd()}/problems/{args.problem}/settings.yaml", 'r') as f:
                experiment_config = yaml.safe_load(f)
            config['experiment'] = experiment_config
        
        if args.max_evolutions != None:
            config['max_evolutions'] = args.max_evolutions
        
        if args.init_pop_size != None:
            config['init_pop_size'] = args.init_pop_size
        
        if args.pop_size != None:
            config['pop_size'] = args.pop_size
        
        if args.num_children != None:
            config['num_children'] = args.num_children
            
        if args.max_tree_depth != None:
            config['max_tree_depth'] = args.max_tree_depth
        
        if args.fast_exploration_for_crossover:
            config['fast_exploration_for_crossover'] = True
        
        if args.max_debug_rounds != None:
            config['max_debug_rounds'] = args.max_debug_rounds
        
        if args.max_experiment_repeats != None:
            config['max_experiment_repeats'] = args.max_experiment_repeats
    
        if args.elitist_as_root_period != None:
            config['elitist_as_root_period'] = args.elitist_as_root_period
        
        if args.elitist_enlargement_factor != None:
            config['elitist_enlargement_factor'] = args.elitist_enlargement_factor
        
        if args.elitist_experiment_factor != None:
            config['elitist_experiment_factor'] = args.elitist_experiment_factor
        
        if args.reflection_compression != None:
            config['reflection_compression'] = args.reflection_compression
            
        if args.reflection_period != None:
            config['reflection_period'] = args.reflection_period
        
        if args.reflection_clearance_period != None:
            config['reflection_clearance_period'] = args.reflection_clearance_period
        
        if args.reflection_disabled_for_crossover:
            config['reflection_disabled_for_crossover'] = True
            
        if args.reflection_elitist_synchro:
            config['reflection_elitist_synchro'] = True
        
        if args.evaluation_description_disabled:
            config['evaluation_description_disabled'] = True
        
        if args.ideas_coordinated_generation_disabled:
            config['ideas_coordinated_generation_disabled'] = True
        
        if args.timeout_seconds != None:
            config['evaluation']['timeout_seconds'] = args.timeout_seconds
            
        if args.num_islands != None:
            config['database']['num_islands'] = args.num_islands
        
        if args.reset_period_minutes != None:
            config['database']['reset_period_minutes'] = args.reset_period_minutes
            
        if args.cluster_sampling_temperature_init != None:
            config['database']['cluster_sampling_temperature_init'] = args.cluster_sampling_temperature_init
        
        if args.cluster_sampling_temperature_period != None:
            config['database']['cluster_sampling_temperature_period'] = args.cluster_sampling_temperature_period
    
    # -----output dir can always be updated (even if loading from checkpoint)-----
    if args.output_dir:
        config['output_dir'] = args.output_dir
    else:
        config['output_dir'] = None
    
    # -----Load experiment config-----
    algorithm = config['algorithm'].lower().strip()
    problem = config['problem'].lower().strip()  # the problem to solve
        
    if 'experiment' not in config:
        experiment_config_path = Path.cwd() / "problems" / problem / "settings.yaml"
        # Load experiment config
        with open(experiment_config_path, 'r') as f:
            experiment_config = yaml.safe_load(f)
        config['experiment'] = experiment_config
                
    # -----Log config file-----
    output_dir = config['output_dir'] or  Path.cwd() / "outputs" / algorithm / problem
    os.makedirs(output_dir, exist_ok=True)  # output dir
    os.makedirs(f"{output_dir}/details", exist_ok=True)  # folder in output dir to store details like solution and eval scripts
    config_path = f"{output_dir}/config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    
    #try:  # disable try...except... when debugging
    # =====Print out some important config=====
    print(f"\n>>>[OR-Agent] Running algorithm \"{config['algorithm']}\" on problem \"{config['problem']}\"")
    print(f"\n>>>[OR-Agent] Make sure that your problem is defined in \"{Path.cwd()}/problems/{config['problem']}\" directory")
    
    print(f"\n>>>[OR-Agent] Settings:")
    output_dir = config['output_dir'] or f"{Path.cwd()}/outputs/{config['algorithm']}/{config['problem']}"
    print(f">>>[OR-Agent] Output directory: {output_dir}")
    print(f">>>[OR-Agent] max_evolutions: {config['max_evolutions']}")
    print(f">>>[OR-Agent] init_pop_size: {config['init_pop_size']}")
    print(f">>>[OR-Agent] pop_size: {config['pop_size']}")
    print(f">>>[OR-Agent] num_children: {config['num_children']}")
    print(f">>>[OR-Agent] max_tree_depth: {config['max_tree_depth']}")
    print(f">>>[OR-Agent] fast_exploration_for_crossover: {config['fast_exploration_for_crossover']}")
    print(f">>>[OR-Agent] max_debug_rounds: {config['max_debug_rounds']}")
    print(f">>>[OR-Agent] max_experiment_repeats: {config['max_experiment_repeats']}")
    print(f">>>[OR-Agent] elitist_as_root_period: {config['elitist_as_root_period']}")
    print(f">>>[OR-Agent] elitist_enlargement_factor: {config['elitist_enlargement_factor']}")
    print(f">>>[OR-Agent] elitist_experiment_factor: {config['elitist_experiment_factor']}")
    print(f">>>[OR-Agent] reflection_compression: {config['reflection_compression']}")
    print(f">>>[OR-Agent] reflection_period: {config['reflection_period']}")
    print(f">>>[OR-Agent] reflection_clearance_period: {config['reflection_clearance_period']}")
    print(f">>>[OR-Agent] reflection_disabled_for_crossover: {config['reflection_disabled_for_crossover']}")
    print(f">>>[OR-Agent] reflection_elitist_synchro: {config['reflection_elitist_synchro']}")
    print(f">>>[OR-Agent] evaluation_description_disabled: {config['evaluation_description_disabled']}")
    print(f">>>[OR-Agent] ideas_coordinated_generation_disabled: {config['ideas_coordinated_generation_disabled']}")
    print(f">>>[OR-Agent] llm_provider: {config['model']['llm_provider']}")
    print(f">>>[OR-Agent] timeout_seconds: {config['evaluation']['timeout_seconds']}")
    print(f">>>[OR-Agent] autosave_interval_minutes: {config['autosave_interval_minutes']}")
    # Get username and hostname
    username = getpass.getuser()
    hostname = socket.gethostname()
    # Combine them
    #user_host = f"{username}@{hostname}"
    user_host = f"{username}"
    print(f">>>[OR-Agent] python_path: {config['evaluation']['python_path'][user_host]}")
    
    # =====Load evolution model=====
    algorithm = config['algorithm'].lower().strip()
    
    if algorithm == "oragent":
        from oragent.core import ORAgent as Agent
    elif algorithm == "reevo":
        from oragent.reevo import ReEvo as Agent
    elif algorithm == "ael":
        from oragent.ael import AEL as Agent
    elif algorithm == "eoh":
        from oragent.eoh import EoH as Agent
    elif algorithm == "funsearch":
        from oragent.funsearch import FunSearch as Agent
    else:
        raise NotImplementedError

    # Create evolution model instance
    # All agents implement the same interface:
    # - constructor that takes checkpoint or config dict as inputs;
    # - Agent.run() start the process.
    agent = Agent(checkpoint=args.checkpoint, config=config)
    
    # =====Run agent=====
    # This is the main process; generated solutions will be automatically saved to output directory
    agent.run()
    return 0

    #except KeyboardInterrupt:
    #    logging.info("Interrupted by user")
    #    return 130

    #except Exception:
    #    logging.exception("Fatal error")
    #    return 1


def _exit(code: int) -> NoReturn:
    sys.exit(code)


if __name__ == "__main__":
    _exit(main())