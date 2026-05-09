# Benchmark Problems

## Overview

This benchmark is adapted from [**ReEvo**](https://ai4co.github.io/reevo/)[1], which originally consists of six types of combinatorial optimization problems (COPs). We have extended the benchmark by adding a cooperative driving problem that involves complex simulation environments using [SUMO](https://eclipse.dev/sumo/).

This benchmark is also shared at [Huggingface](https://huggingface.co/datasets/qiliuchn/operations-research).

## Table of Contents

1. [Types of Functions to Evolve](#types-of-functions-to-evolve)
2. [Problem Details](#problem-details)
3. [How to Customize Your Own Benchmark](#how-to-customize-your-own-benchmark)
4. [References](#references)

## Types of Functions to Evolve

The functions to evolve are categorized into three groups:

### Classical Metaheuristics (ACO / GA / GLS)
- **Ant Colony Optimization (ACO)**[2]: Evolve ACO heuristic components, such as the computation of desirability and pheromone guidance.
- **Guided Local Search (GLS)**[3]: Evolve the penalty heuristic that guides perturbations during GLS.
- **Genetic Algorithm (GA)**[4]: Evolve GA-related operators and heuristics (the domain-specific logic within the GA pipeline, as defined by the problem wrapper).

### Attention Reshaping in Neural Combinatorial Optimization (POMO / LEHD)
- **Policy Optimization with Multiple Optima (POMO)**[5]: A reinforcement learning training and inference framework for neural constructive solvers that exploits symmetry and uses multiple rollouts from different starting conditions to stabilize and improve solution quality. We evolve attention reshaping heuristics inserted into the neural solver (not the model weights). For POMO settings, download checkpoints from the [official repository](https://github.com/yd-kwon/POMO) and place them in the corresponding directories (e.g., place `checkpoint-3100.pt` for TSP at `problems/tsp_pomo/checkpoints/checkpoint-3100.pt`).

- **Neural Combinatorial Optimization with Light Encoder, Heavy Decoder (LEHD)**[6]: This approach shifts modeling capacity into the decoder while keeping the encoder lightweight, aiming for better generalization and scaling in constructive routing solvers. We evolve attention reshaping heuristics. For LEHD settings, download checkpoints and data from the [official repository](https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD) and place them in the corresponding directories.

### Direct Solution Construction Heuristics
We can evolve functions that directly construct solutions. For example:
- For online bin packing problems, evolve the function that generates priority scores for each bin; the solver then selects the bin with the highest priority.
- For cooperative driving problems, evolve the function that generates actions for all drivers.


## Problem Details

The benchmark problems are stored at `[project_root]/problems`. Detailed descriptions of each problem are provided below.

### Traveling Salesman Problems (TSPs)
The Traveling Salesman Problem (TSP) is a classic optimization challenge that seeks the shortest possible route for a salesman to visit each city in a list exactly once and return to the origin city.

- **TSP via Ant Colony Optimization (`tsp_aco`)**: Find the shortest path that visits all given nodes and returns to the starting node. ACO implementations are adapted from [DeepACO](https://github.com/henry-yeh/DeepACO)[2].
- **TSP via Guided Local Search (`tsp_gls`)**: Use Guided Local Search (GLS)[3] to find the shortest path.
- **TSP via LEHD (`tsp_lehd`)**: Use LEHD[6] to find the shortest path.
- **TSP via POMO (`tsp_pomo`)**: Use POMO[5] to find the shortest path.
- **TSP via Constructive Routing Solvers (`tsp_constructive`)**: Evolve functions that directly construct solutions for TSP.

### Capacitated Vehicle Routing Problems (CVRPs)
The Capacitated Vehicle Routing Problem (CVRP) extends the TSP by adding constraints on vehicle capacity. Each vehicle can carry a limited load, and the objective is to minimize the total distance traveled while delivering goods to various locations.

- **CVRP via Ant Colony Optimization (`cvpr_aco`)**: Solve CVRP using Ant Colony Optimization (ACO)[2].
- **CVRP via LEHD (`cvpr_lehd`)**: Solve CVRP using LEHD[6].
- **CVRP via POMO (`cvpr_pomo`)**: Solve CVRP using POMO[5].

### Bin Packing Problems (BPPs)
The Bin Packing Problem requires packing objects of different volumes into a finite number of bins or containers of fixed volume to minimize the number of bins used. This problem is widely applicable in manufacturing, shipping, and storage optimization.

- **BPP via Ant Colony Optimization (`bpp_offline_aco`)**
- **Online BPP (`bpp_online`) via Priority Score Heuristics**

### Orienteering Problems (OPs)
In the Orienteering Problem (OP), the goal is to maximize the total score collected by visiting nodes while subject to a maximum tour length constraint.

- **OP for Routing Problems via Ant Colony Optimization (`op_aco`)**

### Multiple Knapsack Problems (MKPs)
The Multiple Knapsack Problem (MKP) involves distributing a set of items, each with a given weight and value, among multiple knapsacks to maximize the total value without exceeding the capacity of any knapsack.

- **MKP via Ant Colony Optimization (`mkp_aco`)**: Solve MKP using Ant Colony Optimization (ACO)[2].

### Decap Placement Problem (DPPs)
The Decap Placement Problem (DPP) is a critical hardware design optimization issue that involves finding the optimal placement of decoupling capacitors (decap) within a power distribution network (PDN) to enhance power integrity (PI). Decoupling capacitors are hardware components that help reduce power noise and ensure a stable power supply to operating integrated circuits in hardware devices such as CPUs, GPUs, and AI accelerators.

- **Decap Placement Problem (DPP) for Electronic Design Automation (EDA) Problems via Genetic Algorithm (GA)[4] (`dpp_ga`)**

### Cooperative Driving Problem (CDPs)
The Cooperative Driving Problem (CDP) is a complex optimization challenge that involves optimizing the driving behavior of multiple vehicles on a road segment.

- **Cooperative Driving Problem (CDP) (`driving`)**: Evolve functions that directly construct driving actions for each time step.




## How to Customize Your Own Benchmark

### Command line arguments requirements (same for all problems)
Command line arguments:
1. `root_dir`: the project root directory; knowing project root can help you to load data; default: current working directory (os.getcwd());
    Eval script need this to load dataset since eval script may be generated and stored in a different location to support parallelism;
2. `file_output_prefix`: the output file prefix: this prefix can be used to save output files during evaluation for inspection purposes; 
    we use prefix since you may want more than just a folder name; say you may want to add solution id to the output filename;
    file will be saved by: `with open(f"{file_output_prefix}<filename>", 'w'):\n...`;
    absolute path is recommended;
    default: '', which means just save to current working directory;
3. `mode`: train or val; default: val;
4. `problem_size`; default: 50 (Note: this value differs for each problem!);


### How to run
You can manually run the script this way:
```
python eval.py \
    --root_dir=<path_to_project_root> \
    --file_output_prefix=<path_to_output_file> \
    --mode=val \
    --problem_size=50
```

`Evaluator` class will run eval script like this:
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
This makes it easier for you to add new problems - you don't need to modify the `Evaluator` class.


### Output requirements (same for all problems)
Eval script should print out `metrics`,`features`, and `score`;
1. `metrics`: a dict that map test name (str) to metrics (Dict),
     or a dict that maps performance index name to values; 
    `metrics` dict is used for user and AI agent inspection;
    It's optional but we strongly recommend you to prepare a detailed metrics for each problem; as this can help LLM to better understand the solution performance!
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

Then stdout should be:
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

### Dynamic solution function loading
Solution function scripts will be generated on the fly and loaded dynamically.
To enable parallelism, we will save different solution script to different files. Hence `Evaluator` will need to load the solution script dynamically.
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




## References

[1] Ye, H., Wang, J., Cao, Z., Berto, F., Hua, C., Kim, H., Park, J., & Song, G. (2024). Reevo: Large language models as hyper-heuristics with reflective evolution. *Advances in Neural Information Processing Systems*, 37, 43571–43608.

[2] Ye, H., Wang, J., Cao, Z., Liang, H., & Li, Y. (2023). DeepACO: Neural-enhanced ant systems for combinatorial optimization. *Advances in Neural Information Processing Systems*, 36, 43706–43728.

[3] Voudouris, C., & Tsang, E. (1999). Guided local search and its application to the traveling salesman problem. *European Journal of Operational Research*, 113(2), 469–499.

[4] Park, H., Kim, H., Kim, H., Park, J., Choi, S., Kim, J., Son, K., Suh, H., Kim, T., Ahn, J., & Kim, J. (2023, October). Versatile genetic algorithm-bayesian optimization (GA-BO) bi-level optimization for decoupling capacitor placement. In *2023 IEEE 32nd Conference on Electrical Performance of Electronic Packaging and Systems (EPEPS)* (pp. 1–3). IEEE.

[5] Kwon, Y. D., Choo, J., Kim, B., Yoon, I., Gwon, Y., & Min, S. (2020). POMO: Policy optimization with multiple optima for reinforcement learning. *Advances in Neural Information Processing Systems*, 33, 21188–21198.

[6] Luo, F., Lin, X., Liu, F., Zhang, Q., & Wang, Z. (2023). Neural combinatorial optimization with heavy decoder: Toward large scale generalization. *Advances in Neural Information Processing Systems*, 36, 8845–8864.