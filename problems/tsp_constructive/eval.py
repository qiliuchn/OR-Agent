# Evaluation script for TSP-constructive problem
import os
import sys
import traceback
import numpy as np
from typing import Dict, List, Tuple, Any
import math
import argparse
from scipy.spatial import distance_matrix
from copy import copy
import inspect
import seed_solution as solution_module  # Note: solution module script is generated and saved on the fly


# =====Load function to evolve=====
problem = "tsp_constructive"
select_next_node = getattr(solution_module, "select_next_node")  # Get function to evolve


# =====Evaluation Function=====
def eval_heuristic(node_positions: np.ndarray) -> float:
    '''
    Generate solution for TSP problem using the GPT-generated heuristic algorithm.
    
    Args:
        node_positions (np.ndarray): 2D array of node positions of shape (problem_size, 2).
    
    Returns:
        obj (float):The length of the generated tour.
    '''
    problem_size = node_positions.shape[0]
    # calculate distance matrix
    dist_mat = distance_matrix(node_positions, node_positions)
    # set the starting node
    start_node = 0
    solution = [start_node]
    # init unvisited nodes
    unvisited = set(range(problem_size))
    # remove the starting node
    unvisited.remove(start_node)
    # run the heuristic
    for _ in range(problem_size - 1):
        next_node = select_next_node(
            current_node=solution[-1],
            destination_node=start_node,
            unvisited_nodes=copy(unvisited),
            distance_matrix=dist_mat.copy(),
        )
        solution.append(next_node)
        if next_node in unvisited:
            unvisited.remove(next_node)
        else:
            raise KeyError(f"Node {next_node} is already visited.")
    
    # calculate the length of the tour
    obj = 0
    for i in range(problem_size):
        obj += dist_mat[solution[i], solution[(i + 1) % problem_size]]
    return obj
    

# =====Helper functions=====
def get_feature(metrics: Dict[int, float]) -> Tuple[int, ...]:
    """
    Convert the metrics dict to a feature vector

    Args:
        metrics (dict): A mapping of test problem size (int) to a score (float).

    Returns:
        (tuple): a tuple of discretized scores sorted by problem size
    """
    scores = metrics.values()
    features = tuple([int(x) for x in scores])
    return features
    
def get_score(metrics: Dict[int, float]) -> float:
    """
    Convert the metrics dict to a score

    Args:
        metrics (dict): A mapping of test problem size (int) to a score (float).

    Returns:
        (float): a score
    """
    return sum(metrics.values()) / len(metrics)


# =====Main Function=====
if __name__ == '__main__':
    # -----Parse command line arguments (same for all problems)-----
    parser = argparse.ArgumentParser(description='Evaluation script.')
    parser.add_argument(
        '--root_dir',
        type=str,
        default=os.getcwd(),
        help='Project root directory for loading data (default: current working directory)'
    )
    parser.add_argument(
        '--file_output_prefix',
        type=str,
        default='',
        help='Output file prefix for saving evaluation results. '
             'Absolute path recommended. Files saved as {prefix}filename '
             '(default: empty string, saves to current directory)')
    parser.add_argument(
        '--mode',
        type=str,
        default='val',
        choices=['train', 'val'],
        help='Execution mode: train or val (default: val)'
    )
    parser.add_argument(
        '--problem_size',
        type=int,
        default=50,  # Customize this to your needs
        help='Problem size parameter'
    )
    # Parse arguments
    args = parser.parse_args()
    root_dir = args.root_dir
    file_output_prefix = args.file_output_prefix
    mode = args.mode
    problem_size = args.problem_size
    # Print parsed arguments for verification
    print(f"root_dir: {root_dir}")
    print(f"file_output_prefix: {file_output_prefix}")
    print(f"mode: {mode}")
    #print(f"problem_size: {problem_size}")
    
    # -----Run the evaluation-----
    # Run instances 20, 50, 100; execution time: 12s
    try:
        basepath = os.path.join(root_dir, "problems", problem , "dataset")
        # Initialize performance and feature variables
        metrics = {}
        # ---Train mode---
        if mode == 'train':
            # Load dataset
            dataset_path = os.path.join(basepath, f"train{problem_size}_dataset.npy")
            node_positions = np.load(dataset_path)
            n_instances = node_positions.shape[0]  # data shape: (n_instances, problem_size, 2)
            print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
            
            objs = []
            for i in range(n_instances):
                # Invoke evaluation function
                obj = eval_heuristic(node_positions[i])
                print(f"[*] Instance {i}: {obj}")
                objs.append(obj)
            
            print("[*] Average:")
            print(np.mean(objs))
            metrics[problem_size] = float(np.mean(objs))
        # ---Val mode---
        else:
            for problem_size in [20, 50, 100]:
                dataset_path = os.path.join(basepath, f"val{problem_size}_dataset.npy")
                print(f"[*] Evaluating {dataset_path}")
                node_positions = np.load(dataset_path)
                n_instances = node_positions.shape[0]
                objs = []
                for i in range(n_instances):
                    obj = eval_heuristic(node_positions[i])
                    objs.append(obj)
                print(f"[*] Average for {problem_size} cities: {np.mean(objs)}")
                metrics[problem_size] = float(np.mean(objs))
                
        if metrics:
            features = get_feature(metrics)
            score = get_score(metrics)
        else:
            features = None
            score = None

        # -----Print results to stdout (same for all problems)-----
        print('__SANDBOX_RESULT__')        
        print('__METRICS_START__')
        print(repr(metrics))
        print('__METRICS_END__')
        
        print('__FEATURES_START__')
        print(repr(features))
        print('__FEATURES_END__')
        
        print('__SCORE_START__')
        print(repr(score))
        print('__SCORE_END__')
        
        print('__SANDBOX_SUCCESS__')
        
    except Exception as e:
        print('__SANDBOX_ERROR__:')
        print(f'Error type: {type(e).__name__}')
        print(f'Error message: {str(e)}')
        print('Full traceback:')
        traceback.print_exc()