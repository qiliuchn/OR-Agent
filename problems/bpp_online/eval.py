# Evaluation script for online binpacking problem.
import os 
import sys
import traceback
import numpy as np
import pickle
import argparse
from typing import Dict, Tuple, List, Any
import seed_solution as solution_module  # Note: solution module script is generated and saved on the fly


# =====Load function to evolve=====
problem = "bpp_online"
priority = getattr(solution_module, "priority")  # Get function to evolve


# =====Binpacking functions=====
def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Returns indices of bins that have sufficient capacity for a given item.

    Args:
        item: Size of the item to place (float)
        bins: NumPy array of remaining bin capacities (float array)

    Returns:
        NumPy array of indices where bins have capacity >= item size
    """
    return np.nonzero((bins - item) >= 0)[0]

def online_binpack(items: tuple[float], bins: np.ndarray) -> tuple[list[list[float]], np.ndarray]:
    """
    Performs online bin-packing of items into bins using a priority heuristic.

    Args:
        items: Tuple of item sizes to pack (float values)
        bins: NumPy array of initial bin capacities (float array)

    Returns:
        Tuple of (packing, remaining_capacities):
        - packing: List of lists, where each inner list contains items in a bin
        - remaining_capacities: Updated bin capacities after packing
    """
    # Track which items are added to each bin.
    packing = [[] for _ in bins]
    # Add items to bins.
    for item in items:
        # Extract bins that have sufficient space to fit item.
        valid_bin_indices = get_valid_bin_indices(item, bins)
        # Score each bin based on heuristic.
        priorities = priority(item, bins[valid_bin_indices])
        # Add item to bin with highest priority.
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    # Remove unused bins from packing.
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins


# ======Evaluation function=====
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

def evaluate(instances: dict) -> float:
    """Evaluate heuristic function on a set of online binpacking instances."""
    # List storing number of bins used for each instance.
    num_bins = []
    metrics = {}
    # Perform online binpacking for each instance.
    for name in instances:
        if name == 'l1_bound':  # Skip l1_bound; l1_bound is a float that represents the L1 lower bound (best performance) for benchmarking
            continue
        instance = instances[name]
        capacity = instance['capacity']  # Initial capacity of each bin; note: each bin has the same capacity
        items = instance['items']  # Items to pack
        items = np.array(items) if isinstance(items, list) else items  # Convert to NumPy array
        # Create num_items bins so there will always be space for all items,
        # regardless of packing order. Array has shape (num_items,).
        bins = np.array([capacity for _ in range(instance['num_items'])])
        # Pack items into bins and return remaining capacity in bins_packed, which
        # has shape (num_items,).
        _, bins_packed = online_binpack(items.astype(float), bins)
        # If remaining capacity in a bin is equal to initial capacity, then it is unused. Count number of used bins.
        num_bins.append((bins_packed != capacity).sum())
        metrics[name] = float(num_bins[-1])
    # return negative of average number of bins used across instances (as we want to minimize number of bins).
    return np.mean(num_bins), metrics


# =======Main function=====
if __name__ == "__main__":
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
        default=100,  # Customize this to your needs
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
    # Execution time: 7s
    try:
        basepath = os.path.join(root_dir, "problems", problem)
        file_name = f"weibull_5k_{mode}.pickle"  # it contains multiple instances; each instance has 5000 items; bin capacity is 100
        dataset_path = os.path.join(basepath, "dataset", file_name)

        dataset = pickle.load(open(dataset_path, 'rb'))
        
        # Evaluate heuristic function on dataset
        avg_num_bins, metrics = evaluate(dataset)
        l1_bound = dataset['l1_bound']
        excess = (avg_num_bins - l1_bound) / l1_bound
        print(file_name)
        print(f'\t Average number of bins: {avg_num_bins}')
        print(f'\t Lower bound on optimum: {l1_bound}')
        print(f'\t Excess: {100 * excess:.2f}%')
        
        print("[*] Average:")
        print(excess * 100)
                
        if metrics:
            features = get_feature(metrics)
            score = avg_num_bins
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