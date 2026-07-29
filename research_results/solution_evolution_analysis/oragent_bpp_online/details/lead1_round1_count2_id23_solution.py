import os
import sys
import time
import random
import math
import json
import numpy as np
import pandas as pd
import scipy
import traci
import torch

import numpy as np
from typing import List

def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """
    Priority heuristic for online bin packing that balances Best Fit with diversity considerations.
    Optimized to avoid O(n^2) complexity issues that caused timeouts.
    
    The approach combines:
    - Best Fit (minimizing waste)
    - Diversity promotion via variance-based penalty
    - Efficient computation without nested loops
    
    Args:
        item: Size of the item to place
        bins_remain_cap: NumPy array of remaining bin capacities
        
    Returns:
        NumPy array of priority scores for each bin
    """
    # Calculate post-placement remaining capacities
    post_placement_caps = bins_remain_cap - item
    
    # Initialize scores
    scores = np.full_like(bins_remain_cap, -np.inf, dtype=float)
    
    # Only consider bins that can accommodate the item
    feasible_bins = bins_remain_cap >= item
    
    if not np.any(feasible_bins):
        return scores  # All bins remain with -inf scores
    
    # Work only with feasible bins
    feasible_post_caps = post_placement_caps[feasible_bins]
    n_feasible = len(feasible_post_caps)
    
    # 1. Best Fit component: prefer bins with less remaining space after placement
    best_fit_scores = -feasible_post_caps  # Higher score for less remaining space
    
    # 2. Diversity component: promote variety in remaining capacities
    # Instead of O(n^2) pairwise comparisons, use statistical measures
    diversity_scores = np.zeros(n_feasible)
    if n_feasible > 1:
        mean_cap = np.mean(feasible_post_caps)
        # Bins with capacities far from the mean get higher diversity scores
        # This encourages spreading out remaining capacities
        # Normalize to prevent dominance over other components
        std_cap = np.std(feasible_post_caps) if np.std(feasible_post_caps) > 0 else 1.0
        diversity_scores = np.abs(feasible_post_caps - mean_cap) / (std_cap + 1e-8)
    
    # 3. Multiple matching: favor useful leftover capacities
    # Find closest multiples of common sizes to the remaining capacities
    current_item = item
    # Consider common item sizes based on the current item
    common_sizes = [current_item, current_item * 0.9, current_item * 1.1, 
                    current_item * 0.5, current_item * 1.5, current_item * 0.75, current_item * 1.25]
    
    # Filter positive sizes
    common_sizes = [size for size in common_sizes if size > 0]
    
    multiple_matching_scores = np.zeros(n_feasible)
    if len(common_sizes) > 0 and n_feasible > 0:
        for common_size in common_sizes:
            # Calculate distances to all possible multiples of this common size
            max_mult = int(np.max(feasible_post_caps) // common_size) + 2
            if max_mult > 0:
                # Calculate distances to multiples: 0*common_size, 1*common_size, ..., max_mult*common_size
                distances_to_multiples = np.min([
                    np.abs(feasible_post_caps - n * common_size) 
                    for n in range(max_mult)
                ], axis=0)
                # Add contribution to scores (higher for closer matches)
                multiple_matching_scores += 1.0 / (1.0 + distances_to_multiples)
    
    # Combine all components with appropriate weights
    # Adjusted weights based on parent solution performance
    combined_scores = (
        0.8 * best_fit_scores +     # Best fit is important but not dominant
        0.2 * diversity_scores +    # Moderate diversity encouragement  
        0.7 * multiple_matching_scores  # Increased weight for multiple matching like parent
    )
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = combined_scores
    
    return scores