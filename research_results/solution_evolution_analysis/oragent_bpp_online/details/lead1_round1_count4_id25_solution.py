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

def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """
    Priority heuristic that combines best-fit principle with multiple-based scoring.
    Based on insights from parent solutions, this version focuses on effectiveness
    while working within the constraints of a stateless function.
    
    The approach combines:
    1. Best Fit principle (prioritizing bins with less remaining space after placement)
    2. Multiple-based scoring (favoring bins that leave capacities near common sizes)
    
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
    
    # For feasible bins, calculate priority based on how close the remaining capacity is to multiples of common sizes
    feasible_post_caps = post_placement_caps[feasible_bins]
    feasible_scores = np.zeros_like(feasible_post_caps, dtype=float)
    
    # Estimate common item sizes based on the current item
    # Consider the current item and some fractions/multiples of it
    current_item = item
    estimated_common_sizes = [
        current_item, 
        current_item * 0.5,   # half
        current_item * 0.75,  # three quarters
        current_item * 0.9,   # 90%
        current_item * 1.1,   # 110%
        current_item * 1.25,  # 1.25 times
        current_item * 1.5,   # 1.5 times
        current_item * 2.0    # double
    ]
    
    # Filter out non-positive sizes
    estimated_common_sizes = [size for size in estimated_common_sizes if size > 0]
    
    # For each estimated common size, calculate how well the remaining capacity matches
    for common_size in estimated_common_sizes:
        # Calculate how close the remaining capacity is to being a multiple of the common size
        max_multiple = max(1, int(np.max(feasible_post_caps) // common_size) + 2)
        
        # Compute distances to nearest multiples efficiently
        distances_to_multiples = np.full_like(feasible_post_caps, np.inf)
        
        for n in range(0, max_multiple):
            candidate_distances = np.abs(feasible_post_caps - n * common_size)
            distances_to_multiples = np.minimum(distances_to_multiples, candidate_distances)
        
        # Add to the score (higher score for closer matches)
        # Use a sigmoid-like function to make scores meaningful
        feasible_scores += 1.0 / (1.0 + distances_to_multiples)
    
    # Also add Best Fit component: prefer bins with less remaining space
    # This prevents overfilling bins unnecessarily
    best_fit_component = -post_placement_caps  # Higher score for less remaining space
    
    # Combine both components with better balance (using parent solution's weight)
    combined_scores = feasible_scores + 0.1 * best_fit_component  # Use parent solution's weight for best fit
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = combined_scores
    
    return scores