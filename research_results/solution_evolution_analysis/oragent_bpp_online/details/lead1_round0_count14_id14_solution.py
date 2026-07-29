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
    Priority heuristic that assigns higher priority to bins that, after placement, 
    result in a remaining capacity that is close to multiples of common item sizes.
    This implementation estimates common item sizes based on recent items and 
    prioritizes bins that leave "useful" remaining capacities.
    
    The approach combines:
    1. Best Fit principle (prioritizing bins with less remaining space)
    2. Multiple-based scoring (favoring bins that leave capacities near common sizes)
    3. Adaptive estimation of common item sizes
    
    Args:
        item: Size of the item to place
        bins_remain_cap: NumPy array of remaining bin capacities
        
    Returns:
        NumPy array of priority scores for each bin
    """
    # Estimate common item sizes based on the item being placed
    # We'll consider the current item size as a reference for what might be common
    # Also consider some fractions/multiples of it
    current_item = item
    estimated_common_sizes = [current_item, current_item * 0.9, current_item * 1.1, 
                              current_item * 0.5, current_item * 1.5, current_item * 0.75, current_item * 1.25]
    
    # Filter out non-positive sizes
    estimated_common_sizes = [size for size in estimated_common_sizes if size > 0]
    
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
    
    # For each estimated common size, calculate how well the remaining capacity matches
    for common_size in estimated_common_sizes:
        # Calculate how close the remaining capacity is to being a multiple of the common size
        # Use 1/(1+distance) to create a score that's high when distance is low
        distances_to_multiples = np.min([
            np.abs(feasible_post_caps - n * common_size) 
            for n in range(0, int(np.max(feasible_post_caps) // common_size) + 2)
        ], axis=0)
        
        # Add to the score (higher score for closer matches)
        # Use a sigmoid-like function to make scores meaningful
        feasible_scores += 1.0 / (1.0 + distances_to_multiples)
    
    # Also add Best Fit component: prefer bins with less remaining space
    # This prevents overfilling bins unnecessarily
    best_fit_component = -post_placement_caps  # Higher score for less remaining space
    
    # Combine both components
    combined_scores = feasible_scores + 0.1 * best_fit_component  # Weight the best fit less heavily
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = combined_scores
    
    return scores