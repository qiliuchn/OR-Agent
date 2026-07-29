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
    result in a remaining capacity that is close to common item sizes.
    This implementation estimates common item sizes based on the current item and 
    prioritizes bins that leave "useful" remaining capacities.
    
    The approach combines:
    1. Estimation of common item sizes based on the current item
    2. Scoring bins based on how well their post-placement capacity matches these sizes
    3. Best Fit principle as a secondary criterion to avoid wasting space
    
    Args:
        item: Size of the item to place
        bins_remain_cap: NumPy array of remaining bin capacities
        
    Returns:
        NumPy array of priority scores for each bin
    """
    # Estimate common item sizes based on the current item
    # Consider the current item size and its common fractions/multiples
    current_item = item
    estimated_common_sizes = [
        current_item, 
        current_item * 0.5,   # half size
        current_item * 0.75,  # three quarters
        current_item * 0.9,   # 90% size
        current_item * 1.1,   # 110% size
        current_item * 1.25,  # 1.25x size
        current_item * 1.5,   # 1.5x size
        current_item * 0.25,  # quarter size
        current_item * 2.0    # double size
    ]
    
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
        # Consider multiples: 0, 1x, 2x, 3x, etc. of the common size
        max_multiple = int(np.max(feasible_post_caps) // common_size) + 2
        distances_to_multiples = np.min([
            np.abs(feasible_post_caps - n * common_size) 
            for n in range(0, max_multiple)
        ], axis=0)
        
        # Add to the score (higher score for closer matches to multiples)
        # Use a function that gives significant scores only for close matches
        feasible_scores += 1.0 / (1.0 + distances_to_multiples)
    
    # Also add Best Fit component: prefer bins with less remaining space after placement
    # This prevents overfilling bins unnecessarily while still allowing for good fits
    best_fit_component = -post_placement_caps  # Higher score for less remaining space after placement
    
    # Combine both components with appropriate weights
    # Give more weight to the common size matching since it was the key insight from parent solution
    combined_scores = feasible_scores + 0.1 * best_fit_component
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = combined_scores
    
    return scores