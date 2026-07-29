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
    
    feasible_post_caps = post_placement_caps[feasible_bins]
    feasible_scores = np.zeros_like(feasible_post_caps, dtype=float)
    
    # Estimate common item sizes based on the item being placed
    current_item = item
    estimated_common_sizes = [current_item, current_item * 0.9, current_item * 1.1, 
                              current_item * 0.5, current_item * 1.5, current_item * 0.75, current_item * 1.25]
    
    # Filter out non-positive sizes
    estimated_common_sizes = [size for size in estimated_common_sizes if size > 0]
    
    # Calculate how close the remaining capacity is to being a multiple of common sizes
    multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    for common_size in estimated_common_sizes:
        # Calculate distances to the nearest multiple of common_size
        multipliers = feasible_post_caps / common_size
        lower_mult = np.floor(multipliers)
        upper_mult = lower_mult + 1
        
        dist_to_lower = np.abs(feasible_post_caps - lower_mult * common_size)
        dist_to_upper = np.abs(feasible_post_caps - upper_mult * common_size)
        
        min_distances = np.minimum(dist_to_lower, dist_to_upper)
        
        # Add to the score (higher score for closer matches)
        multiple_fit_score += 1.0 / (1.0 + min_distances)
    
    # Add Best Fit component: prefer bins with less remaining space
    # This prevents overfilling bins unnecessarily
    best_fit_component = -feasible_post_caps  # Higher score for less remaining space
    
    # Combine both components
    # Use smaller weight for best fit component to balance between multiple-based scoring and best fit
    feasible_scores = multiple_fit_score + 0.07 * best_fit_component
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores