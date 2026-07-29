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
    
    Implementation idea: Develop a context-aware multiplier set that dynamically adjusts 
    the fractional/multiplicative relationships based on the magnitude of the current 
    item size (e.g., using finer granularity for small items <0.3 and coarser for large 
    items >0.7), while maintaining statelessness by deriving all multipliers from the 
    current item alone. This addresses the limitation of fixed multipliers across all 
    item scales and leverages the insight that packing strategies should differ for 
    small vs. large items. The approach combines:
    1. Dynamic estimation of common item sizes based on the current item and its scale
    2. Context-aware multiplier selection (finer for small items, coarser for large)
    3. Scoring bins based on how well their post-placement capacity matches these sizes
    4. Best Fit principle as a secondary criterion to avoid wasting space
    
    Args:
        item: Size of the item to place
        bins_remain_cap: NumPy array of remaining bin capacities
        
    Returns:
        NumPy array of priority scores for each bin
    """
    # Dynamically adjust the multiplier set based on item size
    current_item = item
    
    # Determine multiplier granularity based on item size
    # Adjusted thresholds based on typical item size distributions in bin packing problems
    if current_item < 0.3:  # Small items - use finest granularity
        estimated_common_sizes = [
            current_item, 
            current_item * 0.1,   # tenth size
            current_item * 0.125, # eighth size  
            current_item * 0.167, # sixth size
            current_item * 0.2,   # fifth size
            current_item * 0.25,  # quarter size
            current_item * 0.33,  # one third
            current_item * 0.4,   # two fifths
            current_item * 0.5,   # half size
            current_item * 0.6,   # three fifths
            current_item * 0.66,  # two thirds
            current_item * 0.75,  # three quarters
            current_item * 0.8,   # four fifths
            current_item * 0.9,   # 90% size
            current_item * 1.1,   # 110% size
            current_item * 1.2,   # six fifths
            current_item * 1.25,  # 1.25x size
            current_item * 1.33,  # four thirds
            current_item * 1.4,   # seven fifths
            current_item * 1.5,   # 1.5x size
            current_item * 1.6,   # eight fifths
            current_item * 1.8,   # nine fifths
            current_item * 2.0,   # double size
            current_item * 2.5,   # 2.5x size
            current_item * 3.0,   # triple size
        ]
    elif current_item < 0.7:  # Small to medium items - use moderate granularity
        estimated_common_sizes = [
            current_item, 
            current_item * 0.2,   # fifth size
            current_item * 0.25,  # quarter size
            current_item * 0.33,  # one third
            current_item * 0.4,   # two fifths
            current_item * 0.5,   # half size
            current_item * 0.6,   # three fifths
            current_item * 0.66,  # two thirds
            current_item * 0.75,  # three quarters
            current_item * 0.8,   # four fifths
            current_item * 0.9,   # 90% size
            current_item * 1.1,   # 110% size
            current_item * 1.2,   # six fifths
            current_item * 1.25,  # 1.25x size
            current_item * 1.33,  # four thirds
            current_item * 1.4,   # seven fifths
            current_item * 1.5,   # 1.5x size
            current_item * 1.6,   # eight fifths
            current_item * 1.8,   # nine fifths
            current_item * 2.0,   # double size
        ]
    else:  # Larger items - use coarser granularity
        estimated_common_sizes = [
            current_item, 
            current_item * 0.33,  # one third
            current_item * 0.5,   # half size
            current_item * 0.66,  # two thirds
            current_item * 0.75,  # three quarters
            current_item * 0.8,   # four fifths
            current_item * 0.9,   # 90% size
            current_item * 1.1,   # 110% size
            current_item * 1.2,   # six fifths
            current_item * 1.25,  # 1.25x size
            current_item * 1.33,  # four thirds
            current_item * 1.5,   # 1.5x size
            current_item * 2.0,   # double size
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
    
    # For feasible bins, calculate priority based on how close the remaining capacity is to common sizes
    feasible_post_caps = post_placement_caps[feasible_bins]
    feasible_scores = np.zeros_like(feasible_post_caps, dtype=float)
    
    # For each estimated common size, calculate how well the remaining capacity matches
    for common_size in estimated_common_sizes:
        # Calculate how close the remaining capacity is to being a multiple of the common size
        # Consider multiples: 0, 1x, 2x, 3x, etc. of the common size
        max_multiple = int(np.max(feasible_post_caps) // common_size) + 2
        # Cap the maximum number of multiples to prevent computational explosion
        max_considered_multiples = min(max_multiple, 10)  # Limit to first 10 multiples
        if max_considered_multiples > 0:
            distances_to_multiples = np.min([
                np.abs(feasible_post_caps - n * common_size) 
                for n in range(0, max_considered_multiples)
            ], axis=0)
            
            # Use a Gaussian-like function for scoring to give higher importance to very close matches
            # sigma controls the width of the function - smaller sigma gives sharper decay
            # Using a smaller sigma for tighter focus on very close matches
            sigma = max(common_size * 0.05, 0.01)  # 5% of the common size as the standard deviation (tighter focus)
            gaussian_scores = np.exp(-0.5 * (distances_to_multiples / sigma) ** 2)
            
            # Add to the score (higher score for closer matches to multiples)
            feasible_scores += gaussian_scores
    
    # Also add Best Fit component: prefer bins with less remaining space after placement
    # This prevents overfilling bins unnecessarily while still allowing for good fits
    best_fit_component = -post_placement_caps  # Higher score for less remaining space after placement
    
    # Combine both components with appropriate weights
    # Give more weight to the common size matching since it was the key insight from parent solution
    combined_scores = feasible_scores + 0.02 * best_fit_component  # Reduced weight to focus more on common size matching
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = combined_scores
    
    return scores