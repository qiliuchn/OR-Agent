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
    
    Implementation idea: Since maintaining explicit state may not be allowed in pure
    online algorithms, we simulate recency effects by leveraging the current bin
    configuration as implicit information about recent item patterns. Bins with very
    low remaining capacity suggest recent placement of small items, which may indicate
    a pattern of smaller upcoming items. We combine this with an adaptive estimation
    of common item sizes based on both the current item and inferred distribution
    from bin utilization patterns.
    
    The approach combines:
    1. Adaptive estimation of common item sizes based on current item and bin patterns
    2. Scoring bins based on how well their post-placement capacity matches these sizes
    3. Best Fit principle as a secondary criterion to avoid wasting space
    4. Implicit recency modeling through bin utilization analysis
    
    Args:
        item: Size of the item to place
        bins_remain_cap: NumPy array of remaining bin capacities
        
    Returns:
        NumPy array of priority scores for each bin
    """
    # Analyze current bin distribution to infer likely item patterns
    # Low remaining capacity bins may indicate recent small item placements
    sorted_remaining = np.sort(bins_remain_cap[bins_remain_cap > 0])
    
    # Estimate common item sizes based on current item and observed bin patterns
    current_item = item
    estimated_common_sizes = [current_item]  # Always include current item
    
    # Add fractions/multiples of current item
    current_multipliers = [0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0]
    for mult in current_multipliers:
        estimated_common_sizes.append(current_item * mult)
    
    # Infer additional common sizes from bin distribution if possible
    if len(sorted_remaining) > 1:
        # Look for common gaps/patterns in remaining capacities
        # These might represent frequently occurring item sizes
        for i in range(min(3, len(sorted_remaining))):
            cap = sorted_remaining[i]
            if cap > 0.01:  # Avoid very small values
                # Add this capacity as a potentially common size
                estimated_common_sizes.append(cap)
                # Add some variations around this observed capacity
                estimated_common_sizes.extend([cap * 0.8, cap * 1.2])
    
    # Remove duplicates and filter out non-positive sizes
    estimated_common_sizes = list(set([size for size in estimated_common_sizes if size > 0.01]))
    
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
        max_multiple = int(np.max(feasible_post_caps) // common_size) + 2 if common_size > 0 else 0
        if max_multiple > 0:
            distances_to_multiples = np.min([
                np.abs(feasible_post_caps - n * common_size) 
                for n in range(0, max_multiple)
            ], axis=0)
            
            # Add to the score (higher score for closer matches to multiples)
            # Use a function that gives significant scores only for close matches
            feasible_scores += 1.0 / (1.0 + distances_to_multiples)
    
    # Add Best Fit component: prefer bins with less remaining space after placement
    # This prevents overfilling bins unnecessarily while still allowing for good fits
    best_fit_component = -post_placement_caps  # Higher score for less remaining space after placement
    
    # Combine both components with appropriate weights
    # Give more weight to the common size matching since it was the key insight from parent solution
    combined_scores = feasible_scores + 0.01 * best_fit_component  # Further reduced weight to emphasize common size matching
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = combined_scores
    
    return scores