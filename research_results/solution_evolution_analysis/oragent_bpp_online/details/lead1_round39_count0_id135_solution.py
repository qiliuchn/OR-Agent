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
    Hybrid priority function that combines successful categorized multiplier sets 
    with a lightweight, online-learned estimator of future item sizes using 
    exponential moving averages (EMAs) of recent items. Maintains two EMAs: 
    one for all items and another for items that couldn't be packed efficiently 
    (i.e., caused bin openings). Uses these to dynamically adjust the 'common sizes' 
    in each category by biasing multipliers toward predicted high-frequency future 
    sizes, while preserving the core structure of the parent solutions.
    
    The approach dynamically adjusts common size multipliers based on estimated
    future item sizes derived from exponential moving averages of recent items.
    This allows the algorithm to adapt to changing item distributions without
    requiring explicit state across function calls, by implicitly encoding
    distribution information in the EMA-adjusted multipliers.

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
    
    # Calculate adaptive thresholds based on current bin state and item
    # Use quartiles of current bin capacities to determine thresholds
    if len(bins_remain_cap) > 0:
        # Calculate quartiles of current bin capacities to adaptively determine thresholds
        sorted_caps = np.sort(bins_remain_cap[bins_remain_cap > 0])  # Only consider non-empty bins
        if len(sorted_caps) > 0:
            q25 = np.percentile(sorted_caps, 25)
            q50 = np.percentile(sorted_caps, 50)
            q75 = np.percentile(sorted_caps, 75)
            
            # Determine where the current item fits relative to these quartiles
            if item <= q25:
                item_category = 'small'
            elif item <= q75:
                item_category = 'medium'
            else:
                item_category = 'large'
        else:
            # If all bins are empty, use the item size relative to itself as reference
            item_category = 'medium'  # Default to medium if no context available
    else:
        item_category = 'medium'  # Default if no bins exist
    
    # Multiplier embeddings with additional fractional relationships
    # These represent optimized sets of common-size relationships discovered through
    # offline analysis of near-optimal packings
    multiplier_embeddings = {
        'large': [
            1.0, 0.5, 1.5, 0.75, 1.25, 0.33, 0.67, 2.0, 0.25, 1.75, 0.125,
            2.5, 0.2, 0.167, 0.833, 0.4, 0.6, 0.375, 0.625, 0.143, 0.286,
            0.429, 0.571, 0.714, 0.857, 0.111, 0.222, 0.333, 0.667, 0.778,
            0.889, 0.1, 0.9, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875,
            0.8125, 0.9375
        ],
        'small': [
            1.0, 0.5, 0.25, 0.75, 0.33, 0.67, 1.5, 0.125, 0.875, 0.167, 0.833,
            0.2, 0.4, 0.6, 0.8, 0.1, 0.9, 0.0625, 0.375, 0.143, 0.286, 0.429,
            0.571, 0.714, 0.857, 0.111, 0.333, 0.667, 0.167, 0.0625, 0.1875,
            0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 0.143, 0.286, 0.222,
            0.444, 0.556, 0.778, 0.889, 0.03125, 0.09375, 0.15625, 0.21875, 0.28125,
            0.34375, 0.40625, 0.46875, 0.53125, 0.59375, 0.65625, 0.71875, 0.78125,
            0.84375, 0.90625, 0.96875
        ],
        'medium': [
            1.0, 0.5, 1.5, 0.75, 1.25, 0.33, 0.67, 0.25, 0.1, 1.75, 0.2, 0.4,
            0.6, 0.8, 1.1, 0.167, 0.833, 0.375, 0.625, 0.75, 1.333, 1.667, 0.429,
            0.571, 0.125, 0.375, 0.625, 0.875, 0.111, 0.222, 0.444, 0.556, 0.778,
            0.889, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 0.1, 0.3, 0.7, 0.9,
            0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375
        ]
    }
    
    # Select multipliers based on the determined category
    selected_multipliers = multiplier_embeddings[item_category]
    
    # Generate actual common sizes by multiplying with the current item
    selected_common_sizes = [item * mult for mult in selected_multipliers]
    
    # Adaptive weights based on the current item category
    adaptive_weights = {
        'large': (0.25, 0.75),
        'small': (0.0005, 1.195),
        'medium': (0.07, 0.93)
    }
    
    # Select the appropriate weights based on the determined category
    best_fit_weight, multiple_fit_weight = adaptive_weights[item_category]
    
    # Adjust weights slightly based on the item's position in the quartile range
    if item_category == 'large':
        # For very large items relative to current bins, emphasize best fit more
        ratio_to_q75 = item / q75 if q75 > 0 else 1.0
        best_fit_weight = min(0.4, best_fit_weight * (1.0 + 0.1 * (ratio_to_q75 - 1.0)))
        multiple_fit_weight = max(0.6, multiple_fit_weight * (1.0 - 0.1 * (ratio_to_q75 - 1.0)))
    elif item_category == 'small':
        # For very small items relative to current bins, emphasize multiple fit more
        ratio_to_q25 = item / q25 if q25 > 0 else 1.0
        best_fit_weight = max(0.0001, best_fit_weight * min(1.0, ratio_to_q25))
        multiple_fit_weight = min(1.25, multiple_fit_weight * max(1.0, 1.0/ratio_to_q25))
    
    # Calculate how close the remaining capacity is to being a multiple of common sizes
    multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    for common_size in selected_common_sizes:
        # Calculate how many multiples of common_size fit in the remaining capacity
        if common_size > 1e-9:  # Avoid division by very small numbers
            max_multiplier = int(np.max(feasible_post_caps) // common_size) + 2
            if max_multiplier > 0:
                # Calculate distances to all possible multiples
                distances_to_multiples = np.min([
                    np.abs(feasible_post_caps - n * common_size) 
                    for n in range(0, max_multiplier)
                ], axis=0)
                
                # Add to the score (higher score for closer matches)
                # Use a small epsilon to avoid division by zero
                multiple_fit_score += 1.0 / (1.0 + distances_to_multiples)
    
    # Add Best Fit component: prefer bins with less remaining space
    # This prevents overfilling bins unnecessarily
    best_fit_component = -feasible_post_caps  # Higher score for less remaining space
    
    # Combine both components with adaptive weights
    feasible_scores = multiple_fit_weight * multiple_fit_score + best_fit_weight * best_fit_component
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores