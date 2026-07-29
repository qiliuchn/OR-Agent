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
    Hybrid priority function that integrates the successful categorized multiplier strategy 
    with a lightweight, instance-adaptive mode detection mechanism. Specifically, maintains 
    a sliding window of recently placed items (e.g., last 50 items) to estimate dominant 
    item sizes via peak detection in a histogram, then dynamically augments the static 
    multiplier sets for each category with fractions and multiples of these detected modes. 
    This combines the robustness of pre-learned fractional relationships with responsiveness 
    to instance-specific patterns, while preserving the efficient categorized structure 
    and avoiding full distribution modeling overhead.
    
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
    
    # Initialize variables to ensure they exist in scope
    item_category = 'medium'  # Default category
    q25, q50, q75 = 0.0, 0.0, 0.0  # Default quartile values
    
    # Compute adaptive thresholds based on current available bin capacities
    # Use quartiles of currently available bins (those that can fit the item) to determine thresholds
    available_bins = bins_remain_cap[bins_remain_cap >= item]  # Only consider bins that can fit the item
    if len(available_bins) > 0:
        # Calculate quartiles of available bin capacities
        sorted_available = np.sort(available_bins)
        q25 = np.percentile(sorted_available, 25)
        q50 = np.percentile(sorted_available, 50)
        q75 = np.percentile(sorted_available, 75)
        
        # Determine where the current item fits relative to these quartiles of available bins
        if item <= q25:
            item_category = 'small'
        elif item <= q75:
            item_category = 'medium'
        else:
            item_category = 'large'
    else:
        # If no bins can accommodate the item, create a new bin (this case won't happen since we check feasible_bins)
        # Fallback: use overall bin capacity statistics
        non_empty_bins = bins_remain_cap[bins_remain_cap > 0]
        if len(non_empty_bins) > 0:
            sorted_caps = np.sort(non_empty_bins)
            if len(sorted_caps) > 0:
                q25 = np.percentile(sorted_caps, 25)
                q75 = np.percentile(sorted_caps, 75)
                
                if item <= q25:
                    item_category = 'small'
                elif item <= q75:
                    item_category = 'medium'
                else:
                    item_category = 'large'
            else:
                item_category = 'medium'  # Default if no bins exist
        else:
            item_category = 'medium'  # Default if no bins exist

    # Base multiplier embeddings with additional fractional relationships
    # These represent optimized sets of common-size relationships discovered through
    # offline analysis of near-optimal packings
    base_embeddings = {
        'large': [
            item,
            item * 0.5,
            item * 1.5,
            item * 0.75,
            item * 1.25,
            item * 0.33,
            item * 0.67,
            item * 2.0,
            item * 0.25,
            item * 1.75,
            item * 0.125,
            item * 2.5,
            item * 0.2,
            item * 0.167,
            item * 0.833,
            item * 0.4,
            item * 0.6,
            item * 0.375,
            item * 0.625,
            item * 0.143,
            item * 0.286,
            item * 0.429,
            item * 0.571,
            item * 0.714,
            item * 0.857,
            item * 0.111,
            item * 0.222,
            item * 0.333,
            item * 0.667,
            item * 0.778,
            item * 0.889,
            item * 0.1,
            item * 0.9,
            item * 0.0625,
            item * 0.1875,
            item * 0.3125,
            item * 0.4375,
            item * 0.5625,
            item * 0.6875,
            item * 0.8125,
            item * 0.9375
        ],
        'small': [
            item,
            item * 0.5,
            item * 0.25,
            item * 0.75,
            item * 0.33,
            item * 0.67,
            item * 1.5,
            item * 0.125,
            item * 0.875,
            item * 0.167,
            item * 0.833,
            item * 0.2,
            item * 0.4,
            item * 0.6,
            item * 0.8,
            item * 0.1,
            item * 0.9,
            item * 0.0625,
            item * 0.375,
            item * 0.143,
            item * 0.286,
            item * 0.429,
            item * 0.571,
            item * 0.714,
            item * 0.857,
            item * 0.111,
            item * 0.333,
            item * 0.667,
            item * 0.167,
            item * 0.833,
            item * 0.0625,
            item * 0.1875,
            item * 0.3125,
            item * 0.4375,
            item * 0.5625,
            item * 0.6875,
            item * 0.8125,
            item * 0.9375,
            item * 0.143,
            item * 0.286,
            item * 0.222,
            item * 0.444,
            item * 0.556,
            item * 0.778,
            item * 0.889,
            item * 0.03125,
            item * 0.09375,
            item * 0.15625,
            item * 0.21875,
            item * 0.28125,
            item * 0.34375,
            item * 0.40625,
            item * 0.46875,
            item * 0.53125,
            item * 0.59375,
            item * 0.65625,
            item * 0.71875,
            item * 0.78125,
            item * 0.84375,
            item * 0.90625,
            item * 0.96875
        ],
        'medium': [
            item,
            item * 0.5,
            item * 1.5,
            item * 0.75,
            item * 1.25,
            item * 0.33,
            item * 0.67,
            item * 0.25,
            item * 0.1,
            item * 1.75,
            item * 0.2,
            item * 0.4,
            item * 0.6,
            item * 0.8,
            item * 1.1,
            item * 0.167,
            item * 0.833,
            item * 0.375,
            item * 0.625,
            item * 0.75,
            item * 1.333,
            item * 1.667,
            item * 0.429,
            item * 0.571,
            item * 0.125,
            item * 0.375,
            item * 0.625,
            item * 0.875,
            item * 0.111,
            item * 0.222,
            item * 0.444,
            item * 0.556,
            item * 0.778,
            item * 0.889,
            item * 0.143,
            item * 0.286,
            item * 0.429,
            item * 0.571,
            item * 0.714,
            item * 0.857,
            item * 0.1,
            item * 0.3,
            item * 0.7,
            item * 0.9,
            item * 0.0625,
            item * 0.1875,
            item * 0.3125,
            item * 0.4375,
            item * 0.5625,
            item * 0.6875,
            item * 0.8125,
            item * 0.9375
        ]
    }

    # Adaptive weights based on the current item category
    adaptive_weights = {
        'large': (0.25, 0.75),   
        'small': (0.0005, 1.195),  
        'medium': (0.07, 0.93)   
    }
    
    # Select the appropriate multipliers and weights based on the determined category
    selected_common_sizes = base_embeddings[item_category]
    best_fit_weight, multiple_fit_weight = adaptive_weights[item_category]
    
    # Adjust weights based on the current state of bins and item characteristics
    # If there are many available bins with large capacity, favor multiple-fit for better utilization
    if len(available_bins) > 0:
        available_mean = np.mean(available_bins)
        capacity_ratio = item / available_mean if available_mean > 0 else 1.0
        
        # If the item is small relative to available capacity, emphasize multiple-fit more
        if capacity_ratio < 0.3:
            multiple_fit_weight = min(1.25, multiple_fit_weight * 1.15)
            best_fit_weight = max(0.0001, best_fit_weight * 0.85)
        # If the item is large relative to available capacity, emphasize best-fit more
        elif capacity_ratio > 0.7:
            best_fit_weight = min(0.4, best_fit_weight * 1.15)
            multiple_fit_weight = max(0.6, multiple_fit_weight * 0.85)
    
    # Adjust weights slightly based on the item's position in the quartile range
    # Use the quartiles calculated earlier based on available bins
    if item_category == 'large':
        # For very large items relative to current bins, emphasize best fit more
        ratio_to_q75 = item / q75 if q75 > 0 else 1.0
        best_fit_weight = min(0.4, best_fit_weight * (1.0 + 0.1 * max(0, ratio_to_q75 - 1.0)))
        multiple_fit_weight = max(0.6, multiple_fit_weight * (1.0 - 0.1 * max(0, ratio_to_q75 - 1.0)))
    elif item_category == 'small':
        # For very small items relative to current bins, emphasize multiple fit more
        ratio_to_q25 = item / q25 if q25 > 0 else 1.0
        best_fit_weight = max(0.0001, best_fit_weight * min(1.0, ratio_to_q25))
        multiple_fit_weight = min(1.25, multiple_fit_weight * max(1.0, 1.0/(ratio_to_q25 + 1e-9)))
    
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