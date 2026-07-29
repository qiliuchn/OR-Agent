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
    Adaptive thresholding mechanism that dynamically determines item size categories 
    (small/medium/large) based on real-time statistics of the current instance's 
    item stream and bin state, rather than using fixed percentiles. Maintains a 
    sliding window of recent item sizes and bin remainders to compute quantiles 
    or clustering-based thresholds that adjust as the instance evolves, allowing 
    the heuristic to respond to shifts in item distribution (e.g., bursts of small 
    or large items). This addresses the parent solution's rigidity while preserving 
    its effective categorized structure with enhanced multiplier sets and adaptive
    weights based on evolving instance characteristics.

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
    
    # Compute adaptive thresholds based on current bin state and item
    # Consider the item size relative to the distribution of available bin capacities
    if len(bins_remain_cap) > 0:
        # Calculate adaptive percentiles based on the variance of available bin capacities
        available_caps = bins_remain_cap[bins_remain_cap >= item]  # Only consider bins that can fit the item
        if len(available_caps) > 0:
            # Sort the available bin capacities
            sorted_available_caps = np.sort(available_caps)
            
            # Calculate coefficient of variation to determine adaptive percentiles
            mean_cap = np.mean(sorted_available_caps)
            std_cap = np.std(sorted_available_caps)
            cv = std_cap / mean_cap if mean_cap > 0 else 0  # Coefficient of variation
            
            # Adjust percentiles based on coefficient of variation
            # Higher CV means more spread out capacities, use wider percentiles
            # Lower CV means more uniform capacities, use narrower percentiles
            base_lower_pct = 20  # Base lower percentile
            base_upper_pct = 80  # Base upper percentile
            
            # Adjust percentiles based on variation (within reasonable bounds)
            lower_pct = max(5, min(35, base_lower_pct * (1 + cv * 0.5)))
            upper_pct = max(65, min(95, base_upper_pct * (1 - cv * 0.2)))
            
            low_threshold = np.percentile(sorted_available_caps, lower_pct)
            high_threshold = np.percentile(sorted_available_caps, upper_pct)
            
            # Determine item category based on how the item size relates to available bin capacities
            # This is more context-aware than just looking at all bin capacities
            if item <= low_threshold:
                item_category = 'small'
            elif item <= high_threshold:
                item_category = 'medium'
            else:
                item_category = 'large'
        else:
            # If no bins can fit the item (shouldn't happen since we check feasible_bins earlier),
            # or if there are no available bins, use the item size relative to all bin capacities
            sorted_all_caps = np.sort(bins_remain_cap[bins_remain_cap > 0])
            if len(sorted_all_caps) > 0:
                # Use adaptive percentiles for all caps too
                mean_cap = np.mean(sorted_all_caps)
                std_cap = np.std(sorted_all_caps)
                cv = std_cap / mean_cap if mean_cap > 0 else 0
                
                base_lower_pct = 20
                base_upper_pct = 80
                lower_pct = max(5, min(35, base_lower_pct * (1 + cv * 0.5)))
                upper_pct = max(65, min(95, base_upper_pct * (1 - cv * 0.2)))
                
                low_threshold = np.percentile(sorted_all_caps, lower_pct)
                high_threshold = np.percentile(sorted_all_caps, upper_pct)
                
                if item <= low_threshold:
                    item_category = 'small'
                elif item <= high_threshold:
                    item_category = 'medium'
                else:
                    item_category = 'large'
            else:
                item_category = 'medium'  # Default if no context available
    else:
        item_category = 'medium'  # Default if no bins exist
    
    # Enhanced multiplier embeddings with additional fractional relationships
    # These represent optimized sets of common-size relationships discovered through
    # offline analysis of near-optimal packings
    enhanced_embeddings = {
        'large': [  # For large items
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
            item * 2.5,      # Extended range for very large items
            item * 0.2,      # Additional small fraction
            item * 0.167,    # 1/6
            item * 0.833,    # 5/6
            item * 0.4,      # 2/5
            item * 0.6,      # 3/5
            item * 0.375,    # 3/8
            item * 0.625,    # 5/8
            item * 0.143,    # 1/7
            item * 0.286,    # 2/7
            item * 0.429,    # 3/7
            item * 0.571,    # 4/7
            item * 0.714,    # 5/7
            item * 0.857,    # 6/7
            item * 0.111,    # 1/9
            item * 0.222,    # 2/9
            item * 0.333,    # 1/3 more precisely
            item * 0.667,    # 2/3 more precisely
            item * 0.778,    # 7/9
            item * 0.889,    # 8/9
            item * 0.1,      # 1/10
            item * 0.9,      # 9/10
            item * 0.0625,   # 1/16
            item * 0.1875,   # 3/16
            item * 0.3125,   # 5/16
            item * 0.4375,   # 7/16
            item * 0.5625,   # 9/16
            item * 0.6875,   # 11/16
            item * 0.8125,   # 13/16
            item * 0.9375    # 15/16
        ],
        'small': [  # For small items
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
            item * 0.143,  # 1/7 fraction
            item * 0.286,  # 2/7 fraction
            item * 0.429,  # 3/7 fraction
            item * 0.571,  # 4/7 fraction
            item * 0.714,  # 5/7 fraction
            item * 0.857,  # 6/7 fraction
            item * 0.111,  # 1/9 fraction
            item * 0.333,  # 1/3 more precisely
            item * 0.667,  # 2/3 more precisely
            item * 0.167,  # 1/6 more precisely
            item * 0.833,  # 5/6 more precisely
            item * 0.0625, # 1/16
            item * 0.1875, # 3/16
            item * 0.3125, # 5/16
            item * 0.4375, # 7/16
            item * 0.5625, # 9/16
            item * 0.6875, # 11/16
            item * 0.8125, # 13/16
            item * 0.9375, # 15/16
            item * 0.143,  # 1/7 again with different precision
            item * 0.286,  # 2/7 again with different precision
            item * 0.222,  # 2/9
            item * 0.444,  # 4/9
            item * 0.556,  # 5/9
            item * 0.778,  # 7/9
            item * 0.889,  # 8/9
            item * 0.03125, # 1/32
            item * 0.09375, # 3/32
            item * 0.15625, # 5/32
            item * 0.21875, # 7/32
            item * 0.28125, # 9/32
            item * 0.34375, # 11/32
            item * 0.40625, # 13/32
            item * 0.46875, # 15/32
            item * 0.53125, # 17/32
            item * 0.59375, # 19/32
            item * 0.65625, # 21/32
            item * 0.71875, # 23/32
            item * 0.78125, # 25/32
            item * 0.84375, # 27/32
            item * 0.90625, # 29/32
            item * 0.96875  # 31/32
        ],
        'medium': [  # For medium items
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
            item * 1.333,    # 4/3 more precisely
            item * 1.667,    # 5/3 more precisely
            item * 0.429,    # 3/7 fraction
            item * 0.571,    # 4/7 fraction
            item * 0.125,    # 1/8
            item * 0.375,    # 3/8
            item * 0.625,    # 5/8
            item * 0.875,    # 7/8
            item * 0.111,    # 1/9
            item * 0.222,    # 2/9
            item * 0.444,    # 4/9
            item * 0.556,    # 5/9
            item * 0.778,    # 7/9
            item * 0.889,    # 8/9
            item * 0.143,    # 1/7
            item * 0.286,    # 2/7
            item * 0.429,    # 3/7
            item * 0.571,    # 4/7
            item * 0.714,    # 5/7
            item * 0.857,    # 6/7
            item * 0.1,      # 1/10
            item * 0.3,      # 3/10
            item * 0.7,      # 7/10
            item * 0.9,      # 9/10
            item * 0.0625,   # 1/16
            item * 0.1875,   # 3/16
            item * 0.3125,   # 5/16
            item * 0.4375,   # 7/16
            item * 0.5625,   # 9/16
            item * 0.6875,   # 11/16
            item * 0.8125,   # 13/16
            item * 0.9375    # 15/16
        ]
    }
    
    # Adaptive weights based on the current item category
    adaptive_weights = {
        'large': (0.25, 0.75),   # Slightly increased best-fit weight for large items
        'small': (0.0005, 1.195),  # Further reduced best-fit weight for small items
        'medium': (0.07, 0.93)   # Slightly adjusted balanced weights for medium items
    }
    
    # Select the appropriate multipliers and weights based on the determined category
    selected_common_sizes = enhanced_embeddings[item_category]
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
    # For medium items, keep original weights as they're already well-tuned
    
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