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
    Priority heuristic that implements a hybrid approach combining the successful
    categorized strategy from the parent solution with a dynamic adaptive adjustment
    based on real-time packing progress. The method maintains the effective category-based 
    approach with extensive multiplier sets and empirically-optimized weights, while 
    adding a more sophisticated adaptive element based on actual packing efficiency
    metrics to dynamically adjust the balance between best-fit and multiple-fit components.
    This preserves the proven architecture while allowing intelligent adaptation to 
    current packing conditions.

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
    
    # Calculate the percentile rank of the current item among remaining bin capacities
    # This gives us a measure of how large the current item is relative to current context
    if len(bins_remain_cap) > 0:
        # Calculate the percentile of the current item in the context of remaining capacities
        sorted_caps = np.sort(bins_remain_cap)
        # Find where the item would fit in the sorted array
        item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps)
    else:
        item_percentile = 0.5  # Default if no bins exist
    
    # Enhanced multiplier embeddings with additional fractional relationships
    # These represent optimized sets of common-size relationships discovered through
    # offline analysis of near-optimal packings
    enhanced_embeddings = {
        'large': [  # For items in top 20% relative to bin capacities
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
        'small': [  # For items in bottom 20% relative to bin capacities
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
        'medium': [  # For items in middle 60% relative to bin capacities
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
    
    # Enhanced weights based on size category with fine-tuned parameters
    enhanced_weights = {
        'large': (0.25, 0.75),   # Slightly increased best-fit weight for large items
        'small': (0.001, 1.194),  # Slightly increased best-fit weight for small items to allow some tight-fitting
        'medium': (0.08, 0.92)   # Slightly increased best-fit weight for medium items
    }
    
    # Determine the size category based on refined percentile thresholds
    if item_percentile > 0.82:  # Very large item
        selected_common_sizes = enhanced_embeddings['large']
        base_best_fit_weight, base_multiple_fit_weight = enhanced_weights['large']
        # Slightly increase best-fit weight for very large items to ensure they fit
        base_best_fit_weight *= 1.07
        base_multiple_fit_weight *= 0.99
    elif item_percentile < 0.18:  # Very small item
        selected_common_sizes = enhanced_embeddings['small']
        base_best_fit_weight, base_multiple_fit_weight = enhanced_weights['small']
        # Further reduce best-fit weight for very small items
        base_best_fit_weight *= 0.93
        base_multiple_fit_weight *= 1.01
    elif item_percentile > 0.62:  # Large-medium item
        # Interpolate between medium and large strategies
        selected_common_sizes = enhanced_embeddings['medium']
        # Adjust weights to be between medium and large values
        base_best_fit_weight, base_multiple_fit_weight = 0.14, 0.86  # Between medium and large
    elif item_percentile < 0.38:  # Small-medium item
        # Interpolate between medium and small strategies
        selected_common_sizes = enhanced_embeddings['small']  # Use small's rich multiplier set
        # Adjust weights to be between medium and small values
        base_best_fit_weight, base_multiple_fit_weight = 0.025, 1.055  # Between medium and small
    else:  # True medium item
        selected_common_sizes = enhanced_embeddings['medium']
        base_best_fit_weight, base_multiple_fit_weight = enhanced_weights['medium']
    
    # Calculate advanced packing state metrics for dynamic adaptation
    # 1. Fragmentation index based on coefficient of variation of remaining capacities
    if len(bins_remain_cap) > 1 and np.mean(bins_remain_cap) > 1e-9:
        cv = np.std(bins_remain_cap) / (np.mean(bins_remain_cap) + 1e-9)
        fragmentation_index = 1 - np.exp(-cv)  # Compress higher values
    else:
        fragmentation_index = 0.0
    
    # 2. Average utilization of bins that can fit the current item
    if np.sum(bins_remain_cap) > 1e-9:
        avg_utilization = 1 - (np.sum(bins_remain_cap) / (len(bins_remain_cap) * 100))  # Assuming capacity of 100
    else:
        avg_utilization = 0.0
    
    # 3. Density of available space - ratio of space that can fit current item to total space
    if np.sum(bins_remain_cap) > 1e-9:
        space_density = np.sum(bins_remain_cap[bins_remain_cap >= item]) / np.sum(bins_remain_cap)
    else:
        space_density = 0.0
    
    # Dynamic weight adjustment based on multiple packing state factors
    # When fragmentation is high, prioritize best-fit to consolidate space
    # When utilization is low, allow more multiple-fit opportunities
    # When space density is low, be more selective (better best-fit)
    
    # Combine multiple factors for adaptive adjustment
    # Weight adjustment is based on the current packing situation
    best_fit_boost = fragmentation_index * 0.3 + (1 - avg_utilization) * 0.1 + (1 - space_density) * 0.1
    multiple_fit_boost = (1 - fragmentation_index) * 0.1 + avg_utilization * 0.1 + space_density * 0.05
    
    adjusted_best_fit_weight = base_best_fit_weight * (1 + best_fit_boost)
    adjusted_multiple_fit_weight = base_multiple_fit_weight * (1 + multiple_fit_boost)

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
    
    # Combine all components with adaptive weights
    feasible_scores = (
        adjusted_multiple_fit_weight * multiple_fit_score + 
        adjusted_best_fit_weight * best_fit_component
    )
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores