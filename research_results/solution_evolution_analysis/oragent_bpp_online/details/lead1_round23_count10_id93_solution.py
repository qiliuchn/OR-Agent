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
    Priority heuristic that integrates a dead-zone fragmentation penalty into an
    adaptive thresholding framework. This combines the distribution-aware item
    categorization from Node 84 with the fragmentation-aware placement bias from
    Node 85, creating a hybrid heuristic that both adapts to current bin state
    dispersion and actively discourages mid-sized gaps. The approach modifies
    the priority score to include a smooth negative penalty for post-placement
    remainders in [0.2, 0.7] of bin capacity, potentially overcoming the local
    optimum observed in both parent approaches when used separately.
    
    Implementation idea: Integrate a dead-zone fragmentation penalty directly into
    the adaptive thresholding framework of Node 84 by modifying the priority score
    to include a smooth negative penalty for post-placement remainders in [0.2, 0.7]
    of bin capacity. This combines the distribution-aware item categorization from
    Node 84 with the fragmentation-aware placement bias from Node 85, creating a
    hybrid heuristic that both adapts to current bin state dispersion and actively
    discourages mid-sized gaps—potentially overcoming the local optimum observed
    in both parent approaches when used separately.

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
    
    # Compute adaptive thresholds based on current distribution of remaining capacities
    if len(bins_remain_cap) > 0:
        # Calculate quartiles to determine IQR
        q1 = np.percentile(bins_remain_cap, 25)
        q3 = np.percentile(bins_remain_cap, 75)
        iqr = q3 - q1
        median_val = np.median(bins_remain_cap)
        
        # Define adaptive thresholds based on IQR
        # Large items: those larger than median + 0.5*IQR
        # Small items: those smaller than median - 0.5*IQR
        large_threshold = median_val + 0.5 * iqr
        small_threshold = median_val - 0.5 * iqr
        
        # Calculate normalized position of the current item relative to the distribution
        if iqr > 1e-9:  # Avoid division by zero
            item_position = (item - median_val) / iqr
        else:
            item_position = 0.0
            
    else:
        # Default thresholds if no bins exist
        large_threshold = 0.5
        small_threshold = 0.5
        item_position = 0.0
    
    # Enhanced multiplier embeddings with additional fractional relationships
    # These represent optimized sets of common-size relationships discovered through
    # offline analysis of near-optimal packings
    enhanced_embeddings = {
        'large': [  # For items significantly larger than median
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
        'small': [  # For items significantly smaller than median
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
        'medium': [  # For items around the median
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
        'small': (0.0005, 1.195),  # Further reduced best-fit weight for small items
        'medium': (0.07, 0.93)   # Slightly adjusted balanced weights for medium items
    }
    
    # Determine the size category based on adaptive thresholds
    if item > large_threshold:  # Large item based on distribution
        selected_common_sizes = enhanced_embeddings['large']
        best_fit_weight, multiple_fit_weight = enhanced_weights['large']
        # Slightly increase best-fit weight for large items to ensure they fit
        best_fit_weight *= 1.07
        multiple_fit_weight *= 0.99
    elif item < small_threshold:  # Small item based on distribution
        selected_common_sizes = enhanced_embeddings['small']
        best_fit_weight, multiple_fit_weight = enhanced_weights['small']
        # Further reduce best-fit weight for small items
        best_fit_weight *= 0.93
        multiple_fit_weight *= 1.01
    else:  # Medium item based on distribution
        selected_common_sizes = enhanced_embeddings['medium']
        best_fit_weight, multiple_fit_weight = enhanced_weights['medium']
    
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
    
    # Fragmentation penalty: penalize post-placement remainders in [0.2, 0.7] of bin capacity
    # First, find the original bin capacities by adding back the item size
    original_bin_capacities = feasible_post_caps + item
    
    # Normalize the post-placement capacities to be relative to original bin capacity
    normalized_remainders = np.divide(feasible_post_caps, original_bin_capacities, 
                                      out=np.ones_like(feasible_post_caps)*-1, 
                                      where=original_bin_capacities!=0)
    
    # Create fragmentation penalty: apply negative penalty for normalized remainders in [0.2, 0.7]
    frag_penalty = np.zeros_like(feasible_post_caps)
    frag_mask = (normalized_remainders >= 0.2) & (normalized_remainders <= 0.7)
    
    # Use a smooth penalty function instead of hard cut-offs
    # Apply quadratic penalty within the [0.2, 0.7] range, peaking at 0.45 (middle of range)
    mid_point = 0.45
    width = 0.25  # Half-width of the interval [0.2, 0.7]
    
    # Calculate how far each remainder is from the midpoint of the problematic range
    dist_from_mid = np.abs(normalized_remainders - mid_point)
    # Only apply penalty if within the problematic range
    penalty_mask = dist_from_mid <= width
    # Quadratic penalty function: peaks at the center of the bad range
    # Use a very subtle penalty strength to minimally disrupt other scoring components
    frag_penalty[penalty_mask] = -((width**2 - dist_from_mid[penalty_mask]**2) / width**2) * 0.05
    
    # Combine all components with enhanced weights
    feasible_scores = (
        multiple_fit_weight * multiple_fit_score + 
        best_fit_weight * best_fit_component +
        frag_penalty  # Add fragmentation penalty
    )
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores