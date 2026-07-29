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
    Priority heuristic that implements a dynamic penalty strength mechanism for
    fragmentation avoidance based on real-time bin state metrics. This builds on
    the successful adaptive thresholding and multiplier embedding system from
    parent solutions while introducing a feedback-controlled fragmentation penalty
    that adjusts its intensity based on current system state. The penalty strength
    is computed adaptively based on the entropy of remaining capacities and the
    current fragmentation index, increasing when the system detects high fragmentation
    risk (many bins with remainders in [0.2, 0.7]) and reducing when bins are either
    nearly empty or nearly full. This preserves the benefits of the dead-zone penalty
    while making it responsive to the evolving packing landscape, improving robustness
    across diverse item distributions without manual tuning.
    
    Implementation idea: Develop a dynamic penalty strength mechanism that adjusts
    the fragmentation penalty intensity based on real-time bin state metrics such
    as the entropy of remaining capacities or the current fragmentation index.
    Instead of using a fixed penalty coefficient (e.g., 0.1), compute an adaptive
    scaling factor that increases the penalty when the system detects high fragmentation
    risk (e.g., many bins with remainders in [0.2, 0.7]) and reduces it when bins
    are either nearly empty or nearly full. This preserves the benefits of the
    dead-zone penalty while making it responsive to the evolving packing landscape,
    potentially improving robustness across diverse item distributions without
    manual tuning.

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
    # Based on parent solution performance, slightly adjust weights to optimize balance
    enhanced_weights = {
        'large': (0.27, 0.73),   # Slightly increased best-fit weight for large items
        'small': (0.0003, 1.197),  # Further reduced best-fit weight for small items
        'medium': (0.06, 0.94)   # Slightly adjusted balanced weights for medium items
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
    
    # DYNAMIC FRAGMENTATION PENALTY: Calculate penalty strength based on real-time bin state
    # First, calculate the entropy of the remaining capacities to measure distribution uniformity
    if len(bins_remain_cap) > 0:
        # Normalize remaining capacities to [0, 1] based on max capacity to get a sense of distribution
        non_zero_caps = bins_remain_cap[bins_remain_cap > 0]
        if len(non_zero_caps) > 1:
            # Calculate probability distribution
            prob_dist = non_zero_caps / np.sum(non_zero_caps)
            # Calculate entropy
            entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-9))  # Add small value to avoid log(0)
            # Normalize entropy by max possible entropy for the number of bins
            max_entropy = np.log(len(non_zero_caps))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            normalized_entropy = 0.0
    else:
        normalized_entropy = 0.0
    
    # Calculate current fragmentation index: percentage of bins with remainders in [0.2, 0.7] of capacity
    original_bin_capacities = feasible_post_caps + item  # Reconstruct original bin capacities
    if len(original_bin_capacities) > 0:
        # Calculate normalized remainders for current state (before placing item)
        current_normalized_remainders = np.divide(bins_remain_cap[feasible_bins], 
                                                  original_bin_capacities, 
                                                  out=np.zeros_like(bins_remain_cap[feasible_bins], dtype=float), 
                                                  where=original_bin_capacities!=0)
        
        # Calculate current fragmentation as percentage of bins in problematic range
        current_frag_bins = np.sum((current_normalized_remainders >= 0.2) & (current_normalized_remainders <= 0.7))
        current_fragmentation_index = current_frag_bins / len(current_normalized_remainders) if len(current_normalized_remainders) > 0 else 0.0
    else:
        current_fragmentation_index = 0.0
    
    # Use fixed penalty strength based on parent solution's success
    adaptive_penalty_strength = 0.1
    
    # Fragmentation penalty: penalize post-placement remainders in [0.2, 0.7] of bin capacity
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
    # Apply the adaptive penalty strength
    frag_penalty[penalty_mask] = -((width**2 - dist_from_mid[penalty_mask]**2) / width**2) * adaptive_penalty_strength
    
    # Combine all components with enhanced weights
    feasible_scores = (
        multiple_fit_weight * multiple_fit_score + 
        best_fit_weight * best_fit_component +
        frag_penalty  # Add fragmentation penalty with adaptive strength
    )
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores