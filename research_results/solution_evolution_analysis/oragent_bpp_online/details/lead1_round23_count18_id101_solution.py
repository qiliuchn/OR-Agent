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
    Priority heuristic that implements an instance-adaptive weighting mechanism
    based on real-time metrics of bin utilization dispersion (packing coherence index).
    The approach dynamically adjusts the balance between best-fit and multiple-fit
    components based on the current distribution of remaining bin capacities.
    Low coherence (high dispersion) triggers increased best-fit emphasis to reduce
    fragmentation, while high coherence (uniform remainders) favors multiple-fit
    exploration to exploit structured gaps. This addresses performance variance
    across instances by making the heuristic responsive to emergent packing
    structure rather than relying solely on relative item size.

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
    
    # Calculate packing coherence index based on current bin capacity distribution
    # Using coefficient of variation as a measure of dispersion
    if len(bins_remain_cap) > 1 and np.std(bins_remain_cap) > 0:
        # Coefficient of variation (std/mean) as inverse of coherence
        cv = np.std(bins_remain_cap) / (np.mean(bins_remain_cap) + 1e-9)
        # Normalize to [0, 1] range using a sigmoid-like transformation
        coherence_index = 1 - np.exp(-cv)  # Higher value means more coherent
    else:
        coherence_index = 0.5  # Default when insufficient data
    
    # Define base weights based on coherence index
    # High coherence -> favor multiple-fit exploration
    # Low coherence -> favor best-fit to reduce fragmentation
    base_multiple_weight = 0.8 + 0.4 * coherence_index  # Ranges from 0.8 to 1.2
    base_best_fit_weight = 0.4 - 0.2 * coherence_index  # Ranges from 0.2 to 0.4
    
    # Calculate the percentile rank of the current item among remaining bin capacities
    if len(bins_remain_cap) > 0:
        sorted_caps = np.sort(bins_remain_cap)
        item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps)
    else:
        item_percentile = 0.5  # Default if no bins exist
    
    # Enhanced multiplier embeddings with additional fractional relationships from parent solution
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
    
    # Determine the size category based on refined percentile thresholds
    if item_percentile > 0.82:  # Very large item
        selected_common_sizes = enhanced_embeddings['large']
        # Apply coherence-adjusted weights with additional tuning for large items
        best_fit_weight = base_best_fit_weight * 1.07
        multiple_fit_weight = base_multiple_weight * 0.99
    elif item_percentile < 0.18:  # Very small item
        selected_common_sizes = enhanced_embeddings['small']
        # Apply coherence-adjusted weights with additional tuning for small items
        best_fit_weight = base_best_fit_weight * 0.93
        multiple_fit_weight = base_multiple_weight * 1.01
        # Ensure minimum best-fit influence for small items
        best_fit_weight = max(best_fit_weight, 0.001)
    elif item_percentile > 0.62:  # Large-medium item
        selected_common_sizes = enhanced_embeddings['medium']
        best_fit_weight = (base_best_fit_weight * 0.8 + 0.14 * 0.2)  # Blend with fixed medium weights
        multiple_fit_weight = (base_multiple_weight * 0.8 + 0.86 * 0.2)
    elif item_percentile < 0.38:  # Small-medium item
        selected_common_sizes = enhanced_embeddings['small']  # Use small's rich multiplier set
        best_fit_weight = (base_best_fit_weight * 0.8 + 0.025 * 0.2)  # Blend with fixed small-medium weights
        multiple_fit_weight = (base_multiple_weight * 0.8 + 1.055 * 0.2)
    else:  # True medium item
        selected_common_sizes = enhanced_embeddings['medium']
        best_fit_weight = base_best_fit_weight
        multiple_fit_weight = base_multiple_weight
    
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
    best_fit_component = -feasible_post_caps  # Higher score for less remaining space
    
    # Combine all components with adaptive weights (without penalty term)
    feasible_scores = (
        multiple_fit_weight * multiple_fit_score + 
        best_fit_weight * best_fit_component
    )
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores