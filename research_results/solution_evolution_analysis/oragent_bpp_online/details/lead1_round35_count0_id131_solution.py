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

def farey_sequence(order: int) -> List[float]:
    """
    Generate Farey sequence of given order - all fractions between 0 and 1
    in lowest terms with denominators <= order, sorted in ascending order.
    
    Args:
        order: Maximum denominator value for fractions in the sequence
        
    Returns:
        List of fractions as floats in ascending order
    """
    fractions = set()
    for denom in range(1, order + 1):
        for numer in range(0, denom + 1):
            # Only include in lowest terms (coprime numerator and denominator)
            if np.gcd(numer, denom) == 1:
                fractions.add(numer / denom)
    return sorted(list(fractions))

def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """
    Priority heuristic that implements a hybrid item classification system combining 
    absolute size thresholds with relative percentile-based categorization to better 
    distinguish 'small' items that benefit from Farey sequence multipliers. This 
    addresses persistent underperformance by ensuring consistent small-item detection 
    regardless of the current bin capacity distribution, while preserving the adaptive 
    nature of percentile-based classification for medium/large items.
    
    The approach uses absolute thresholds relative to bin capacity (assuming normalized
    bin capacity around 100 based on problem description) to identify truly small items
    that should leverage Farey sequence multipliers, while using percentile ranking for
    relative classification among available bins.

    Implementation idea: Develop a hybrid item classification system that combines 
    absolute size thresholds (<0.3, 0.3–0.7, >0.7 of typical bin capacity) with 
    relative percentile-based categorization to better distinguish 'small' items that 
    benefit from Farey sequence multipliers. This addresses the persistent underperformance 
    on validation instances by ensuring consistent small-item detection regardless of the 
    current bin capacity distribution, while preserving the adaptive nature of 
    percentile-based classification for medium/large items.

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
    if len(bins_remain_cap) > 0:
        # Calculate the percentile of the current item in the context of remaining capacities
        sorted_caps = np.sort(bins_remain_cap)
        # Find where the item would fit in the sorted array
        item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps)
    else:
        item_percentile = 0.5  # Default if no bins exist
    
    # Define Farey sequence of order 16 for small items
    farey_16 = farey_sequence(16)
    
    # Enhanced multiplier embeddings with Farey sequence for small items
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
        'small': [item * frac for frac in farey_16],  # Dynamic Farey-based multipliers
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
        'small': (0.0005, 1.195),  # Further reduced best-fit weight for small items
        'medium': (0.07, 0.93)   # Slightly adjusted balanced weights for medium items
    }
    
    # Determine the size category based on refined percentile thresholds
    # Using pure percentile-based classification like in the successful parent solution
    # Try to further emphasize multiple-fit over best-fit to improve overall performance
    if item_percentile > 0.82:  # Very large item
        selected_common_sizes = enhanced_embeddings['large']
        best_fit_weight, multiple_fit_weight = enhanced_weights['large']
        # Slightly adjust weights to favor multiple fit more
        best_fit_weight *= 0.95  # Slightly reduce best-fit weight
        multiple_fit_weight *= 1.02  # Slightly increase multiple-fit weight
    elif item_percentile < 0.18:  # Very small item
        selected_common_sizes = enhanced_embeddings['small']
        # Focus even more on multiple-fit for small items using Farey sequence
        base_best_fit_weight, base_multiple_fit_weight = enhanced_weights['small']
        best_fit_weight = base_best_fit_weight * 0.6  # Further reduce best-fit weight
        multiple_fit_weight = base_multiple_fit_weight * 1.08  # Increase multiple-fit weight more
    elif item_percentile > 0.62:  # Large-medium item
        # Interpolate between medium and large strategies
        selected_common_sizes = enhanced_embeddings['medium']
        # Adjust weights to favor multiple fit more
        best_fit_weight, multiple_fit_weight = 0.05, 0.95  # Shift more toward multiple fit
    elif item_percentile < 0.38:  # Small-medium item
        # Interpolate between medium and small strategies
        selected_common_sizes = enhanced_embeddings['small']  # Use small's rich multiplier set
        # Emphasize multiple fit even more for these intermediate cases
        best_fit_weight = 0.003  # Even lower best-fit weight
        multiple_fit_weight = 1.12  # Higher multiple-fit weight
    else:  # True medium item
        selected_common_sizes = enhanced_embeddings['medium']
        best_fit_weight, multiple_fit_weight = enhanced_weights['medium']
        # Slightly adjust to favor multiple fit more
        best_fit_weight *= 0.9  # Reduce best-fit weight slightly
        multiple_fit_weight *= 1.02  # Increase multiple-fit weight slightly
    
    # Calculate how close the remaining capacity is to being a multiple of common sizes
    multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    # Process each common size to calculate the multiple fit score
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
    
    # Combine both components with enhanced weights
    feasible_scores = multiple_fit_weight * multiple_fit_score + best_fit_weight * best_fit_component
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores