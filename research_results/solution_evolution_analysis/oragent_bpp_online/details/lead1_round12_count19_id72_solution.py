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
    Enhanced priority heuristic that builds on the successful quantile-adaptive approach
    with learned embeddings. This version includes refined adaptive weighting based on
    the current state of bin utilization and improved handling of extreme cases.
    The method maintains full statelessness while optimizing the balance between
    immediate packing efficiency and future packing flexibility.

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
    
    # Calculate additional context information for adaptive weighting
    # Measure how filled the bins currently are (utilization level)
    if len(bins_remain_cap) > 0:
        initial_capacity = 100.0  # Assuming standard bin capacity based on problem description
        current_utilization = 1.0 - (np.mean(bins_remain_cap) / initial_capacity)
    else:
        current_utilization = 0.5
    
    # Learned multiplier embeddings based on size category
    # These represent optimized sets of common-size relationships discovered through
    # offline analysis of near-optimal packings
    learned_embeddings = {
        'large': [  # For items in top 25% relative to bin capacities
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
            item * 0.2       # Additional small fraction
        ],
        'small': [  # For items in bottom 25% relative to bin capacities
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
            item * 0.833   # 5/6 more precisely
        ],
        'medium': [  # For items in middle 50% relative to bin capacities
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
            item * 0.571     # 4/7 fraction
        ]
    }
    
    # Learned weights based on size category (optimized through offline analysis)
    # With additional adaptive adjustment based on current utilization
    learned_weights = {
        'large': (0.23, 0.77),   # Slightly increased best-fit weight for large items
        'small': (0.001, 1.19),  # Further reduced best-fit weight for small items
        'medium': (0.065, 0.935)   # Slightly adjusted balanced weights for medium items
    }
    
    # Determine the size category based on percentile with refined thresholds
    if item_percentile > 0.80:  # Very large item
        estimated_common_sizes = learned_embeddings['large']
        base_best_fit_weight, base_multiple_fit_weight = learned_weights['large']
        # Slightly increase best-fit weight for very large items to ensure they fit
        best_fit_weight = base_best_fit_weight * 1.05
        multiple_fit_weight = base_multiple_fit_weight
    elif item_percentile < 0.20:  # Very small item
        estimated_common_sizes = learned_embeddings['small']
        base_best_fit_weight, base_multiple_fit_weight = learned_weights['small']
        # Further reduce best-fit weight for very small items
        best_fit_weight = base_best_fit_weight * 0.95
        multiple_fit_weight = base_multiple_fit_weight
    elif item_percentile > 0.60:  # Large-medium item
        # Interpolate between medium and large strategies
        estimated_common_sizes = learned_embeddings['medium']
        # Adjust weights to be between medium and large values
        best_fit_weight, multiple_fit_weight = 0.13, 0.87  # Between medium and large
    elif item_percentile < 0.40:  # Small-medium item
        # Interpolate between medium and small strategies
        estimated_common_sizes = learned_embeddings['small']  # Use small's rich multiplier set
        # Adjust weights to be between medium and small values
        best_fit_weight, multiple_fit_weight = 0.03, 1.05  # Between medium and small
    else:  # True medium item
        estimated_common_sizes = learned_embeddings['medium']
        base_best_fit_weight, base_multiple_fit_weight = learned_weights['medium']
        best_fit_weight = base_best_fit_weight
        multiple_fit_weight = base_multiple_fit_weight
    
    # Adaptive adjustment based on current utilization level
    # In high-utilization scenarios, prioritize best-fit to avoid opening new bins
    # In low-utilization scenarios, allow more flexibility for multiple-fit alignment
    if current_utilization > 0.75:  # High utilization - favor best-fit to fill bins
        best_fit_weight *= 1.15
        multiple_fit_weight *= 0.85
    elif current_utilization < 0.25:  # Low utilization - allow more flexibility
        best_fit_weight *= 0.85
        multiple_fit_weight *= 1.15
    
    # Filter out non-positive sizes
    estimated_common_sizes = [size for size in estimated_common_sizes if size > 0]
    
    # Calculate how close the remaining capacity is to being a multiple of common sizes
    multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    for common_size in estimated_common_sizes:
        # Calculate how many multiples of common_size fit in the remaining capacity
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
    
    # Enhanced best-fit component with penalty for creating very small residuals
    # and bonus for creating useful residual sizes that match common item sizes
    small_residual_penalty = np.zeros_like(feasible_post_caps, dtype=float)
    # Penalize bins that would leave very small remainders (less than 5% of capacity)
    small_residual_mask = (feasible_post_caps > 0) & (feasible_post_caps < 5.0)  # assuming 5% of 100 capacity
    small_residual_penalty[small_residual_mask] = -2.5  # Increased penalty for small unusable remainders
    
    # Bonus for creating useful residual sizes that could fit future items well
    useful_residual_bonus = np.zeros_like(feasible_post_caps, dtype=float)
    # Encourage leaving remainders that match common item sizes (based on current item size)
    common_item_sizes = [item, item*0.5, item*0.25, item*0.75, item*0.33, item*0.67]
    for target_size in common_item_sizes:
        if target_size > 0:
            # Bonus for leaving remainders close to common item sizes
            size_match = np.abs(feasible_post_caps - target_size)
            useful_residual_bonus += 0.5 / (1.0 + size_match)
    
    # Combine components with adaptive weights
    feasible_scores = (multiple_fit_weight * multiple_fit_score + 
                      best_fit_weight * best_fit_component + 
                      0.5 * small_residual_penalty +  # Add penalty for small residuals
                      0.3 * useful_residual_bonus)    # Add bonus for useful residuals
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores