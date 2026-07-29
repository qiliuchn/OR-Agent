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
    Priority heuristic that adaptively estimates common item sizes based on current 
    context without requiring persistent state. The approach infers the likely item 
    size distribution from the relationship between the current item size, bin capacity, 
    and the distribution of remaining bin capacities. This allows context-aware 
    selection of common sizes to target for post-placement remaining capacities.
    
    Implementation idea: Estimate the dominant item size patterns by analyzing the 
    current state - if most bins still have high capacity, we assume smaller items 
    dominate the sequence and adjust our common size estimates accordingly. We also 
    consider the relative size of the current item compared to typical bin capacity 
    to decide which multiples to prioritize.
    
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
    
    # Infer context from current state
    # Calculate some statistics about remaining bin capacities
    mean_remaining = np.mean(bins_remain_cap)
    std_remaining = np.std(bins_remain_cap)
    max_remaining = np.max(bins_remain_cap)
    
    # Determine if we're likely in a "small items" vs "large items" phase
    # based on the current state
    if len(bins_remain_cap) > 0:
        # If most bins still have high capacity, assume small items dominate
        high_capacity_ratio = np.sum(bins_remain_cap > 0.55 * max_remaining) / len(bins_remain_cap)
        
        # Adjust strategy based on context
        if high_capacity_ratio > 0.35:  # Likely in small items phase - adjusted threshold
            # Focus on smaller, more granular multiples
            estimated_common_sizes = [
                item, 
                item * 0.5, 
                item * 0.25, 
                item * 0.75, 
                item * 0.33, 
                item * 0.67,
                item * 1.5,
                item * 0.125,  # Adding even smaller fractions for very fine granularity
                item * 0.875,   # And some intermediate values
                item * 0.167    # Adding 1/6 fraction for more granularity
            ]
        else:  # Mixed or larger items phase
            # Focus on standard multiples
            estimated_common_sizes = [
                item,
                item * 0.5,
                item * 1.5,
                item * 0.75,
                item * 1.25,
                item * 0.33,
                item * 0.67,
                item * 2.0,
                item * 0.1,   # Adding small fraction for edge cases
                item * 1.75,   # And some other values
                item * 0.2    # Adding 1/5 fraction
            ]
    else:
        # Fallback if bins array is empty (shouldn't happen in practice)
        estimated_common_sizes = [
            item,
            item * 0.5,
            item * 1.5,
            item * 0.75,
            item * 1.25,
            item * 0.25
        ]
    
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
    
    # Adaptive weight for best fit based on context
    # In scenarios with high variance in remaining capacities, 
    # best fit might be more important
    if std_remaining > 0.12 * max_remaining:
        best_fit_weight = 0.06
    else:
        best_fit_weight = 0.03
    
    # Combine both components
    feasible_scores = multiple_fit_score + best_fit_weight * best_fit_component
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores