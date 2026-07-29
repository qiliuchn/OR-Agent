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
    Priority heuristic for online bin packing using a combination of best-fit and 
    multiple-fit strategies with efficient lookahead estimation.
    
    For each feasible bin, calculates a priority score based on:
    1. How well the bin fits the current item (best-fit component)
    2. How well the remaining capacity aligns with common item sizes (multiple-fit component)
    3. An efficient lookahead estimation that approximates future costs without full simulation

    Args:
        item: Size of the item to place
        bins_remain_cap: NumPy array of remaining bin capacities
        
    Returns:
        NumPy array of priority scores for each bin
    """
    # Initialize scores with negative infinity for invalid bins
    scores = np.full_like(bins_remain_cap, -np.inf, dtype=float)
    
    # Identify feasible bins (those with enough capacity for the item)
    feasible_bins = bins_remain_cap >= item
    
    if not np.any(feasible_bins):
        return scores  # Return all -inf if no feasible bins exist
    
    # Get remaining capacities for feasible bins only
    feasible_caps = bins_remain_cap[feasible_bins]
    post_placement_caps = feasible_caps - item  # After placing the current item
    
    # Calculate item percentile among all non-zero bin capacities for adaptive weighting
    non_zero_caps = bins_remain_cap[bins_remain_cap > 1e-9]
    if len(non_zero_caps) > 0:
        # Sort the non-zero capacities to find where the current item ranks
        sorted_caps = np.sort(non_zero_caps)
        # Calculate the percentile position of the current item
        item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps)
    else:
        item_percentile = 0.5  # Default if no bins have capacity
    
    # Determine weights based on item size relative to existing capacities
    if item_percentile > 0.82:  # Very large item
        best_fit_weight, multiple_fit_weight = 0.25 * 1.07, 0.75 * 0.99
    elif item_percentile < 0.18:  # Very small item
        best_fit_weight, multiple_fit_weight = 0.001 * 0.93, 1.194 * 1.01
    elif item_percentile > 0.62:  # Large-medium item
        best_fit_weight, multiple_fit_weight = 0.14, 0.86
    elif item_percentile < 0.38:  # Small-medium item
        best_fit_weight, multiple_fit_weight = 0.025, 1.055
    else:  # True medium item
        best_fit_weight, multiple_fit_weight = 0.08, 0.92
    
    # Best-fit component: prefer bins that leave minimal remaining space
    best_fit_score = -post_placement_caps  # Negative because smaller remaining capacity is better
    
    # Multiple-fit component: prefer bins whose remaining capacity aligns well with common sizes
    multiple_fit_score = np.zeros_like(post_placement_caps, dtype=float)
    
    # Define common sizes to check against (current item and related sizes)
    common_sizes = np.array([item, item * 0.5, item * 1.5, item * 0.75, item * 1.25])
    
    # Add quantiles of current non-zero capacities as additional common sizes
    if len(non_zero_caps) > 0:
        remainder_quantiles = np.quantile(non_zero_caps, [0.25, 0.5, 0.75])
        common_sizes = np.concatenate([common_sizes, remainder_quantiles])
    
    # For each common size, calculate how well each bin's remaining capacity matches
    for common_size in common_sizes:
        if common_size > 1e-9:
            # Calculate the modulo to see how close the remaining capacity is to being a multiple
            mod_values = post_placement_caps % common_size
            # Prefer when the remainder is close to 0 (good fit) or close to the full common_size (leaves space for another)
            distance_to_good_fit = np.minimum(mod_values, common_size - mod_values)
            multiple_fit_score += 1.0 / (1.0 + distance_to_good_fit)
    
    # Combine the components to get the base priority score
    base_priority = (
        multiple_fit_weight * multiple_fit_score + 
        best_fit_weight * best_fit_score
    )
    
    # Efficient lookahead estimation: estimate how each choice might affect future fragmentation
    # Rather than simulating, we calculate a heuristic based on the current state
    # and the potential for future good fits after this placement
    
    # Calculate the potential for good fits in the future based on post-placement capacities
    future_fit_potential = np.zeros_like(post_placement_caps, dtype=float)
    
    # For each post-placement capacity, assess how well it might fit future items
    # We'll use the same common sizes approach but applied to post-placement capacities
    for common_size in common_sizes:
        if common_size > 1e-9:
            mod_values = post_placement_caps % common_size
            distance_to_good_fit = np.minimum(mod_values, common_size - mod_values)
            # Prefer capacities that are close to multiples of common sizes
            # (meaning they could accommodate other items well in the future)
            future_fit_potential += 1.0 / (1.0 + distance_to_good_fit)
    
    # Normalize by the number of common sizes to keep scale reasonable
    if len(common_sizes) > 0:
        future_fit_potential /= len(common_sizes)
    
    # Apply lookahead weight to the future fit potential
    lookahead_weight = 0.1
    final_scores = base_priority + lookahead_weight * future_fit_potential
    
    # Assign calculated scores to the corresponding positions in the full array
    scores[feasible_bins] = final_scores
    
    return scores
