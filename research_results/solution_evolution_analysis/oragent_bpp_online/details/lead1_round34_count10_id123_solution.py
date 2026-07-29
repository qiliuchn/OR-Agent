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
    Priority heuristic for online bin packing that combines best-fit and first-fit strategies
    with dynamic common size considerations. For each feasible bin, calculates a priority score
    based on how well the item fits and potential for multiple items to fit in the same bin.
    
    Args:
        item: Size of the item to place
        bins_remain_cap: NumPy array of remaining bin capacities
        
    Returns:
        NumPy array of priority scores for each bin
    """
    # Calculate post-placement remaining capacities
    post_placement_caps = bins_remain_cap - item
    
    # Initialize scores with -infinity for infeasible bins
    scores = np.full_like(bins_remain_cap, -np.inf, dtype=float)
    
    # Only consider bins that can accommodate the item
    feasible_bins = bins_remain_cap >= item
    
    if not np.any(feasible_bins):
        return scores  # Return all -inf scores if no bins can fit the item
    
    feasible_post_caps = post_placement_caps[feasible_bins]
    feasible_scores = np.zeros_like(feasible_post_caps, dtype=float)
    
    if len(feasible_post_caps) == 0:
        return scores
    
    # Generate dynamic common sizes using context-aware sampling (from parent solution)
    # Use a simplified approach: sample quantiles and key ratios from the distribution
    if len(feasible_post_caps) > 0:
        # Identify important scales from the current remainder distribution
        # Use a simplified approach: sample quantiles and key ratios from the distribution
        unique_caps = np.unique(feasible_post_caps)
        if len(unique_caps) > 0:
            # Sample key values from the current distribution as common sizes
            # Quantiles provide good coverage of the distribution
            quantiles = np.array([0.1, 0.25, 0.33, 0.5, 0.67, 0.75, 0.9])
            quantile_values = np.quantile(unique_caps, quantiles)
            
            # Also include some ratios of the current item size
            item_ratios = np.array([item, item * 0.5, item * 1.5, item * 0.75, item * 1.25])
            
            # Include some simple fractions of the current remainders
            remainder_fractions = np.concatenate([
                unique_caps * 0.5,
                unique_caps * 0.33,
                unique_caps * 0.67,
                unique_caps * 0.25,
                unique_caps * 0.75
            ])
            
            # Combine all dynamically generated common sizes
            dynamic_common_sizes = np.concatenate([
                quantile_values,
                item_ratios,
                remainder_fractions
            ])
            
            # Remove duplicates and very small values
            dynamic_common_sizes = np.unique(dynamic_common_sizes)
            dynamic_common_sizes = dynamic_common_sizes[dynamic_common_sizes > 1e-9]
        else:
            # Fallback if no unique values exist
            dynamic_common_sizes = np.array([item, item * 0.5, item * 1.5])
    else:
        dynamic_common_sizes = np.array([item, item * 0.5, item * 1.5])

    # To manage computational complexity, limit the number of common sizes considered
    if len(dynamic_common_sizes) > 30:
        # Take a representative subset if too many common sizes
        step = len(dynamic_common_sizes) // 30
        dynamic_common_sizes = dynamic_common_sizes[::step][:30]
    
    # Calculate the percentile rank of the current item among remaining bin capacities
    if len(bins_remain_cap) > 0:
        # Calculate the percentile of the current item in the context of remaining capacities
        sorted_caps = np.sort(bins_remain_cap)
        item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps)
    else:
        item_percentile = 0.5  # Default if no bins exist
    
    # Calculate multiple fit scores for all feasible bins at once
    multiple_fit_scores = np.zeros_like(feasible_post_caps)
    
    for common_size in dynamic_common_sizes:
        if common_size <= 1e-9:
            continue
        
        # Vectorized calculation of distances to nearest multiples
        multipliers = np.floor(feasible_post_caps / common_size).astype(int)
        dist_to_multiple1 = np.abs(feasible_post_caps - multipliers * common_size)
        dist_to_multiple2 = np.abs(feasible_post_caps - (multipliers + 1) * common_size)
        dist_to_multiple = np.minimum(dist_to_multiple1, dist_to_multiple2)
        
        multiple_fit_scores += 1.0 / (1.0 + dist_to_multiple)
    
    # Calculate best fit component (prefer bins with less remaining space after placement)
    best_fit_component = -feasible_post_caps  # Higher score for less remaining space
    
    # Determine weights based on item percentile
    if item_percentile > 0.7:  # Large item - prioritize best fit
        best_fit_weight, multiple_fit_weight = 0.8, 0.2
    elif item_percentile < 0.3:  # Small item - prioritize multiple fit potential
        best_fit_weight, multiple_fit_weight = 0.3, 0.7
    else:  # Medium item - balance both
        best_fit_weight, multiple_fit_weight = 0.6, 0.4
    
    # Combine components
    feasible_scores = (
        multiple_fit_weight * multiple_fit_scores + 
        best_fit_weight * best_fit_component
    )
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores