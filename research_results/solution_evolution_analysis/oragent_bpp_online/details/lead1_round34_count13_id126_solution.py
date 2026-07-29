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
from collections import deque

def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """
    Priority heuristic implementing an integrated approach similar to the parent solution,
    combining best-fit, multiple-fit, and recyclability components with adaptive weights
    based on item percentile categories. This addresses the performance degradation seen
    with the switching architecture by re-integrating complementary components.
    
    Implementation idea: Revert from the switching architecture to an integrated approach
    similar to the parent, but keep improvements to focus on most effective elements.
    Use item percentile-based weight adjustment and simplified multiple-fit scoring.
    
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
        item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps)
    else:
        item_percentile = 0.5  # Default if no bins exist
    
    # Define category-specific weights based on refined percentile thresholds
    # Using similar thresholds as parent solution
    if item_percentile > 0.82:  # Very large item
        best_fit_weight, multiple_fit_weight, recyclability_weight = 0.25 * 1.07, 0.75 * 0.99, 0.05
    elif item_percentile < 0.18:  # Very small item
        best_fit_weight, multiple_fit_weight, recyclability_weight = 0.001 * 0.93, 1.194 * 1.01, 0.1
    elif item_percentile > 0.62:  # Large-medium item
        best_fit_weight, multiple_fit_weight, recyclability_weight = 0.14, 0.86, 0.05
    elif item_percentile < 0.38:  # Small-medium item
        best_fit_weight, multiple_fit_weight, recyclability_weight = 0.025, 1.055, 0.08
    else:  # True medium item
        best_fit_weight, multiple_fit_weight, recyclability_weight = 0.08, 0.92, 0.05
    
    # Generate dynamic common sizes using context-aware sampling (enhanced from parent)
    if len(feasible_post_caps) > 0:
        # Identify important scales from the current remainder distribution
        unique_caps = np.unique(feasible_post_caps)
        if len(unique_caps) > 0:
            # Sample key values from the current distribution as common sizes
            # Quantiles provide good coverage of the distribution
            quantiles = np.array([0.1, 0.25, 0.33, 0.5, 0.67, 0.75, 0.9])
            quantile_values = np.quantile(unique_caps, quantiles)
            
            # Also include some ratios of the current item size
            item_ratios = np.array([item, item * 0.5, item * 1.5, item * 0.75, item * 1.25])
            
            # Include some simple fractions of the current remainders to capture patterns
            remainder_fractions = np.concatenate([
                unique_caps * 0.5,
                unique_caps * 0.33,
                unique_caps * 0.67
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
    if len(dynamic_common_sizes) > 25:
        # Take a representative subset if too many common sizes
        step = len(dynamic_common_sizes) // 25
        dynamic_common_sizes = dynamic_common_sizes[::step][:25]
    
    # Calculate multiple fit score based on nearest multiple of dynamic common sizes
    multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    for common_size in dynamic_common_sizes:
        if common_size > 1e-9:  # Avoid division by very small numbers
            # Calculate distance to nearest multiple of the common size
            # Find the nearest multiplier
            multipliers = np.round(feasible_post_caps / common_size)
            nearest_multiples = multipliers * common_size
            distances_to_nearest_multiple = np.abs(feasible_post_caps - nearest_multiples)
            
            # Add to the score (higher score for closer matches)
            multiple_fit_score += 1.0 / (1.0 + distances_to_nearest_multiple)
    
    # Add Best Fit component: prefer bins with less remaining space
    best_fit_component = -feasible_post_caps  # Higher score for less remaining space
    
    # Calculate recyclability metric based on distribution entropy
    # Use quantiles of the current remainder distribution as proxy for common small items
    recyclability_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    # Estimate empirical modes of historical small items by using quantiles of the current distribution
    # This is a proxy for the recyclability of remainders
    if len(feasible_post_caps) > 0:
        # Use quantiles of the current remainder distribution as proxy for common small items
        small_item_modes = np.array([np.quantile(feasible_post_caps, q) for q in [0.1, 0.2, 0.3]])
        small_item_modes = small_item_modes[small_item_modes > 1e-9]  # Filter out tiny values
        
        if len(small_item_modes) > 0:
            # Calculate distance to closest small item mode for each remainder
            min_distances = np.full_like(feasible_post_caps, np.inf)
            for mode in small_item_modes:
                distances = np.abs(feasible_post_caps - mode)
                min_distances = np.minimum(min_distances, distances)
            
            # Higher recyclability score for remainders closer to common small item sizes
            recyclability_score = 1.0 / (1.0 + min_distances)
    
    # Adjust weights based on entropy of remainder distribution
    if len(feasible_post_caps) > 1:
        # Calculate entropy-like measure of the distribution
        remainder_probs = feasible_post_caps / np.sum(feasible_post_caps)
        remainder_probs = remainder_probs[remainder_probs > 0]  # Remove zeros
        if len(remainder_probs) > 0:
            entropy = -np.sum(remainder_probs * np.log(remainder_probs + 1e-9))
            # Normalize entropy by max possible entropy for this number of bins
            max_entropy = np.log(len(remainder_probs)) if len(remainder_probs) > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.5
            
            # Adjust weights based on entropy
            # Higher entropy (more uniform distribution) may benefit from more recyclability consideration
            recyclability_weight *= (0.5 + 0.5 * normalized_entropy)
            multiple_fit_weight *= (1.0 - 0.2 * normalized_entropy)  # Slightly reduce if highly entropic

    # Combine all components with category-specific weights
    feasible_scores = (
        multiple_fit_weight * multiple_fit_score + 
        best_fit_weight * best_fit_component +
        recyclability_weight * recyclability_score
    )
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores