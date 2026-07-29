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
    Priority heuristic that implements a hybrid approach combining fixed 
    percentile-based item categorization with real-time feasibility bottleneck detection.
    The approach computes the ratio of feasible bins to total bins and the entropy 
    of feasible post-capacities during each call. If this indicates a 'critical state' 
    (e.g., feasibility ratio < 0.2 and low entropy), it temporarily overrides the 
    default weights to favor extreme best-fit behavior (e.g., best_fit_weight = 0.9, 
    multiple_fit_weight = 0.1) to avoid fragmentation. This preserves the core 
    architecture from the parent solution but adds a reactive safeguard against 
    pathological states, addressing the persistent underperformance on val_0/val_1 
    where feasible options become sparse late in the sequence.
    
    Implementation idea: Hybridize the fixed percentile-based item categorization 
    with a real-time feasibility bottleneck detector: compute the ratio of feasible 
    bins to total bins and the entropy of feasible_post_caps during each call, and 
    if this indicates a 'critical state' (e.g., feasibility ratio < 0.2 and low 
    entropy), temporarily override the default weights to favor extreme best-fit 
    behavior (e.g., best_fit_weight = 0.9, multiple_fit_weight = 0.1) to avoid 
    fragmentation. This preserves the core architecture but adds a reactive 
    safeguard against pathological states, addressing the persistent underperformance 
    on val_0/val_1 where feasible options become sparse late in the sequence.

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
    
    # Compute feasibility bottleneck indicators
    total_bins = len(bins_remain_cap)
    feasible_count = np.sum(feasible_bins)
    feasibility_ratio = feasible_count / total_bins if total_bins > 0 else 0
    
    # Calculate alternative measures of capacity constraint
    if len(feasible_post_caps) > 0:
        # Count how many bins have very little remaining capacity (near full)
        # Assuming original bin capacity is around the max of initial bins_remain_cap
        original_capacity = np.max(bins_remain_cap) if len(bins_remain_cap) > 0 else 1.0
        # Count bins that are nearly full (>90% used)
        near_full_threshold = 0.1 * original_capacity
        near_full_bins = np.sum(feasible_post_caps < near_full_threshold)
        near_full_ratio = near_full_bins / len(feasible_post_caps) if len(feasible_post_caps) > 0 else 0
        
        # Calculate variance of feasible post-capacities as another measure
        capacity_variance = np.var(feasible_post_caps)
    else:
        near_full_ratio = 0
        capacity_variance = 0
    
    # Determine if we're in a critical state
    # Making the critical state detection more sensitive
    is_critical_state = (feasibility_ratio < 0.35) or (near_full_ratio > 0.25)
    
    # Define category-specific weights based on refined percentile thresholds
    if is_critical_state:
        # In critical state, extremely favor best-fit to avoid fragmentation
        best_fit_weight, multiple_fit_weight = 0.99, 0.01  # Maximum best-fit in critical state
    elif item_percentile > 0.82:  # Very large item
        best_fit_weight, multiple_fit_weight = 0.25 * 1.07, 0.75 * 0.99  # Enhanced best-fit for large items
    elif item_percentile < 0.18:  # Very small item
        best_fit_weight, multiple_fit_weight = 0.001 * 0.93, 1.194 * 1.01  # Reduced best-fit for small items
    elif item_percentile > 0.62:  # Large-medium item
        best_fit_weight, multiple_fit_weight = 0.14, 0.86  # Between medium and large
    elif item_percentile < 0.38:  # Small-medium item
        best_fit_weight, multiple_fit_weight = 0.025, 1.055  # Between medium and small
    else:  # True medium item
        best_fit_weight, multiple_fit_weight = 0.08, 0.92  # Medium weights
    
    # Generate dynamic common sizes using context-aware sampling
    # Use a kernel density-based approach to identify important remainder values
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
    
    # Calculate multiple fit score based on dynamic common sizes
    multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    for common_size in dynamic_common_sizes:
        if common_size > 1e-9:  # Avoid division by very small numbers
            # Calculate distance to nearest multiple of the common size
            # Consider multiples up to a reasonable limit
            max_multiplier = int(np.max(feasible_post_caps) // common_size) + 2
            if max_multiplier > 0:
                # Calculate distances to all possible multiples
                distances_to_multiples = np.min([
                    np.abs(feasible_post_caps - n * common_size) 
                    for n in range(0, max_multiplier)
                ], axis=0)
                
                # Add to the score (higher score for closer matches)
                multiple_fit_score += 1.0 / (1.0 + distances_to_multiples)
    
    # Add Best Fit component: prefer bins with less remaining space
    best_fit_component = -feasible_post_caps  # Higher score for less remaining space
    
    # Combine all components with category-specific weights
    feasible_scores = (
        multiple_fit_weight * multiple_fit_score + 
        best_fit_weight * best_fit_component
    )
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores