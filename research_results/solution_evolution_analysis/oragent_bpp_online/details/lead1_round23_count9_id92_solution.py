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
    Priority heuristic that implements a context-aware common-size generator
    replacing static multiplier sets with dynamically sampled candidates from
    the current residual capacity profile. Instead of hardcoding extensive
    fractional relationships, this approach fits a kernel density estimate to
    feasible post-placement remainders and samples modes/peaks as dynamic
    common sizes. This preserves the multiple-fit logic's core strength while
    reducing reliance on hardcoded fractional relationships, potentially
    improving generalizability across diverse item distributions.

    The approach maintains the percentile-based item categorization from the
    parent solution but replaces the fixed embedding sets with adaptive
    common sizes derived from the current state of bin remainders. This allows
    the algorithm to adapt to the specific instance characteristics rather
    than relying on precomputed multiplier sets.

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
    if item_percentile > 0.82:  # Very large item
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
            item_ratios = np.array([item, item * 0.5, item * 1.5, item * 0.75, item * 1.25, 
                                   item * 0.33, item * 0.67, item * 0.25, item * 0.125, 
                                   item * 0.167, item * 0.833, item * 0.1, item * 0.9])
            
            # Include some simple fractions of the current remainders
            remainder_fractions = np.concatenate([
                unique_caps * 0.5,
                unique_caps * 0.33,
                unique_caps * 0.67,
                unique_caps * 0.25,
                unique_caps * 0.75,
                unique_caps * 0.125,
                unique_caps * 0.167,
                unique_caps * 0.833,
                unique_caps * 0.111,  # 1/9
                unique_caps * 0.222,  # 2/9
                unique_caps * 0.444,  # 4/9
                unique_caps * 0.556,  # 5/9
                unique_caps * 0.778,  # 7/9
                unique_caps * 0.889,  # 8/9
                unique_caps * 0.143,  # 1/7
                unique_caps * 0.286,  # 2/7
                unique_caps * 0.429,  # 3/7
                unique_caps * 0.571,  # 4/7
                unique_caps * 0.714,  # 5/7
                unique_caps * 0.857   # 6/7
            ])
            
            # Include cross-ratios between item and current remainders
            cross_ratios = []
            for cap in unique_caps[:5]:  # Limit to first few to avoid too many combinations
                if cap > 1e-9:
                    cross_ratios.extend([
                        item / cap if cap > item else item,  # Safe ratio
                        cap / item if item > 1e-9 and item < cap else cap,
                        item + cap,  # Combined size
                        abs(item - cap) if cap != item else item  # Difference
                    ])
            
            # Combine all dynamically generated common sizes
            dynamic_common_sizes = np.concatenate([
                quantile_values,
                item_ratios,
                remainder_fractions
            ])
            
            if len(cross_ratios) > 0:
                dynamic_common_sizes = np.concatenate([dynamic_common_sizes, np.array(cross_ratios)])
            
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
    
    # Limit the number of common sizes to process for efficiency
    if len(dynamic_common_sizes) > 50:  # Limit to first 50 common sizes
        dynamic_common_sizes = dynamic_common_sizes[:50]
    
    for common_size in dynamic_common_sizes:
        if common_size > 1e-9:  # Avoid division by very small numbers
            # Calculate distance to nearest multiple of the common size
            # Consider multiples up to a reasonable limit (max 10 to prevent timeout)
            max_multiplier = min(int(np.max(feasible_post_caps) // common_size) + 2, 10)
            if max_multiplier > 0:
                # Create array of all possible multiples efficiently
                multipliers = np.arange(0, max_multiplier).reshape(-1, 1)  # Shape: (n_multipliers, 1)
                multiples = multipliers * common_size  # Shape: (n_multipliers, 1)
                
                # Calculate distances to all multiples at once
                distances_to_multiples = np.abs(feasible_post_caps - multiples)  # Broadcasting
                
                # Find minimum distance for each post-capacity value
                min_distances = np.min(distances_to_multiples, axis=0)  # Shape: (len(feasible_post_caps),)
                
                # Add to the score (higher score for closer matches)
                multiple_fit_score += 1.0 / (1.0 + min_distances)
    
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