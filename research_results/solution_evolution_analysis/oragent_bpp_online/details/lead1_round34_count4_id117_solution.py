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
    
    # Calculate context-sensitive weights based on entropy or coefficient of variation of the remainder distribution
    if len(bins_remain_cap) > 1:
        # Calculate coefficient of variation as a measure of dispersion
        mean_remain = np.mean(bins_remain_cap)
        std_remain = np.std(bins_remain_cap)
        if mean_remain > 1e-9:  # Avoid division by zero
            cv_remain = std_remain / mean_remain
        else:
            cv_remain = 0
        
        # Alternatively calculate entropy-like measure using normalized histogram
        if len(bins_remain_cap) > 1:
            # Normalize remainders to avoid scale effects
            norm_remain = bins_remain_cap / (mean_remain if mean_remain > 1e-9 else 1.0)
            # Calculate entropy based on distribution of normalized remainders
            # Use a simple approach: compute relative frequencies of different ranges
            hist, _ = np.histogram(norm_remain, bins=10)
            hist = hist + 1e-9  # Add small value to avoid log(0)
            prob = hist / np.sum(hist)
            entropy = -np.sum(prob * np.log(prob + 1e-9))  # Add small value to avoid log(0)
        else:
            entropy = 0
    else:
        cv_remain = 0
        entropy = 0
    
    # Use context information to adjust the balance between quantile-based and ratio-based common sizes
    # High entropy or high CV indicates more heterogeneous remainder distribution
    # Low entropy or low CV indicates more clustered/homogeneous remainder distribution
    
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
            
            # Adjust the emphasis on quantile vs ratio based on context
            # When remainder distribution is homogeneous (low entropy/CV), emphasize item ratios
            # When remainder distribution is heterogeneous (high entropy/CV), emphasize quantiles
            context_factor = entropy + cv_remain  # Combined measure of heterogeneity
            
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
            
            # Apply context-dependent weighting to different types of common sizes
            # Use a more nuanced approach with three levels of heterogeneity
            if context_factor < 0.3:  # Very homogeneous distribution
                # Strongly emphasize item-ratio-based common sizes to exploit repetitive patterns
                weighted_quantiles = quantile_values
                weighted_item_ratios = np.concatenate([item_ratios] * 4)  # Quadruple emphasis
                weighted_remainder_fractions = remainder_fractions
            elif context_factor < 0.7:  # Moderately heterogeneous distribution
                # Balanced approach
                weighted_quantiles = np.concatenate([quantile_values] * 2)  # Double emphasis
                weighted_item_ratios = np.concatenate([item_ratios] * 2)  # Double emphasis
                weighted_remainder_fractions = np.concatenate([remainder_fractions] * 2)  # Double emphasis
            else:  # Highly heterogeneous distribution
                # Emphasize quantile-derived common sizes to capture structural scales
                weighted_quantiles = np.concatenate([quantile_values] * 3)  # Triple emphasis
                weighted_item_ratios = item_ratios
                weighted_remainder_fractions = np.concatenate([remainder_fractions] * 3)  # Triple emphasis
            
            # Combine all dynamically generated common sizes
            dynamic_common_sizes = np.concatenate([
                weighted_quantiles,
                weighted_item_ratios,
                weighted_remainder_fractions
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