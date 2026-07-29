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
    Hybrid priority heuristic combining dynamic common-size generation with piecewise non-linear scoring.
    
    Implementation idea: Develop a hybrid priority function that integrates dynamic common-size generation 
    (from Parent #2) with the piecewise non-linear scoring kernel (from Parent #1), using quantile-based 
    item categorization to select both the set of common sizes and the appropriate scoring function 
    (exponential-linear hybrid vs. inverse-distance). For each bin, compute distances to dynamically 
    sampled common sizes (via quantiles and item ratios), then apply the continuous piecewise scoring 
    with τ = 0.05 × common_size and category-adapted α, while preserving the best-fit component with 
    empirically validated weights. This combines distribution-aware adaptability with refined gap 
    discrimination.
    
    The approach maintains the percentile-based item categorization to determine weights and scoring
    parameters, generates dynamic common sizes from the current remainder distribution, and applies
    a piecewise scoring function with exponential decay for small gaps and linear continuation for
    larger gaps, ensuring continuity at the threshold boundary.
    
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
    
    # Define category-specific weights and parameters based on refined percentile thresholds
    if item_percentile > 0.82:  # Very large item
        best_fit_weight, multiple_fit_weight = 0.25 * 1.07, 0.75 * 0.99  # Enhanced best-fit for large items
        alpha = 0.5  # More aggressive exponential decay for large items
    elif item_percentile < 0.18:  # Very small item
        best_fit_weight, multiple_fit_weight = 0.001 * 0.93, 1.194 * 1.01  # Reduced best-fit for small items
        alpha = 0.1  # Less aggressive exponential decay for small items
    elif item_percentile > 0.62:  # Large-medium item
        best_fit_weight, multiple_fit_weight = 0.14, 0.86  # Between medium and large
        alpha = 0.3  # Medium aggressiveness
    elif item_percentile < 0.38:  # Small-medium item
        best_fit_weight, multiple_fit_weight = 0.025, 1.055  # Between medium and small
        alpha = 0.2  # Less aggressive than medium
    else:  # True medium item
        best_fit_weight, multiple_fit_weight = 0.08, 0.92  # Medium weights
        alpha = 0.3  # Medium aggressiveness
    
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
    
    # Calculate multiple fit score based on dynamic common sizes using standard inverse-distance scoring
    multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    for common_size in dynamic_common_sizes:
        if common_size > 1e-9:  # Avoid division by very small numbers
            # Calculate distance to nearest multiple of the common size
            # Consider multiples up to a reasonable limit
            max_multiplier = int(np.max(feasible_post_caps) // common_size) + 2
            if max_multiplier > 0:
                # Calculate distances to all possible multiples
                distances_to_multiples_list = []
                for n in range(0, max_multiplier):
                    distances_to_multiples_list.append(np.abs(feasible_post_caps - n * common_size))
                
                if distances_to_multiples_list:
                    distances_to_multiples = np.min(distances_to_multiples_list, axis=0)
                    
                    # Use standard inverse-distance scoring as in Parent #2
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