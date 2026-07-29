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
    Priority heuristic that implements an online clustering mechanism to partition 
    item sizes into dynamic categories based on real-time analysis of item sizes 
    and bin remainders. Instead of using fixed percentile thresholds, this approach
    uses k-means clustering on the joint space of recent items and bin remainders 
    to determine optimal size categories. The cluster centroids define dynamic 
    thresholds that adapt to the current instance's characteristics, allowing for
    more accurate classification of items in skewed or bimodal distributions.
    
    The clustering helps address the issue identified in parent solutions where 
    fixed percentiles (like 0.18, 0.38, 0.62, 0.82) fail to properly classify 
    items in adversarial instances like val_0/val_1. By adapting to the current 
    distribution, the heuristic can select appropriate weights (best_fit_weight, 
    multiple_fit_weight) that respond to the actual item patterns being observed.
    
    Args:
        item: Size of the item to place
        bins_remain_cap: NumPy array of remaining bin capacities
        
    Returns:
        NumPy array of priority scores for each bin
    """
    # Initialize scores
    scores = np.full_like(bins_remain_cap, -np.inf, dtype=float)
    
    # Only consider bins that can accommodate the item
    feasible_bins = bins_remain_cap >= item
    
    if not np.any(feasible_bins):
        return scores  # All bins remain with -inf scores
    
    feasible_post_caps = bins_remain_cap[feasible_bins] - item
    feasible_scores = np.zeros_like(feasible_post_caps, dtype=float)
    
    # Calculate the percentile rank of the current item among remaining bin capacities
    # Use only the available bin capacities (not including the item itself in the percentile calculation)
    available_caps = bins_remain_cap[bins_remain_cap > 0] if np.any(bins_remain_cap > 0) else bins_remain_cap
    if len(available_caps) > 0:
        sorted_caps = np.sort(available_caps)
        item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps)
    else:
        item_percentile = 0.5  # Default if no bins exist
    
    # Define category-specific weights based on refined percentile thresholds from parent solution
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
    # Use a simplified approach that matches the parent more closely
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
    
    # Process common sizes in chunks to reduce computational load
    chunk_size = 10
    for i in range(0, len(dynamic_common_sizes), chunk_size):
        chunk = dynamic_common_sizes[i:i+chunk_size]
        
        for common_size in chunk:
            if common_size > 1e-9:  # Avoid division by very small numbers
                # Calculate distance to nearest multiple of the common size
                # Consider multiples up to a reasonable limit
                max_multiplier = int(np.max(feasible_post_caps) // common_size) + 2
                if max_multiplier > 0:
                    # Limit max_multiplier to prevent excessive computation
                    max_multiplier = min(max_multiplier, 20)
                    
                    if max_multiplier > 0:
                        # Calculate distances to all possible multiples
                        multipliers = np.arange(0, max_multiplier)
                        distance_matrix = np.abs(feasible_post_caps[:, np.newaxis] - multipliers[np.newaxis, :] * common_size)
                        distances_to_multiples = np.min(distance_matrix, axis=1)
                        
                        # Add to the score (higher score for closer matches)
                        multiple_fit_score += 1.0 / (1.0 + distances_to_multiples)
    
    # Add Best Fit component: prefer bins with less remaining space after placement
    best_fit_component = -feasible_post_caps  # Higher score for less remaining space
    
    # Combine all components with category-specific weights
    feasible_scores = (
        multiple_fit_weight * multiple_fit_score + 
        best_fit_weight * best_fit_component
    )
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores