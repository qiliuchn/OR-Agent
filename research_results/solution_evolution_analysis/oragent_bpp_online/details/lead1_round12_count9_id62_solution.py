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
    Geometric progression-based hybrid priority heuristic that integrates mathematically
    motivated irrational ratios (φ and √2) into the adaptive multiplier selection logic.
    For medium-to-large items (≥10th percentile), replaces rational-fraction-heavy lists
    with geometric sequences based on φ (golden ratio ≈1.618) and √2 to better capture
    emergent packing structures in Weibull-like streams. Small items (<10th percentile)
    still use dense rational fractions for fine-grained control. This approach aims to
    improve generalization over empirical fractions by leveraging mathematical properties
    that emerge in packing configurations without increasing overfitting.
    
    Implementation idea: Replace the current rational-fraction-heavy lists for items in 
    the 25th–75th percentile with geometric sequences based on φ and √2. For items ≥10 
    percentile, use [item * φ^k for k in -2..2] and [item * (√2)^k for k in -3..1], 
    while retaining dense rational fractions only for very small items (<10th percentile).
    This creates more natural spacing between common sizes that may better match actual
    packing patterns in real distributions.
    
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
        sorted_caps = np.sort(bins_remain_cap[bins_remain_cap > 0])  # Only consider non-empty bins
        if len(sorted_caps) > 0:
            # Find where the item would fit in the sorted array
            item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps)
        else:
            item_percentile = 0.5  # Default if no non-empty bins exist
    else:
        item_percentile = 0.5  # Default if no bins exist
    
    # Mathematical constants for geometric progressions
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
    sqrt2 = np.sqrt(2)          # Square root of 2 ≈ 1.414
    # Additional mathematical constants that might be useful for packing
    sqrt3 = np.sqrt(3)          # Square root of 3 ≈ 1.732
    e_base = np.e               # Euler's number ≈ 2.718
    
    # Detect uncertainty conditions
    # Condition 1: High fragmentation - variance in bin capacities is high
    non_empty_caps = bins_remain_cap[bins_remain_cap > 0]
    if len(non_empty_caps) > 1:
        cap_variance = np.var(non_empty_caps)
        cap_mean = np.mean(non_empty_caps)
        # Normalize variance to detect high fragmentation
        normalized_variance = cap_variance / (cap_mean**2) if cap_mean > 0 else 0
        high_fragmentation = normalized_variance > 0.5  # Threshold for high fragmentation
    else:
        high_fragmentation = False
    
    # Calculate a measure of how much the current item deviates from typical remaining capacity
    if len(non_empty_caps) > 0:
        typical_capacity = np.median(non_empty_caps)
        item_outlier = item > 3 * typical_capacity or item < 0.1 * typical_capacity
    else:
        item_outlier = False
    
    # Adjust strategy based on uncertainty conditions and item percentile
    if high_fragmentation or item_outlier:
        # High uncertainty: fall back to more conservative Best Fit
        # Reduce reliance on common-size matching, emphasize immediate fit
        
        # Conservative common sizes for uncertain situations using simplified geometric progressions
        estimated_common_sizes = [item]
        # Add simplified geometric progressions to avoid too many options
        for k in range(-1, 2):  # Reduced range
            estimated_common_sizes.append(item * (phi ** k))
        for k in range(-1, 2):  # Reduced range
            estimated_common_sizes.append(item * (sqrt2 ** k))
        
        # Increase Best Fit weight, decrease multiple fit weight
        best_fit_weight = 0.25  # Slightly reduced from 0.3 to balance better
        multiple_fit_weight = 0.75  # Slightly increased from 0.7
        
    else:
        # Use strategies based on item percentile, borrowing from parent solution structure but with some geometric elements
        if item_percentile < 0.25:  # Small items relative to remaining capacities
            # Focus on fine-grained fractional multipliers to optimize space usage
            # Use rational fractions as in parent, but add a few geometric elements
            estimated_common_sizes = [
                item, 
                item * 0.5, 
                item * 0.25, 
                item * 0.75, 
                item * 0.33, 
                item * 0.67,
                item * 1.5,
                item * 0.125,  # Very fine granularity for small items
                item * 0.875,   
                item * 0.167,   # 1/6 fraction
                item * 0.833,   # 5/6 fraction
                item * 0.2,     # 1/5 fraction
                item * 0.4,     # 2/5 fraction
                item * 0.6,     # 3/5 fraction
                item * 0.8,     # 4/5 fraction
                item * 0.1,     # Even finer granularity
                item * 0.9,     # 9/10 fraction
                item * 0.0625,  # 1/16 fraction for extremely fine granularity
                item * 0.375,   # 3/8 fraction
                item * (phi ** -1),  # Add geometric element: inverse of golden ratio
                item * (phi ** 1)    # Add geometric element: golden ratio
            ]
            
            # For small items, reduce best-fit emphasis and increase multiple fitting
            best_fit_weight = 0.003
            multiple_fit_weight = 1.17
            
        elif item_percentile > 0.75:  # Large items relative to remaining capacities
            # Focus on best-fit strategy with emphasis on avoiding fragmentation
            # Use geometric progressions for large items
            estimated_common_sizes = [item]
            # Add φ-based geometric progression
            for k in range(-1, 2):  # k from -1 to 1
                estimated_common_sizes.append(item * (phi ** k))
            # Also add sqrt2 for larger items to provide more geometric diversity
            for k in range(-1, 1):
                estimated_common_sizes.append(item * (sqrt2 ** k))
            # Add some essential rational fractions for completeness
            estimated_common_sizes.extend([
                item,
                item * 0.5,
                item * 1.5,
                item * 0.75,
                item * 1.25,
                item * 0.33,
                item * 0.67,
                item * 0.25   # Add smaller fraction for better utilization
            ])
            
            # Weight for best-fit component should be higher for large items
            best_fit_weight = 0.18  # Higher emphasis on immediate fit, adjusted from parent's 0.20
            # Reduce emphasis on multiple fitting since we prioritize filling bins
            multiple_fit_weight = 0.82  # Reduced emphasis on multiple fitting, adjusted from parent's 0.80
            
        else:  # Medium items (0.25 to 0.75 percentile) - use parent-inspired approach but with geometric elements
            # Balanced approach with moderate granularity
            estimated_common_sizes = [
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
                item * 1.1     # Slightly over 1x for near-full bins
            ]
            # Add geometric elements to enhance the parent's approach
            for k in range(-1, 2):
                estimated_common_sizes.append(item * (phi ** k))
            for k in range(-1, 1):
                estimated_common_sizes.append(item * (sqrt2 ** k))
            # Add sqrt3 for additional geometric diversity
            for k in range(-1, 1):
                estimated_common_sizes.append(item * (sqrt3 ** k))
            
            # Balanced weights from parent solution, slightly adjusted
            best_fit_weight = 0.06
            multiple_fit_weight = 0.94
    
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
    
    # Combine both components with adaptive weights
    feasible_scores = multiple_fit_weight * multiple_fit_score + best_fit_weight * best_fit_component
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores