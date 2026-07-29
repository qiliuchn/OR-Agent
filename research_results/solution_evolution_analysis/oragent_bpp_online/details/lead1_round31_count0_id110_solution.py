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
    Priority heuristic that adaptively estimates common item sizes based on quantile-based
    analysis of the current item's position relative to remaining bin capacities. The approach
    computes statistical features (percentiles) to determine if the current item is unusually 
    large or small compared to the current packing context, and selects appropriate multiplier 
    strategies accordingly. For items in higher percentiles (relatively large), it emphasizes 
    geometric progressions and sparse irrational sequences to capture packing symmetries. 
    For items in lower percentiles (relatively small), it retains dense rational fractions.
    Medium items use balanced geometric progressions. This builds on the successful 
    common-size matching paradigm while exploring geometric sequences as alternatives 
    to rational fractions.
    
    Implementation idea: Compute what percentile the current item falls into when compared 
    against the distribution of remaining bin capacities. Based on this quantile information,
    select different multiplier sets: for small items (<25th percentile), use dense rational
    fractions; for medium items (25-75th percentile), use geometric progressions with
    base sqrt(2) or phi; for large items (>75th percentile), use sparse geometric sequences
    with base 2 or phi to capture packing symmetries in Weibull-distributed item streams.
    
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
        sorted_caps = np.sort(bins_remain_cap)
        item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps)
    else:
        item_percentile = 0.5  # Default if no bins exist
    
    # Golden ratio constant
    phi = (1 + np.sqrt(5)) / 2  # Approximately 1.618
    
    # Based on the percentile, choose different strategies
    if item_percentile > 0.75:  # Item is relatively large compared to remaining capacities
        # Use sparse geometric progression with base 2 and phi for large items
        estimated_common_sizes = [item]
        
        # Powers of 2 progression (more selective to avoid excessive multipliers)
        for i in range(-2, 3):  # Covers 1/4 to 4 times the item size
            if i != 0:  # Skip original item (already added)
                size = item * (2 ** i)
                if size > 0:
                    estimated_common_sizes.append(size)
        
        # Golden ratio progression
        for i in range(-1, 2):  # phi^(-1) to phi^1 (simpler progression)
            if i != 0:  # Skip original item
                size = item * (phi ** i)
                if size > 0:
                    estimated_common_sizes.append(size)
        
        # Add some key rational fractions that are important for packing
        for factor in [0.5, 0.25, 0.75, 0.33, 0.67, 1.5, 0.2, 0.4, 0.6, 0.8]:
            size = item * factor
            if size > 0:
                estimated_common_sizes.append(size)
        
        # For large items, emphasize best-fit but maintain some focus on common-size matching
        best_fit_weight = 0.25  # Increased back to original better value
        multiple_fit_weight = 0.75  # Reduced back to original better value
        
    elif item_percentile < 0.25:  # Item is relatively small compared to remaining capacities
        # Focus on fine-grained rational fractions for small items
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
            item * 0.375,  # 3/8 fraction
            item * 0.142857, # 1/7 fraction
            item * 0.285714, # 2/7 fraction
            item * 0.428571, # 3/7 fraction
            item * 0.571429, # 4/7 fraction
            item * 0.714286, # 5/7 fraction
            item * 0.857143, # 6/7 fraction
            item * 0.03125,  # 1/32 fraction for ultra fine granularity
        ]
        
        # For small items, reduce best-fit emphasis and increase multiple fitting
        best_fit_weight = 0.002  # Return to previous better value
        multiple_fit_weight = 1.18  # Return to previous better value
        
    else:  # Item is medium-sized relative to remaining capacities (25%-75%)
        # Balanced approach with geometric progressions
        estimated_common_sizes = [item]
        
        # Geometric progression with sqrt(2) base (simplified range)
        for i in range(-2, 3):  # sqrt(2)^(-2) to sqrt(2)^2
            if i != 0:  # Skip original item
                size = item * ((np.sqrt(2)) ** i)
                if size > 0:
                    estimated_common_sizes.append(size)
        
        # Geometric progression with phi (golden ratio) base (simplified range)
        for i in range(-1, 2):  # phi^(-1) to phi^1
            if i != 0:  # Skip original item
                size = item * (phi ** i)
                if size > 0:
                    estimated_common_sizes.append(size)
        
        # Some rational fractions for balance
        for factor in [0.5, 0.25, 0.75, 0.33, 0.67, 1.5, 0.2, 0.4, 0.6, 0.8]:
            size = item * factor
            if size > 0:
                estimated_common_sizes.append(size)
        
        # Balanced weights for medium items
        best_fit_weight = 0.05  # Return to previous better value
        multiple_fit_weight = 0.95  # Return to previous better value
    
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