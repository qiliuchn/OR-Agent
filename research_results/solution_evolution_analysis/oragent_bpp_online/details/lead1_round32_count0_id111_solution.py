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
    Priority heuristic that adaptively estimates common item sizes based on dynamically-adjusted
    percentile thresholds that respond to the current distribution of remaining bin capacities.
    Instead of fixed 25%/75% thresholds, the algorithm calculates adaptive thresholds based on
    the statistical properties of remaining bin capacities, specifically using the interquartile
    range (IQR) to detect shifts in packing context and adjust categorization sensitivity.
    The approach computes statistical features (percentiles) to determine if the current item
    is unusually large or small compared to the current packing context, and selects appropriate
    multiplier strategies accordingly. For items in higher percentiles (relatively large),
    it emphasizes geometric progressions and sparse irrational sequences to capture packing
    symmetries. For items in lower percentiles (relatively small), it retains dense rational
    fractions. Medium items use balanced geometric progressions. This builds on the successful
    common-size matching paradigm while exploring geometric sequences as alternatives to
    rational fractions, but with dynamically adjusted thresholds that respond to the current
    packing situation.
    
    Implementation idea: Calculate dynamic percentile thresholds based on the IQR of remaining
    bin capacities. When the IQR is large (indicating varied remaining capacities), use wider
    thresholds to allow more flexible categorization. When the IQR is small (indicating similar
    remaining capacities), use tighter thresholds to make categorization more sensitive.
    The thresholds are calculated as Q1 - k*IQR and Q3 + k*IQR for some scaling factor k,
    clamped to reasonable bounds to prevent extreme values. Then, based on the dynamic
    percentile position of the current item, select different multiplier sets: for small items,
    use dense rational fractions; for medium items, use geometric progressions with sqrt(2)
    or phi; for large items, use sparse geometric sequences with base 2 or phi.
    
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
    
    # Calculate dynamic thresholds based on IQR of remaining bin capacities
    if len(bins_remain_cap) >= 4:  # Need at least 4 values to calculate meaningful IQR
        q1_idx = len(bins_remain_cap) // 4
        q3_idx = 3 * len(bins_remain_cap) // 4
        q1 = sorted_caps[q1_idx]
        q3 = sorted_caps[q3_idx]
        iqr = q3 - q1
        
        # Calculate dynamic thresholds using IQR
        # Base thresholds around median with IQR-based adjustment
        median_idx = len(bins_remain_cap) // 2
        median_val = sorted_caps[median_idx]
        
        # Adjust thresholds based on IQR magnitude relative to median
        if median_val > 0:
            iqr_ratio = iqr / median_val
            # Scale the threshold adjustments based on IQR ratio
            # Higher IQR leads to wider thresholds (less sensitive)
            # Lower IQR leads to tighter thresholds (more sensitive)
            threshold_adjustment = 0.25 * iqr_ratio  # Increased adjustment factor for more responsiveness
            low_threshold = 0.25 - threshold_adjustment
            high_threshold = 0.75 + threshold_adjustment
            
            # Clamp thresholds to reasonable bounds to prevent extreme values
            low_threshold = max(0.05, min(0.45, low_threshold))  # Wider range
            high_threshold = max(0.55, min(0.95, high_threshold))  # Wider range
        else:
            # Default thresholds if median is zero
            low_threshold = 0.25
            high_threshold = 0.75
    else:
        # Default thresholds if insufficient data for IQR calculation
        low_threshold = 0.25
        high_threshold = 0.75
    
    # Golden ratio constant
    phi = (1 + np.sqrt(5)) / 2  # Approximately 1.618
    
    # Based on the dynamic percentile, choose different strategies
    if item_percentile > high_threshold:  # Item is relatively large compared to remaining capacities
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
        best_fit_weight = 0.25
        multiple_fit_weight = 0.75
        
    elif item_percentile < low_threshold:  # Item is relatively small compared to remaining capacities
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
            item * 0.375,   # 3/8 fraction
            item * 0.142857, # 1/7 fraction
            item * 0.285714, # 2/7 fraction
            item * 0.428571, # 3/7 fraction
            item * 0.571429, # 4/7 fraction
            item * 0.714286, # 5/7 fraction
            item * 0.857143, # 6/7 fraction
            item * 0.03125,  # 1/32 fraction for ultra fine granularity
        ]
        
        # For small items, reduce best-fit emphasis and increase multiple fitting
        best_fit_weight = 0.002
        multiple_fit_weight = 1.18
        
    else:  # Item is medium-sized relative to remaining capacities (between thresholds)
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
        best_fit_weight = 0.05
        multiple_fit_weight = 0.95
    
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