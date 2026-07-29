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
    Priority heuristic that adaptively estimates common item sizes based on dynamic
    quantile-based analysis of the current item's position relative to remaining 
    bin capacities. The approach computes statistical features (adaptive percentiles) 
    to determine if the current item is unusually large or small compared to recent 
    packing context. Instead of fixed quartiles, this implementation uses the empirical
    cumulative distribution function (ECDF) of current bin remainders to define
    adaptive boundaries based on the actual fragmentation state of the system.
    
    For example, in highly fragmented scenarios with many small remainders, even
    moderately sized items may be treated as 'large'. This allows the algorithm to
    automatically adjust its notion of 'small', 'medium', and 'large' items according
    to the current packing state, improving adaptivity to different fragmentation levels.
    
    Implementation based on the successful parent solution but with dynamic thresholds.
    
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
    
    # Compute adaptive thresholds based on ECDF of current bin remainders
    if len(bins_remain_cap) > 0:
        # Sort the remaining capacities to build ECDF
        sorted_caps = np.sort(bins_remain_cap[bins_remain_cap > 0])  # Only consider non-empty bins
        
        if len(sorted_caps) > 0:
            # Determine dynamic thresholds based on the distribution of remainders
            # Calculate where the current item fits in the ECDF of current remainders
            item_position = np.searchsorted(sorted_caps, item)
            item_percentile = item_position / len(sorted_caps) if len(sorted_caps) > 0 else 0.5
            
            # Calculate adaptive thresholds based on the entropy or spread of the current distribution
            # For a more adaptive approach, we can adjust thresholds based on how spread out the remainders are
            if len(sorted_caps) >= 4:  # Need enough samples for meaningful statistics
                # Measure the spread of the distribution using interquartile range or standard deviation
                cap_std = np.std(sorted_caps)
                cap_range = np.max(sorted_caps) - np.min(sorted_caps) if len(sorted_caps) > 1 else 0
                
                # Adjust thresholds based on the fragmentation level
                # When bins are very fragmented (high std/range), we might want different thresholds
                fragmentation_level = cap_std / (np.mean(sorted_caps) + 1e-8)  # Normalize by mean to get coefficient of variation
                
                # Adjust thresholds based on fragmentation level
                # In highly fragmented states, we might want to be more aggressive about using nearly full bins
                if fragmentation_level > 0.5:  # High fragmentation
                    low_threshold = 0.2  # More sensitive to "small" items
                    high_threshold = 0.8  # Less sensitive to "large" items
                elif fragmentation_level < 0.2:  # Low fragmentation
                    low_threshold = 0.25  # More conservative
                    high_threshold = 0.75  # More conservative
                else:  # Medium fragmentation
                    low_threshold = 0.25
                    high_threshold = 0.75
                
                # Use these adaptive thresholds to classify the current item
                if item_percentile > high_threshold:
                    item_category = "large"
                elif item_percentile < low_threshold:
                    item_category = "small"
                else:
                    item_category = "medium"
            else:
                # Fallback to simple percentile if not enough samples
                if item_percentile > 0.75:
                    item_category = "large"
                elif item_percentile < 0.25:
                    item_category = "small"
                else:
                    item_category = "medium"
        else:
            item_category = "medium"  # Default if no bins with capacity
    else:
        item_category = "medium"  # Default if no bins exist
    
    # Define multiplier sets based on the adaptive categorization
    if item_category == "large":
        # Focus on fine-grained fractional multipliers to optimize space usage for large items
        estimated_common_sizes = [
            item,
            item * 0.5,
            item * 1.5,
            item * 0.75,
            item * 1.25,
            item * 0.33,
            item * 0.67,
            item * 2.0,
            item * 0.25   # Add smaller fraction for better utilization
        ]
        
    elif item_category == "small":
        # Focus on fine-grained fractional multipliers to optimize space usage
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
            item * 0.375    # 3/8 fraction
        ]
        
    else:  # item_category == "medium"
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

    # Now calculate adaptive weights based on the current fragmentation state
    if len(bins_remain_cap) > 0 and np.any(feasible_bins):
        # Calculate fragmentation metrics based on the current state
        feasible_caps = bins_remain_cap[feasible_bins]
        if len(feasible_caps) > 1:
            cap_std = np.std(feasible_caps)
            cap_mean = np.mean(feasible_caps)
            fragmentation_level = cap_std / (cap_mean + 1e-8)
            
            # Adjust weights based on fragmentation level and item category
            if item_category == "large":
                # For large items in fragmented bins, emphasize best-fit to consolidate
                if fragmentation_level > 0.5:  # High fragmentation
                    best_fit_weight = 0.25  # Slightly higher than before
                    multiple_fit_weight = 0.75  # Slightly lower than before
                else:  # Low fragmentation
                    best_fit_weight = 0.20
                    multiple_fit_weight = 0.80
            elif item_category == "small":
                # For small items, adjust based on fragmentation
                if fragmentation_level > 0.5:  # High fragmentation
                    best_fit_weight = 0.005  # Slightly higher than before to fill gaps
                    multiple_fit_weight = 1.15  # Slightly lower than before
                else:  # Low fragmentation
                    best_fit_weight = 0.003
                    multiple_fit_weight = 1.17
            else:  # medium
                # Balanced adjustment for medium items
                if fragmentation_level > 0.5:  # High fragmentation
                    best_fit_weight = 0.08  # Slightly higher to consolidate
                    multiple_fit_weight = 0.92  # Slightly lower
                else:  # Low fragmentation
                    best_fit_weight = 0.06
                    multiple_fit_weight = 0.94
        else:
            # Fallback to original weights if insufficient data
            if item_category == "large":
                best_fit_weight = 0.20
                multiple_fit_weight = 0.80
            elif item_category == "small":
                best_fit_weight = 0.003
                multiple_fit_weight = 1.17
            else:  # medium
                best_fit_weight = 0.06
                multiple_fit_weight = 0.94
    else:
        # Fallback to original weights
        if item_category == "large":
            best_fit_weight = 0.20
            multiple_fit_weight = 0.80
        elif item_category == "small":
            best_fit_weight = 0.003
            multiple_fit_weight = 1.17
        else:  # medium
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