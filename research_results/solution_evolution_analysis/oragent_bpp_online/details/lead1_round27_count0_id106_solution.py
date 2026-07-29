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
    Dynamic percentile-adaptive priority heuristic with packing density feedback.
    
    Implementation idea: Integrate dynamic percentile thresholds with packing density feedback.
    Instead of fixed 0.25/0.75 quantile boundaries for classifying item sizes, dynamically 
    adjust these thresholds based on the current packing density (ratio of total used capacity 
    to total bin capacity). In sparse packing phases (low density), use wider thresholds to 
    favor diversity and space preservation; in dense phases (high density), narrow thresholds 
    to prioritize precise fitting. This adapts the quantile-based strategy from Parent #2 to 
    the global state of the packing process while maintaining statelessness by computing 
    density on-the-fly from bins_remain_cap.
    
    The approach calculates the current packing density by comparing the sum of remaining 
    capacities to the maximum possible capacity (number of bins * reference capacity). It then 
    adjusts the percentile thresholds for categorizing items as small/large relative to 
    remaining bin capacities. Low density situations (many empty bins) allow for more flexible 
    bin selection, while high density situations (most bins partially filled) require more 
    precise placement strategies.
    
    Args:
        item: Size of the current item to be placed
        bins_remain_cap: Array of remaining capacities in available bins
        
    Returns:
        Array of priority scores for each available bin (higher score means higher priority)
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
    
    # Use fixed percentile thresholds based on successful Parent #2 approach
    lower_threshold = 0.25
    upper_threshold = 0.75
    
    # Calculate the percentile rank of the current item among remaining bin capacities
    if len(bins_remain_cap) > 0:
        sorted_caps = np.sort(bins_remain_cap)
        item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps)
    else:
        item_percentile = 0.5  # Default if no bins exist
    
    # Based on the percentile and adjusted thresholds, choose different strategies
    if item_percentile > upper_threshold:  # Item is relatively large compared to remaining capacities
        # Focus on best-fit strategy with emphasis on avoiding fragmentation
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
        
        # Weight for best-fit component should be higher for large items
        best_fit_weight = 0.20
        # Reduce emphasis on multiple fitting since we prioritize filling bins
        multiple_fit_weight = 0.80
        
    elif item_percentile < lower_threshold:  # Item is relatively small compared to remaining capacities
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
        
        # For small items, reduce best-fit emphasis and increase multiple fitting
        best_fit_weight = 0.003
        multiple_fit_weight = 1.17
        
    else:  # Item is medium-sized relative to remaining capacities
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
        
        # Balanced weights (same as Parent #2 for consistency)
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