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
    analysis of the current item's position relative to remaining bin capacities, combined with
    a piecewise non-linear scoring function that uses exponential decay for small gaps and
    linear decay for larger gaps. The approach computes statistical features (percentiles) 
    to determine if the current item is unusually large or small compared to the current 
    packing context, and selects appropriate multiplier strategies accordingly. The scoring
    function combines exponential decay for small gaps (to strongly favor near-exact matches) 
    and linear decay for moderate gaps, improving discrimination among high-utility residual 
    spaces while maintaining numerical stability.

    Implementation idea: Replace the linear inverse-distance weighting (1/(1+distance)) with 
    a piecewise non-linear scoring function that combines exponential decay for small gaps 
    (to strongly favor near-exact matches) and a softened penalty for moderate gaps, while 
    maintaining numerical stability. Define a threshold τ (e.g., 5% of item size) such that 
    distances ≤ τ use exp(-α·distance) with α tuned per item category, and distances > τ 
    use a dampened linear decay. This hybrid approach preserves the robustness of linear 
    scoring for larger mismatches while enhancing discrimination among high-utility residual 
    spaces.

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
    # This gives us a measure of how large the current item is relative to current context
    if len(bins_remain_cap) > 0:
        # Calculate the percentile of the current item in the context of remaining capacities
        # This is essentially the proportion of bins that have less remaining capacity than the current item
        sorted_caps = np.sort(bins_remain_cap)
        # Find where the item would fit in the sorted array
        item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps)
    else:
        item_percentile = 0.5  # Default if no bins exist
    
    # Based on the percentile, choose different strategies
    if item_percentile > 0.75:  # Item is relatively large compared to remaining capacities
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
        # Exponential decay parameter for small gaps
        alpha = 0.5
        
    elif item_percentile < 0.25:  # Item is relatively small compared to remaining capacities
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
        # Exponential decay parameter for small gaps
        alpha = 0.1  # Less aggressive for small items
        
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
        
        # Balanced weights
        best_fit_weight = 0.06
        multiple_fit_weight = 0.94
        # Exponential decay parameter for small gaps
        alpha = 0.3  # Medium aggressiveness
        
    # Filter out non-positive sizes
    estimated_common_sizes = [size for size in estimated_common_sizes if size > 0]
    
    # Calculate how close the remaining capacity is to being a multiple of common sizes
    # Using piecewise non-linear scoring function: exponential for small gaps, linear for larger gaps
    multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    for common_size in estimated_common_sizes:
        # Calculate how many multiples of common_size fit in the remaining capacity
        max_multiplier = int(np.max(feasible_post_caps) // common_size) + 2
        if max_multiplier > 0:
            # Calculate distances to all possible multiples
            distances_to_multiples_list = []
            for n in range(0, max_multiplier):
                distances_to_multiples_list.append(np.abs(feasible_post_caps - n * common_size))
            
            if distances_to_multiples_list:
                distances_to_multiples = np.min(distances_to_multiples_list, axis=0)
                
                # Define threshold as 5% of the common size
                tau = 0.05 * common_size
                
                # Piecewise scoring function
                # For distances <= tau, use exponential decay: exp(-alpha * distance)
                # For distances > tau, use linear decay: 1 / (1 + (distance - tau) + exp(-alpha * tau) * (1 + tau))
                # Actually, let's simplify: exponential for small gaps, linear continuation for large gaps
                small_gap_mask = distances_to_multiples <= tau
                
                # Calculate scores for small gaps (exponential decay)
                small_gap_scores = np.exp(-alpha * distances_to_multiples[small_gap_mask])
                
                # Calculate scores for large gaps (linear decay starting from the value at tau boundary)
                # To ensure continuity, start from the exponential value at tau and apply a shifted linear decay
                exp_at_tau = np.exp(-alpha * tau)
                large_gap_scores = exp_at_tau / (1.0 + (distances_to_multiples[~small_gap_mask] - tau))
                
                # Create full scores array
                temp_scores = np.zeros_like(distances_to_multiples, dtype=float)
                temp_scores[small_gap_mask] = small_gap_scores
                temp_scores[~small_gap_mask] = large_gap_scores
                
                # Add to the total score
                multiple_fit_score += temp_scores
    
    # Add Best Fit component: prefer bins with less remaining space
    # This prevents overfilling bins unnecessarily
    best_fit_component = -feasible_post_caps  # Higher score for less remaining space
    
    # Combine both components with adaptive weights
    feasible_scores = multiple_fit_weight * multiple_fit_score + best_fit_weight * best_fit_component
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores