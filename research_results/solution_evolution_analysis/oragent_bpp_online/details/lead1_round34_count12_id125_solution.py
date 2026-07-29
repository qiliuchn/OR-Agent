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
from collections import deque
from typing import Deque

def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """
    Priority heuristic implementing a minimalist, entropy-guided selection of highly relevant scales
    for the multiple-fit scoring component. Instead of the noisy combination of quantiles, item ratios,
    and remainder fractions, this approach uses only 2-3 highly relevant scales: (1) the current item
    size itself, (2) the mode of recent small item sizes estimated via a fixed-size sliding window
    stored in a static variable (exploiting Python's function attribute persistence within a single
    evaluation run), and (3) the dominant scale in the current remainder distribution identified via
    peak detection in a lightweight kernel density estimate. The recyclability metric is redefined as
    the proximity of the post-placement remainder to the sliding-window-estimated small-item mode,
    enabling a principled online approximation of future utility without violating the function
    signature constraints.
    
    Implementation idea: Replace the complex dynamic common size generation with a minimalist
    approach using only the current item size, estimated mode of recent small items via sliding
    window, and dominant scale in remainder distribution. This simplifies the multiple-fit signal
    while making it more semantically meaningful and directly addresses the parent's finding that
    'common size generation introduces noise rather than signal'.
    
    Args:
        item: Size of the item to place
        bins_remain_cap: NumPy array of remaining bin capacities
        
    Returns:
        NumPy array of priority scores for each bin
    """
    # Initialize static storage for the sliding window of recent items if it doesn't exist
    if not hasattr(priority, 'recent_items'):
        priority.recent_items = deque(maxlen=50)  # Fixed-size sliding window
    
    # Add current item to the sliding window
    priority.recent_items.append(item)
    
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
        best_fit_weight, multiple_fit_weight, recyclability_weight = 0.25 * 1.07, 0.75 * 0.99, 0.05
    elif item_percentile < 0.18:  # Very small item
        best_fit_weight, multiple_fit_weight, recyclability_weight = 0.001 * 0.93, 1.194 * 1.01, 0.1
    elif item_percentile > 0.62:  # Large-medium item
        best_fit_weight, multiple_fit_weight, recyclability_weight = 0.14, 0.86, 0.05
    elif item_percentile < 0.38:  # Small-medium item
        best_fit_weight, multiple_fit_weight, recyclability_weight = 0.025, 1.055, 0.08
    else:  # True medium item
        best_fit_weight, multiple_fit_weight, recyclability_weight = 0.08, 0.92, 0.05
    
    # Generate minimal set of relevant common sizes
    # 1. Current item size
    common_sizes = [item]
    
    # 2. Median of recent small items from the sliding window as proxy for mode
    if len(priority.recent_items) > 0:
        recent_array = np.array(priority.recent_items)
        if len(recent_array) >= 4:
            # Use the median of the smallest half as proxy for small item mode
            sorted_recent = np.sort(recent_array)
            # Take the median of the first half as proxy for small item mode
            half_size = max(1, len(sorted_recent) // 2)
            small_half = sorted_recent[:half_size]
            if len(small_half) > 0:
                small_item_mode = np.median(small_half)
                if small_item_mode > 1e-9:  # Only add if not too small
                    common_sizes.append(small_item_mode)
    
    # 3. Dominant scale in current remainder distribution (simple median)
    if len(feasible_post_caps) > 1:
        # Use median as a robust measure of central tendency for the dominant scale
        remainder_median = np.median(feasible_post_caps)
        if remainder_median > 1e-9:
            common_sizes.append(remainder_median)
    
    # Convert to numpy array
    common_sizes = np.array(common_sizes)
    
    # Calculate multiple fit score based on nearest multiple of minimal common sizes
    multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    for common_size in common_sizes:
        if common_size > 1e-9:  # Avoid division by very small numbers
            # Calculate distance to nearest multiple of the common size
            # Find the nearest multiplier
            multipliers = np.round(feasible_post_caps / common_size)
            nearest_multiples = multipliers * common_size
            distances_to_nearest_multiple = np.abs(feasible_post_caps - nearest_multiples)
            
            # Add to the score (higher score for closer matches)
            multiple_fit_score += 1.0 / (1.0 + distances_to_nearest_multiple)
    
    # Add Best Fit component: prefer bins with less remaining space
    best_fit_component = -feasible_post_caps  # Higher score for less remaining space
    
    # Calculate recyclability metric based on proximity to small item mode
    recyclability_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    # Get the small item mode from the sliding window if available
    if len(priority.recent_items) >= 4:
        recent_array = np.array(priority.recent_items)
        sorted_recent = np.sort(recent_array)
        half_size = max(1, len(sorted_recent) // 2)
        small_half = sorted_recent[:half_size]
        if len(small_half) > 0:
            small_item_mode = np.median(small_half)
            if small_item_mode > 1e-9:
                # Calculate distance to the small item mode
                distances_to_small_mode = np.abs(feasible_post_caps - small_item_mode)
                # Higher recyclability score for remainders closer to common small item sizes
                recyclability_score = 1.0 / (1.0 + distances_to_small_mode)
    
    # Adjust weights based on entropy of remainder distribution
    if len(feasible_post_caps) > 1:
        # Calculate entropy-like measure of the distribution
        remainder_probs = feasible_post_caps / np.sum(feasible_post_caps)
        remainder_probs = remainder_probs[remainder_probs > 0]  # Remove zeros
        if len(remainder_probs) > 0:
            entropy = -np.sum(remainder_probs * np.log(remainder_probs + 1e-9))
            # Normalize entropy by max possible entropy for this number of bins
            max_entropy = np.log(len(remainder_probs)) if len(remainder_probs) > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.5
            
            # Adjust weights based on entropy
            # Higher entropy (more uniform distribution) may benefit from more recyclability consideration
            recyclability_weight *= (0.5 + 0.5 * normalized_entropy)
            multiple_fit_weight *= (1.0 - 0.2 * normalized_entropy)  # Slightly reduce if highly entropic
    
    # Combine all components with category-specific weights
    feasible_scores = (
        multiple_fit_weight * multiple_fit_score + 
        best_fit_weight * best_fit_component +
        recyclability_weight * recyclability_score
    )
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores