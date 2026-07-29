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
    Efficient adaptive priority function that combines best-fit and multiple-based scoring
    with dynamic weighting based on item characteristics.
    
    The algorithm calculates two main components:
    1. Best-fit scoring: favors bins with minimum remaining space after placement (to reduce waste)
    2. Multiple-based scoring: favors bins whose post-placement capacity aligns with multiples of common item sizes
    
    Dynamic weights are computed based on the item-to-capacity ratio to adapt to different item sizes.
    
    Args:
        item: Size of the item to place
        bins_remain_cap: NumPy array of remaining bin capacities
        
    Returns:
        NumPy array of priority scores for each bin
    """
    # Calculate relevant state features
    if len(bins_remain_cap) > 0:
        max_capacity = np.max(bins_remain_cap) if np.any(bins_remain_cap > 0) else item * 2
    else:
        max_capacity = item * 2
    
    # Feature: Item size relative to bin capacity (detects large vs small items)
    item_capacity_ratio = item / max_capacity
    
    # Calculate post-placement remaining capacities
    post_placement_caps = bins_remain_cap - item
    
    # Initialize scores
    scores = np.full_like(bins_remain_cap, -np.inf, dtype=float)
    
    # Identify feasible bins (those that can accommodate the item)
    feasible_bins = bins_remain_cap >= item
    
    if not np.any(feasible_bins):
        return scores  # All bins remain with -inf scores
    
    # Extract feasible capacities
    feasible_post_caps = post_placement_caps[feasible_bins]
    
    # Calculate original bin capacities before placement (for diversity component)
    original_feasible_caps = bins_remain_cap[feasible_bins]
    
    # Component 1: Best-fit scoring (higher score for less remaining space after placement)
    best_fit_scores = -feasible_post_caps  # Higher score for less remaining space
    
    # Component 2: Multiple-based scoring (enhanced version)
    # Use a broader range of estimated common sizes based on the parent solution approach
    estimated_common_sizes = [item, item * 0.9, item * 1.1, item * 0.5, item * 1.5, item * 0.75, item * 1.25]
    multiple_scores = np.zeros_like(feasible_post_caps, dtype=float)
    
    for common_size in estimated_common_sizes:
        if common_size > 0:
            # Calculate distance to nearest multiple
            quotients = feasible_post_caps / common_size
            rounded_quotients = np.round(quotients)
            distances_to_multiples = np.abs(feasible_post_caps - rounded_quotients * common_size)
            # Add to the score (higher score for closer matches)
            multiple_scores += 1.0 / (1.0 + distances_to_multiples)
    
    # Dynamic weight calculation based on item characteristics
    # For large items (item_capacity_ratio > 0.5), prioritize best-fit to avoid fragmentation
    if item_capacity_ratio > 0.5:
        w_bestfit = 0.8
        w_multiple = 0.2
    # For medium items (0.2 < item_capacity_ratio <= 0.5), balance components
    elif item_capacity_ratio > 0.2:
        w_bestfit = 0.6
        w_multiple = 0.4
    # For small items, prioritize multiple-based scoring to reduce fragmentation
    else:
        w_bestfit = 0.3
        w_multiple = 0.7
    
    # Normalize components to similar scales before combining
    def safe_normalize(arr):
        std_val = np.std(arr)
        if std_val > 0:
            mean_val = np.mean(arr)
            return (arr - mean_val) / (std_val + 1e-8)
        return arr - np.mean(arr)
    
    best_fit_scores_norm = safe_normalize(best_fit_scores)
    multiple_scores_norm = safe_normalize(multiple_scores)
    
    # Combine components with adaptive weights
    combined_scores = w_bestfit * best_fit_scores_norm + w_multiple * multiple_scores_norm
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = combined_scores
    
    return scores