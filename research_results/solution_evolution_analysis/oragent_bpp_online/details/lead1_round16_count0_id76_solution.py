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

# Global variables to maintain state for the EMA of recent item sizes
_recent_items_ema = None
_decay_factor = 0.9  # Decay factor for the EMA - more responsive to changes

def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """
    Implementation idea:
    This function implements a state-augmented priority function that maintains an exponential
    moving average (EMA) of recent item sizes to better estimate the distribution of upcoming
    items. The EMA allows the algorithm to adapt to changes in item size patterns over time,
    giving more weight to recent observations. Based on this estimated distribution, the
    function scores bins by how well their post-placement capacity matches high-probability
    regions of the estimated item size distribution. This improves upon parent solutions by
    having a more adaptive understanding of common item sizes rather than just relying on
    the current item as a proxy for all future items.
    
    The approach combines:
    1. Maintaining an EMA of recent item sizes to estimate the item size distribution
    2. Using this estimate to identify common item sizes
    3. Prioritizing bins whose post-placement capacity matches these common sizes
    4. Including a Best Fit component to prevent waste
    
    Args:
        item: Size of the item to place
        bins_remain_cap: NumPy array of remaining bin capacities
        
    Returns:
        NumPy array of priority scores for each bin
    """
    global _recent_items_ema
    
    # Update the EMA with the current item
    if _recent_items_ema is None:
        # Initialize with the first item
        _recent_items_ema = item
    else:
        # Update using exponential moving average formula
        _recent_items_ema = _decay_factor * _recent_items_ema + (1 - _decay_factor) * item
    
    # Estimate common item sizes based on the EMA of recent items
    # Using the EMA as the primary reference point for common sizes
    ema_based_size = _recent_items_ema
    
    # Generate common sizes based on the EMA of recent items
    # Focus primarily on EMA-based estimates to reduce noise from current item
    estimated_common_sizes = [
        ema_based_size,                    # EMA-based size
        ema_based_size * 0.5,              # half of EMA size
        ema_based_size * 0.75,             # three quarters of EMA size
        ema_based_size * 0.9,              # 90% of EMA size
        ema_based_size * 1.1,              # 110% of EMA size
        ema_based_size * 1.25,             # 1.25x of EMA size
        ema_based_size * 1.5,              # 1.5x of EMA size
        ema_based_size * 0.25,             # quarter of EMA size
        ema_based_size * 2.0,              # double of EMA size
    ]
    
    # Filter out non-positive sizes
    estimated_common_sizes = [size for size in estimated_common_sizes if size > 0]
    
    # Calculate post-placement remaining capacities
    post_placement_caps = bins_remain_cap - item
    
    # Initialize scores
    scores = np.full_like(bins_remain_cap, -np.inf, dtype=float)
    
    # Only consider bins that can accommodate the item
    feasible_bins = bins_remain_cap >= item
    
    if not np.any(feasible_bins):
        return scores  # All bins remain with -inf scores
    
    # For feasible bins, calculate priority based on how close the remaining capacity is to common sizes
    feasible_post_caps = post_placement_caps[feasible_bins]
    feasible_scores = np.zeros_like(feasible_post_caps, dtype=float)
    
    # For each estimated common size, calculate how well the remaining capacity matches
    for common_size in estimated_common_sizes:
        # Calculate how close the remaining capacity is to being a multiple of the common size
        # Consider multiples: 0, 1x, 2x, 3x, etc. of the common size
        max_multiple = int(np.max(feasible_post_caps) // common_size) + 2
        distances_to_multiples = np.min([
            np.abs(feasible_post_caps - n * common_size) 
            for n in range(0, max_multiple)
        ], axis=0)
        
        # Add to the score (higher score for closer matches to multiples)
        # Use a function that gives significant scores only for close matches
        feasible_scores += 1.0 / (1.0 + distances_to_multiples)
    
    # Also add Best Fit component: prefer bins with less remaining space after placement
    # This prevents overfilling bins unnecessarily while still allowing for good fits
    best_fit_component = -post_placement_caps  # Higher score for less remaining space after placement
    
    # Combine both components with appropriate weights
    # Give more weight to the common size matching since it was the key insight from parent solution
    combined_scores = feasible_scores + 0.05 * best_fit_component  # Weight found in previous experiments
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = combined_scores
    
    return scores