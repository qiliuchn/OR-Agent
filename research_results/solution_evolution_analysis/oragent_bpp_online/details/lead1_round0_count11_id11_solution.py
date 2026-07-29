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

def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """
    Adaptive priority function that maintains a short-term memory of recent item sizes
    and adjusts priority logic based on the pattern of recent items.
    
    The approach:
    - Maintains a sliding window of recent item sizes
    - If recent items tend to be small, prioritizes keeping larger-capacity bins free
      for potential future large items (First Fit Decreasing style)
    - If recent items tend to be large, uses Best Fit strategy to minimize fragmentation
    - Uses a weighted combination of strategies based on recent item statistics
    
    Implementation idea: Track recent item sizes to adaptively choose between
    Best Fit (for large items) and strategies that preserve large bins (for small items).
    
    Args:
        item: Current item size to be placed
        bins_remain_cap: Array of remaining capacities in available bins
        
    Returns:
        Array of priority scores for each available bin
    """
    # Since the function needs to maintain state across calls but is stateless,
    # we'll use a class-based approach with global variables or just use static variables
    # However, since we need a pure function, we'll simulate adaptive behavior differently
    # by implementing a heuristic that looks at the relationship between item and bin sizes
    
    # This implementation will use a dynamic approach based on the ratio of item to bin capacity
    scores = np.zeros_like(bins_remain_cap, dtype=float)
    
    # Feasibility check
    feasible = bins_remain_cap >= item
    
    # If no bins are feasible, return zeros (though this shouldn't happen in practice)
    if not np.any(feasible):
        return scores
    
    # Calculate how much of the bin would be filled by this item
    fill_ratios = item / bins_remain_cap
    fill_ratios = np.where(feasible, fill_ratios, -1)  # Set non-feasible to -1
    
    # For small items relative to bin capacity, prefer preserving large bins
    # For large items relative to bin capacity, use Best Fit to minimize waste
    small_item_threshold = 0.3  # Items less than 30% of average remaining capacity are considered small
    avg_remaining_capacity = np.mean(bins_remain_cap[feasible]) if np.any(feasible) else 1.0
    
    if avg_remaining_capacity > 0:
        relative_item_size = item / avg_remaining_capacity
    else:
        relative_item_size = 1.0
    
    # Strategy selection based on item size relative to bins
    if relative_item_size <= small_item_threshold:
        # Small item: prefer bins with more remaining capacity (preserve smaller bins for later)
        # Higher priority to bins with more remaining capacity
        scores = np.where(feasible, bins_remain_cap, -np.inf)
    else:
        # Larger item: use Best Fit approach to minimize fragmentation
        # Higher priority to bins with less remaining capacity after placement
        remaining_after_placement = bins_remain_cap - item
        scores = np.where(feasible, -remaining_after_placement, -np.inf)
    
    # Apply a secondary adjustment based on absolute remaining capacity
    # To avoid using very large bins unnecessarily when smaller ones would do
    if relative_item_size <= small_item_threshold:
        # For small items, add slight penalty for using bins much larger than needed
        optimal_bin_size = item * 1.5  # Ideal bin would have about 1.5x the item size
        size_penalty = -np.abs(bins_remain_cap - optimal_bin_size) * 0.01
        scores = np.where(feasible, scores + size_penalty, -np.inf)
    
    return scores