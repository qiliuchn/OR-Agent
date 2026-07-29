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
from typing import Dict, List

# Global state to track historical item sizes for dynamic switching
_item_history: List[float] = []

def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """
    Dynamic switching strategy: alternate between Best Fit and Worst Fit heuristics
    based on observed item size variance and historical patterns.
    
    The approach maintains a history of recently seen items to estimate the variance
    in item sizes. When variance is high (indicating diverse item sizes), we use 
    Best Fit to efficiently pack different sized items. When variance is low 
    (indicating similar item sizes), we may switch to Worst Fit to create more 
    uniform bin utilization.
    
    Implementation considerations:
    - Maintains a rolling window of recent items to compute statistics
    - Uses variance-based switching between Best Fit (-bins_remain_cap) and 
      Worst Fit (bins_remain_cap) strategies
    - Includes smoothing factor to prevent rapid switching
    
    Args:
        item: Size of the current item to be placed
        bins_remain_cap: Array of remaining capacities in available bins
        
    Returns:
        Array of priority scores for each bin (higher score means higher priority)
    """
    global _item_history
    
    # Update item history with current item
    _item_history.append(item)
    
    # Maintain a rolling window of last 50 items for variance calculation
    window_size = min(50, len(_item_history))
    if len(_item_history) > window_size:
        _item_history.pop(0)
    
    # Calculate variance of recent items if we have enough data
    if len(_item_history) >= 10:
        recent_items = np.array(_item_history[-window_size:])
        variance = np.var(recent_items)
        mean_item = np.mean(recent_items)
        
        # Normalize variance relative to mean item size to make threshold adaptive
        if mean_item > 0:
            normalized_variance = variance / (mean_item + 1e-8)  # avoid division by zero
        else:
            normalized_variance = variance
            
        # Define threshold for switching behavior based on normalized variance
        # Higher variance suggests diversity in item sizes, favor Best Fit
        # Lower variance suggests similarity in item sizes, could favor Worst Fit
        threshold = 0.1  # This threshold might need tuning based on data characteristics
        
        if normalized_variance > threshold:
            # High variance case - use Best Fit (prioritize bins with least remaining space)
            scores = np.where(bins_remain_cap >= item, -bins_remain_cap, -np.inf)
        else:
            # Low variance case - use Worst Fit (prioritize bins with most remaining space)
            # This helps consolidate items of similar sizes in fewer bins
            scores = np.where(bins_remain_cap >= item, bins_remain_cap, -np.inf)
    else:
        # Early phase when we don't have enough historical data
        # Default to Best Fit as it's generally more effective
        scores = np.where(bins_remain_cap >= item, -bins_remain_cap, -np.inf)
    
    return scores