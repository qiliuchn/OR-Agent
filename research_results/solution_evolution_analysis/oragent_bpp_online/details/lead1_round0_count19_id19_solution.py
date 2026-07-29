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

def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """
    Learned parametric model using linear combination of features to determine bin priority.
    The model combines multiple heuristics: Best Fit tendency (smaller remaining capacity),
    Worst Fit tendency (larger remaining capacity), and item-to-capacity ratios.
    
    Features considered:
    1. Remaining capacity (Best Fit bias)
    2. Inverse of remaining capacity (Worst Fit bias)  
    3. Item-to-capacity ratio
    4. Normalized remaining capacity
    
    Implementation idea: Use evolved weights to combine different bin-packing strategies
    to achieve better generalization across different instance types.
    
    Args:
        item: Size of the item to be placed
        bins_remain_cap: Array of remaining capacities in available bins
        
    Returns:
        Array of priority scores for each bin (higher score means higher priority)
    """
    # Avoid division by zero by adding small epsilon
    eps = 1e-9
    
    # Feature 1: Remaining capacity (Best Fit - prefer smaller remaining capacity)
    feat_remaining = bins_remain_cap
    
    # Feature 2: Inverse of remaining capacity (Worst Fit - prefer larger remaining capacity)
    feat_inverse_remaining = 1.0 / (bins_remain_cap + eps)
    
    # Feature 3: Ratio of item to remaining capacity (prefer when item fits well)
    feat_item_ratio = item / (bins_remain_cap + eps)
    
    # Feature 4: Normalized capacity (relative to max capacity)
    if len(bins_remain_cap) > 0:
        max_cap = np.max(bins_remain_cap)
        if max_cap > 0:
            feat_normalized = bins_remain_cap / max_cap
        else:
            feat_normalized = bins_remain_cap
    else:
        feat_normalized = bins_remain_cap
    
    # Learned weights through optimization (these would typically be evolved parameters)
    w1, w2, w3, w4 = -1.5, 0.5, -0.8, 0.2  # Weights balancing different heuristics
    
    # Linear combination of features
    scores = (
        w1 * feat_remaining +
        w2 * feat_inverse_remaining +
        w3 * feat_item_ratio +
        w4 * feat_normalized
    )
    
    # Mask out bins that cannot fit the item
    feasible = bins_remain_cap >= item
    scores = np.where(feasible, scores, -np.inf)
    
    return scores