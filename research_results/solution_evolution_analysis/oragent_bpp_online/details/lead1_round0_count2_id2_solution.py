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
    Adaptive fit heuristic: For large items, prefer tighter fits (Best Fit) to avoid wasting space.
    For small items, prefer looser bins (Worst Fit) to preserve tight bins for future large items.
    
    Implementation idea: Use a weighted combination that adapts based on the relative size of the 
    current item compared to the bin capacity. When the item is relatively large, prioritize bins 
    that provide a tight fit. When the item is relatively small, prioritize bins with more remaining 
    space to preserve tight-fitting bins for future large items.
    
    Args:
        item: Size of the current item to be placed
        bins_remain_cap: Array of remaining capacities in available bins
        
    Returns:
        Array of priority scores for each available bin (higher score means higher priority)
    """
    # Calculate how large the item is relative to a full bin (assuming standard bin capacity)
    # Using 100 as a reference capacity since that's typical for the dataset
    bin_capacity_ref = 100.0
    item_relative_size = item / bin_capacity_ref
    
    # For bins that can accommodate the item
    feasible = bins_remain_cap >= item
    
    # Initialize scores
    scores = np.full_like(bins_remain_cap, -np.inf, dtype=float)
    
    if not feasible.any():
        return scores
    
    # Calculate different priority strategies
    # Best Fit: prioritize bins with least remaining capacity (tightest fit)
    best_fit_scores = -bins_remain_cap
    
    # Worst Fit: prioritize bins with most remaining capacity (loosest fit)
    worst_fit_scores = bins_remain_cap
    
    # Adaptive strategy: blend between Best Fit and Worst Fit based on item size
    # For large items (item_relative_size closer to 1), lean towards Best Fit
    # For small items (item_relative_size closer to 0), lean towards Worst Fit
    adaptive_scores = (item_relative_size * best_fit_scores) + ((1 - item_relative_size) * worst_fit_scores)
    
    # Apply the adaptive scores only to feasible bins
    scores = np.where(feasible, adaptive_scores, -np.inf)
    
    return scores