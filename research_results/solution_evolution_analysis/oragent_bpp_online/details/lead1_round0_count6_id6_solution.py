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
    Hybrid Best Fit with positional bias: prioritize bins with smallest remaining capacity 
    that can still fit the item, but among equally good candidates, prefer earlier opened bins
    to reduce fragmentation from scattered partial bins.
    
    This approach combines the Best Fit strategy (minimizing wasted space in bins) with a 
    positional component that favors earlier bins, helping to consolidate usage toward the 
    beginning of the bin sequence and potentially reducing overall bin count.
    
    Args:
        item: Size of the item to place (float)
        bins_remain_cap: NumPy array of remaining bin capacities (float array)
    
    Returns:
        NumPy array of priority scores for each bin (higher score means higher priority)
    """
    # Create position weights that decrease with bin index to favor earlier bins
    n_bins = len(bins_remain_cap)
    position_weights = np.arange(n_bins, 0, -1).astype(float)  # [n, n-1, ..., 2, 1]
    
    # Normalize position weights to prevent them from dominating capacity-based scoring
    position_weights = position_weights / n_bins
    
    # Determine which bins can fit the item
    feasible = bins_remain_cap >= item
    
    # Initialize scores array
    scores = np.full_like(bins_remain_cap, -np.inf, dtype=float)
    
    # For feasible bins, calculate priority based on remaining capacity (inverted) plus position bonus
    # Using -bins_remain_cap prioritizes bins with less remaining space (Best Fit)
    # Adding normalized position weights prefers earlier bins when capacities are similar
    scores = np.where(feasible, -bins_remain_cap + position_weights, -np.inf)
    
    return scores