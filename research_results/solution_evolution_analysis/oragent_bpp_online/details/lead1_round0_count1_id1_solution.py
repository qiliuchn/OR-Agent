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
    Balanced heuristic: introduce a tunable parameter to balance between Best Fit (minimizing leftover space) 
    and Worst Fit (maximizing leftover space for future flexibility).
    
    The approach uses a weighted combination of remaining capacity and additional factors to achieve better
    performance than pure Best Fit. This builds on the parent solution's insight that prioritizing bins
    with less remaining capacity helps reduce fragmentation, but also considers leaving some flexibility
    for future items by not over-packing bins.
    
    Args:
        item: Size of the item to be placed
        bins_remain_cap: Array of remaining capacities in available bins
        
    Returns:
        Array of priority scores for each bin (higher score means higher priority)
    """
    # Initialize scores array
    scores = np.zeros_like(bins_remain_cap, dtype=np.float64)
    
    # Identify feasible bins (those that can accommodate the item)
    feasible = bins_remain_cap >= item
    
    # Calculate priority scores only for feasible bins
    # Use a balanced approach between Best Fit and keeping some flexibility
    # We'll use: score = -(remaining_capacity_after_placement)^weight
    # where the weight balances tight packing vs flexibility
    remaining_after_placement = bins_remain_cap - item
    
    # Use a transformation that favors slightly more filled bins but not overly tight fits
    # Apply a weighted scoring that prefers moderately-filled bins over both very full or very empty ones
    scores = np.where(feasible, 
                      -remaining_after_placement * 0.8 - 0.2 * remaining_after_placement**0.9, 
                      -np.inf)
    
    return scores