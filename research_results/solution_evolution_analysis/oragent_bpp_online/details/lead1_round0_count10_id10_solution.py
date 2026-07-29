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
    Penalize bins that would leave 'unusable' residual space: define a threshold (e.g., below 0.1 capacity) 
    and assign low priority to assignments that create such waste. This builds upon the Best Fit heuristic 
    by adding a penalty for creating small unusable spaces in bins.

    Args:
        item: Size of the item to place (float)
        bins_remain_cap: NumPy array of remaining bin capacities (float array)

    Returns:
        NumPy array of priority scores for each bin
    """
    # Calculate what the remaining capacity would be after placing the item
    new_remaining_cap = bins_remain_cap - item
    
    # Define a threshold for "unusable" space (10% of original capacity seems reasonable)
    # Since we don't have the original capacity, we'll use a relative approach
    # For now, let's consider anything < 0.1 as potentially unusable
    unusable_threshold = 0.1
    
    # Start with Best Fit heuristic: prioritize bins with smallest remaining capacity
    # Higher priority for bins that will have less space left after placement
    base_scores = -new_remaining_cap
    
    # Apply penalty to bins that would create unusable residual space
    # If the new remaining capacity is below our threshold, penalize heavily
    unusable_penalty = np.where(new_remaining_cap < unusable_threshold, -1000, 0)
    
    # Combine base score with penalty
    scores = base_scores + unusable_penalty
    
    # Ensure infeasible bins (don't have enough space) get very low priority
    feasible = bins_remain_cap >= item
    final_scores = np.where(feasible, scores, -np.inf)
    
    return final_scores