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
from typing import Union

def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """
    Priority heuristic incorporating entropy of bin state distribution.
    
    Implementation idea: This function combines the best-fit approach with entropy-based
    reasoning. It prioritizes bins that both have minimal remaining capacity (like best-fit)
    and contribute to maintaining a more uniform distribution of remaining capacities,
    which increases flexibility for unknown future items. Instead of calculating full
    variance for each bin placement (which is O(n^2)), this implementation uses a much
    more efficient approach based on simple heuristics that promote uniformity.
    
    Args:
        item: Size of the item to be placed
        bins_remain_cap: Array of remaining capacities for available bins
        
    Returns:
        Array of priority scores for each bin (higher score means higher priority)
    """
    # Check which bins are feasible (have enough capacity)
    feasible = bins_remain_cap >= item
    
    # Basic best-fit heuristic: prioritize bins with least remaining capacity that can fit the item
    basic_score = np.where(feasible, -bins_remain_cap, -np.inf)
    
    # Efficient uniformity heuristic: promote more balanced distributions
    # Instead of calculating variance for each possible placement, we use a simpler approach
    # that considers how close each bin's remaining capacity is to the average remaining capacity
    if np.any(feasible):
        # Calculate the mean remaining capacity among all bins
        mean_capacity = np.mean(bins_remain_cap[bins_remain_cap > 0])  # Only consider non-empty bins
        
        # Calculate a uniformity score that favors bins whose post-placement capacity
        # is closer to the mean (promotes uniformity)
        post_placement_caps = bins_remain_cap - item
        # Only apply uniformity calculation to feasible bins
        post_placement_caps[~feasible] = -np.inf  # Mark infeasible as invalid
        
        # Calculate how close each bin's post-placement capacity is to the mean
        # Bins closer to mean get higher uniformity scores
        uniformity_bonus = -np.abs(post_placement_caps - mean_capacity)
        
        # Apply uniformity bonus only to feasible bins
        uniformity_bonus[~feasible] = 0  # No bonus for infeasible bins
        
        # Combine basic best-fit with uniformity bonus
        # Use a small weight for uniformity to not override best-fit behavior
        alpha = 0.01
        scores = basic_score + alpha * uniformity_bonus
    else:
        # If no bins are feasible, return the basic score (all will be -inf anyway)
        scores = basic_score
    
    return scores