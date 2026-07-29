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
    Best Fit heuristic with diversity penalty: prioritize bins with smallest remaining capacity that can still fit the item,
    but reduce priority for bins whose remaining capacity is very close to many others to promote heterogeneous packing states.
    
    Implementation idea: Start with the Best Fit approach (prioritizing bins with least remaining capacity after placement)
    and add a diversity penalty term that reduces the score of bins with similar remaining capacities to encourage
    more diverse utilization patterns.
    
    Args:
        item: Size of the item to place (float)
        bins_remain_cap: NumPy array of remaining bin capacities (float array)
    
    Returns:
        NumPy array of priority scores for each bin
    """
    # Initialize scores
    scores = np.zeros_like(bins_remain_cap, dtype=np.float64)
    
    # Identify feasible bins (those that can accommodate the item)
    feasible = bins_remain_cap >= item
    
    # For feasible bins, start with Best Fit principle: prioritize bins with less remaining capacity
    # Use negative of remaining capacity so smaller remaining = higher score
    base_scores = np.where(feasible, -bins_remain_cap, -np.inf)
    
    # Calculate diversity penalty for feasible bins only
    penalties = np.zeros_like(bins_remain_cap, dtype=np.float64)
    
    # Only calculate penalties for feasible bins
    if np.any(feasible):
        feasible_caps = bins_remain_cap[feasible]
        feasible_indices = np.where(feasible)[0]
        
        # For each feasible bin, count how many other feasible bins have similar capacity
        # Using a tolerance-based similarity measure
        n_feasible = len(feasible_caps)
        if n_feasible > 1:
            # Create matrix of absolute differences between capacities
            cap_diffs = np.abs(feasible_caps[:, np.newaxis] - feasible_caps)
            
            # Define similarity threshold (e.g., 1% of typical bin capacity or small fixed value)
            # Use a relative threshold based on the item size or a small absolute value
            threshold = max(item * 0.05, 0.5)  # Either 5% of item size or 0.5, whichever is larger
            
            # Count similar bins for each bin (excluding self)
            similar_counts = np.sum(cap_diffs < threshold, axis=1) - 1  # Subtract 1 to exclude self
            
            # Apply penalty proportional to number of similar bins
            # The more similar bins exist, the higher the penalty (lower the score)
            penalties[feasible_indices] = similar_counts * 0.1  # Small penalty factor
    
    # Final scores: base score minus diversity penalty
    scores = base_scores - penalties
    
    return scores