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
from collections import Counter

def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """
    Cluster bins by remaining capacity and assign priority based on cluster rarity: 
    filling a bin in a rare capacity group may be more valuable to reduce future mismatches.
    This approach identifies groups of bins with similar remaining capacity and prioritizes
    bins in less common groups to improve packing efficiency.
    
    Args:
        item: Size of the item to place (float)
        bins_remain_cap: NumPy array of remaining bin capacities (float array)
    
    Returns:
        NumPy array of priority scores for each bin
    """
    # First filter out bins that cannot accommodate the item
    feasible = bins_remain_cap >= item
    scores = np.full_like(bins_remain_cap, -np.inf, dtype=float)
    
    # Work only with feasible bins
    feasible_indices = np.where(feasible)[0]
    if len(feasible_indices) == 0:
        return scores
    
    feasible_caps = bins_remain_cap[feasible]
    
    # Round capacities to nearest 0.1 to create clusters/groups
    rounded_caps = np.round(feasible_caps, decimals=1)
    
    # Count how many bins exist in each capacity cluster
    cap_counts = Counter(rounded_caps)
    
    # Calculate rarity factor: rarer capacity clusters get higher priority
    rarity_scores = np.array([1.0 / cap_counts[cap] for cap in rounded_caps])
    
    # Also incorporate best-fit heuristic (prefer bins with less remaining space)
    # This helps avoid leaving large gaps in bins
    best_fit_scores = -feasible_caps  # Higher score for smaller remaining capacity
    
    # Combine both heuristics: rarity + best-fit
    combined_scores = rarity_scores + best_fit_scores * 0.1  # Weight best-fit less heavily
    
    # Assign the calculated scores to the corresponding positions in the full array
    scores[feasible_indices] = combined_scores
    
    return scores