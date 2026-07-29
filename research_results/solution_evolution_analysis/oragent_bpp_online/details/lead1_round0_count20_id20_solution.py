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
    Context-aware priority function that considers both individual bin capacity and global bin occupancy.
    
    This implementation combines the best-fit heuristic (prioritizing bins with least remaining capacity that can fit the item)
    with a global context awareness that evaluates how well the item fits relative to the distribution of remaining capacities.
    The approach tries to avoid creating unnecessary bins by considering both local fit quality and global packing efficiency.
    
    Args:
        item: Size of the item to be placed in a bin
        bins_remain_cap: NumPy array of remaining capacities in available bins
    
    Returns:
        NumPy array of priority scores for each bin (higher score means higher priority)
    """
    # Create a copy to avoid modifying the original array
    scores = np.zeros_like(bins_remain_cap, dtype=np.float64)
    
    # Identify bins that can fit the item
    feasible = bins_remain_cap >= item
    
    # Initialize all scores to negative infinity for non-feasible bins
    scores.fill(-np.inf)
    
    # For feasible bins, calculate priority based on multiple factors
    if np.any(feasible):
        feasible_bins = bins_remain_cap[feasible]
        feasible_scores = np.zeros_like(feasible_bins, dtype=np.float64)
        
        # Factor 1: Best-fit component - prefer bins with least remaining space after placement
        # After placing the item, the remaining capacity would be (bin_capacity - item)
        remaining_after_placement = feasible_bins - item
        # Higher priority for less remaining space (negative sign makes smaller values have higher priority)
        best_fit_score = -remaining_after_placement
        
        # Factor 2: Global context - consider how well the item fits relative to other available spaces
        # Calculate how tight the fit is compared to other feasible bins
        # Normalize by the maximum remaining capacity among feasible bins to provide context
        max_feasible_cap = np.max(feasible_bins) if len(feasible_bins) > 0 else 1.0
        fit_tightness = item / np.maximum(feasible_bins, 1e-9)  # Avoid division by zero
        
        # Combine both factors
        feasible_scores = best_fit_score + fit_tightness
        
        # Assign the calculated scores back to the correct positions
        scores[feasible] = feasible_scores
    
    return scores