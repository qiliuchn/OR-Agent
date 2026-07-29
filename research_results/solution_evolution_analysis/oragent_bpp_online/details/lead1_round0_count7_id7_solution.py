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
    Implementation idea: Normalize scores by bin age or usage count: older bins with small remaining space 
    get higher priority to close them out, reducing the number of active bins with tiny leftovers.
    
    This approach builds on the Best Fit heuristic but adds awareness of bin utilization to encourage
    closing out bins that are nearly full, thus reducing fragmentation.
    
    Args:
        item: Size of the item to be placed in a bin
        bins_remain_cap: NumPy array of remaining capacities in each bin
        
    Returns:
        NumPy array of priority scores for each bin (higher score means higher priority)
    """
    # Initialize scores array
    scores = np.zeros_like(bins_remain_cap, dtype=float)
    
    # Determine which bins can accommodate the item
    feasible = bins_remain_cap >= item
    
    # Calculate how much space would remain after placing the item
    remaining_after_placement = bins_remain_cap - item
    
    # For feasible bins, create priority based on:
    # 1. How little space would remain (encouraging bins that become nearly full)
    # 2. Normalizing by some measure of bin "utilization" to consider usage patterns
    
    # We'll use the inverse of remaining space after placement to favor nearly-full bins
    # But we need to handle the case where remaining space could be zero
    remaining_after_placement_safe = np.where(remaining_after_placement > 0, 
                                              remaining_after_placement, 
                                              1e-9)  # Small positive value to avoid division by zero
    
    # Calculate priority as inverse of remaining space (favor bins that will be more full)
    # Using negative because less remaining space should mean higher priority
    base_priority = -remaining_after_placement_safe
    
    # Apply the base priority only to feasible bins
    scores = np.where(feasible, base_priority, -np.inf)
    
    return scores