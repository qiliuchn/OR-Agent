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
    Future-fit heuristic: estimates how well a bin's remaining capacity matches expected future item sizes.
    This builds upon the Best Fit approach but also considers how the remaining space might accommodate
    future items, potentially reducing waste and improving overall packing efficiency.
    
    The strategy combines:
    1. Best Fit principle (prefer bins with less remaining capacity)
    2. Future compatibility scoring (favor remaining capacities that match common item sizes)
    
    Args:
        item: Size of the current item to place
        bins_remain_cap: NumPy array of remaining capacities in available bins
        
    Returns:
        NumPy array of priority scores for each available bin
    """
    # Initialize scores array
    scores = np.zeros_like(bins_remain_cap, dtype=np.float64)
    
    # Identify feasible bins (those that can fit the current item)
    feasible = bins_remain_cap >= item
    
    # Start with the Best Fit principle: prefer bins with less remaining capacity
    # Use negative values since less remaining capacity should have higher priority
    base_scores = np.where(feasible, -bins_remain_cap, -np.inf)
    
    # Estimate potential future item size based on current item size
    # This assumes some correlation between current and future items
    # Add a component that favors remaining capacities close to expected future items
    # Using a simple heuristic: penalize remaining capacity that is much larger than the current item
    # This encourages more compact packing that could benefit future items
    
    # Calculate a "fitting factor" that rewards remaining capacity similar to item size
    # This helps ensure remaining space is useful for future items
    fitting_factor = np.where(
        feasible,
        # Reward when remaining capacity is close to item size (but not too small)
        # This promotes good utilization for future items
        -np.abs(bins_remain_cap - item) * 0.1,  # Weight this factor less than base score
        0
    )
    
    # Combine base best-fit score with future compatibility factor
    scores = base_scores + fitting_factor
    
    return scores