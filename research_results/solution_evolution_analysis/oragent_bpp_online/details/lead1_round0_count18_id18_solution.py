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
    Weight bins by their 'opportunity cost': the difference between current remaining capacity and the smallest item seen so far;
    bins with capacity below this threshold are dead weight and should be filled aggressively if possible.
    
    Implementation idea: Instead of just looking at remaining capacity, we consider how much useful capacity remains
    in each bin. Bins with very low remaining capacity may become unusable for future items, so we should try to fill
    them when possible. This approach combines best-fit principles with forward-looking capacity utilization.
    
    Args:
        item: Size of the item to place (float)
        bins_remain_cap: NumPy array of remaining bin capacities (float array)
    
    Returns:
        NumPy array of priority scores for each bin (higher score means higher priority)
    """
    # Initialize scores
    scores = np.zeros_like(bins_remain_cap, dtype=np.float64)
    
    # Identify feasible bins (those that can accommodate the current item)
    feasible = bins_remain_cap >= item
    
    # For infeasible bins, assign very low priority
    scores = np.where(feasible, 0.0, -np.inf)
    
    # Calculate remaining capacity after placing the current item
    remaining_after_placement = bins_remain_cap - item
    
    # To estimate opportunity cost, we need a reference point for minimum future item size.
    # We'll use the current item size as a proxy for future items, but also consider the
    # remaining capacity itself as an indicator of future utility.
    
    # The core insight is that bins with remaining capacity close to the item size are ideal
    # (best fit principle), but we also want to avoid leaving bins with very small capacity
    # that can't be used for future items.
    
    # Score calculation: combine best-fit principle with capacity utilization consideration
    # Higher priority for bins that:
    # 1. Have enough space for the current item
    # 2. Leave minimal remaining space after placement (best-fit)
    # 3. Are not left with tiny amounts of space that would be unusable
    scores = np.where(
        feasible,
        # Prioritize bins with less remaining space after placement (best-fit)
        -(remaining_after_placement + 1e-9),  # Add small epsilon to avoid division issues
        -np.inf
    )
    
    # Additional consideration: penalize bins that would be left with very little space
    # These bins would likely become unusable for future larger items
    very_small_remaining = remaining_after_placement < 0.1  # Threshold for "dead weight" capacity
    penalty = np.where(very_small_remaining & feasible, -1000.0, 0.0)  # Heavy penalty
    scores += penalty
    
    # Normalize by item size to make the scoring more consistent across different item sizes
    scores /= (item + 1e-9)  # Add small value to prevent division by zero
    
    return scores