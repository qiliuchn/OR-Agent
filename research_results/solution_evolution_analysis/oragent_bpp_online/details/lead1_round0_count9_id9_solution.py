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
    Priority heuristic based on the ratio of item size to remaining bin capacity.
    Higher ratios indicate better fit as they represent a larger proportion of the bin's remaining capacity being filled.
    This emphasizes relative fill rather than absolute leftover space, potentially leading to more efficient bin utilization.
    
    Implementation idea: Calculate the ratio of item size to remaining capacity for each bin.
    Bins with higher ratios (better relative fit) receive higher priority scores.
    For bins that cannot accommodate the item, assign a very low priority (-inf).
    
    Args:
        item: Size of the item to place (float)
        bins_remain_cap: NumPy array of remaining bin capacities (float array)
    
    Returns:
        NumPy array of priority scores for each bin
    """
    # Initialize scores array with the same shape as bins_remain_cap
    scores = np.zeros_like(bins_remain_cap, dtype=float)
    
    # Identify bins that can accommodate the item
    feasible = bins_remain_cap >= item
    
    # Calculate the ratio of item size to remaining capacity
    # To avoid division by zero, we only calculate for feasible bins
    ratios = np.where(feasible, item / bins_remain_cap, 0.0)
    
    # Assign the calculated ratios as scores for feasible bins, -inf for infeasible ones
    scores = np.where(feasible, ratios, -np.inf)
    
    return scores