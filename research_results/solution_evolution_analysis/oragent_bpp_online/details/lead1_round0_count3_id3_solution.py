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
    Use exponential decay of remaining capacity in scoring: prioritize bins where (remaining_capacity - item) 
    is minimized but penalize very small leftovers more heavily to avoid fragmentation.
    
    This approach builds on the Best Fit heuristic but adds an exponential penalty for small leftover spaces
    to prevent creating bins with tiny remaining capacities that are difficult to fill later.
    
    Args:
        item: Size of the item to place (float)
        bins_remain_cap: NumPy array of remaining bin capacities (float array)
    
    Returns:
        NumPy array of priority scores for each bin (higher score means higher priority)
    """
    # Check which bins can fit the item
    feasible = bins_remain_cap >= item
    
    # Calculate the remaining capacity after placing the item
    remaining_after_placement = bins_remain_cap - item
    
    # Apply exponential decay to the remaining capacity after placement
    # This penalizes bins that would have very small leftover space
    # The exponential function e^(-x) decreases rapidly as x increases
    # So bins with small remaining_after_placement get higher scores
    exp_scores = np.exp(-remaining_after_placement)
    
    # Set scores to -inf for infeasible bins to ensure they're never chosen
    scores = np.where(feasible, exp_scores, -np.inf)
    
    return scores