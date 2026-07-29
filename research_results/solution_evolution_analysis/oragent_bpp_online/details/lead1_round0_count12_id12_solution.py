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
    Piecewise linear scoring function: prioritize bins with remaining capacity just above item size,
    assign medium priority to those with much larger capacity, and low priority to barely fitting bins
    to avoid tight fits that block future options.
    
    This approach balances between Best Fit (which can create tight fits) and First Fit (which may waste space),
    attempting to find a sweet spot that leaves reasonable space for future items while not wasting too much space.
    
    Args:
        item: Size of the item to be placed
        bins_remain_cap: Array of remaining capacities of available bins
        
    Returns:
        Array of priority scores for each bin (higher score means higher priority)
    """
    # Initialize scores array
    scores = np.zeros_like(bins_remain_cap)
    
    # Identify feasible bins (those that can accommodate the item)
    feasible = bins_remain_cap >= item
    
    # Calculate the remaining space after placing the item
    remaining_after_placement = bins_remain_cap - item
    
    # Define thresholds for different priority levels
    # We'll use a threshold slightly above the item size to identify "just right" fits
    optimal_fit_threshold = item * 0.3  # Allow some buffer beyond the item size
    very_tight_threshold = item * 0.1   # Very tight fits get lower priority
    
    # For feasible bins, calculate priority based on remaining space after placement
    # The idea is to prefer bins where remaining space is moderate (not too tight, not too loose)
    for i in range(len(bins_remain_cap)):
        if feasible[i]:
            remaining = remaining_after_placement[i]
            
            # If remaining space is very small (tight fit), give low priority
            if remaining <= very_tight_threshold:
                scores[i] = remaining  # Low score proportional to remaining space
            
            # If remaining space is moderate (optimal range), give high priority
            elif remaining <= optimal_fit_threshold:
                # Higher score for more remaining space in this range
                scores[i] = remaining + 1.0  # Boost score in optimal range
            
            # If remaining space is large (loose fit), give medium priority
            else:
                # Medium score that decreases as space gets larger (to avoid excessive waste)
                scores[i] = 1.0 / (1.0 + remaining)  # Decreasing function of remaining space
    
    # For infeasible bins, assign negative infinity to ensure they're never chosen
    scores = np.where(feasible, scores, -np.inf)
    
    return scores