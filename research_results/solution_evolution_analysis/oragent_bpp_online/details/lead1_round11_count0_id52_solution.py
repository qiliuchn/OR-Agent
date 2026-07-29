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
    Simplified priority function based on the successful Best Fit approach from Parent #2,
    with refined randomness control parameters. Uses Best Fit as the base heuristic 
    (higher priority to bins with less remaining space that can fit the item) with 
    controlled randomness to escape local optima when beneficial.
    
    This approach follows the proven strategy of Parent #2 which achieved superior results
    by using the simple but effective Best Fit heuristic with carefully controlled randomness
    applied under specific conditions to maintain exploration without degrading performance.
    
    Args:
        item: Size of the item to be placed
        bins_remain_cap: Array of remaining capacities for available bins
        
    Returns:
        Array of priority scores for each bin (higher score means higher priority)
    """
    # Calculate initial capacity for normalization
    initial_capacity = np.max(bins_remain_cap) if len(bins_remain_cap) > 0 else item
    
    # Calculate packing density: how full the available bins are on average
    if len(bins_remain_cap) > 0:
        avg_remaining_ratio = np.mean(bins_remain_cap) / initial_capacity
        packing_density = 1.0 - avg_remaining_ratio  # How much of the capacity is used
    else:
        packing_density = 0.0
    
    # Determine feasibility
    feasible = bins_remain_cap >= item
    
    # Apply Best Fit as base heuristic (higher priority to bins with less remaining space)
    base_scores = np.where(feasible, -bins_remain_cap, -np.inf)
    
    # Determine when to add randomness based on how many bins are in play
    # Use the approach from Experiment #2 which showed slightly better results
    should_add_randomness = (
        len(bins_remain_cap) > 1 and  # Only add randomness if there are multiple bins to choose from
        packing_density > 0.2  # Only when we're reasonably into the packing process
    )

    if should_add_randomness:
        # Add small random perturbation to break ties and explore alternatives
        noise_scale = 0.01 * initial_capacity  # Noise scale from Parent #2 (proven effective)
        random_noise = np.random.uniform(-noise_scale, noise_scale, size=bins_remain_cap.shape)
        scores = base_scores + np.where(feasible, random_noise, 0.0)
    else:
        # Use pure Best Fit heuristic
        scores = base_scores
    
    return scores