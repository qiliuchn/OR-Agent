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
    Adaptive priority function based on parent solution #2 with improved randomness control.
    Uses Best Fit as the base heuristic with controlled randomness to escape local optima,
    but with a more sophisticated approach to determine when to add randomness based on
    packing density and bin utilization.
    
    This approach combines the effectiveness of Best Fit heuristic with controlled randomness 
    to explore alternative packing configurations when beneficial, potentially leading to 
    better global packing efficiency.
    
    Args:
        item: Size of the item to be placed
        bins_remain_cap: Array of remaining capacities for available bins
        
    Returns:
        Array of priority scores for each bin (higher score means higher priority)
    """
    # Calculate initial statistics
    initial_capacity = np.max(bins_remain_cap) if len(bins_remain_cap) > 0 else item
    active_bins_count = len(bins_remain_cap)  # Number of bins that can fit the current item
    
    # Calculate packing density: how full the available bins are on average
    if len(bins_remain_cap) > 0:
        avg_remaining_ratio = np.mean(bins_remain_cap) / initial_capacity
        packing_density = 1.0 - avg_remaining_ratio  # How much of the capacity is used
    else:
        packing_density = 0.0
    
    scores = np.zeros_like(bins_remain_cap, dtype=float)
    
    # Determine feasibility
    feasible = bins_remain_cap >= item
    
    # Apply Best Fit as base heuristic (higher priority to bins with less remaining space before placement)
    base_scores = np.where(feasible, -bins_remain_cap, -np.inf)
    
    # Determine when to add randomness based on how many bins are in play
    # Following parent solution #2 approach but with slight refinement
    total_bins_estimate = len(bins_remain_cap) + int(initial_capacity * 0.5)  # Rough estimate of total bins initially created
    randomness_threshold = 0.3  # Start adding randomness when 30% of bins are active
    
    should_add_randomness = active_bins_count / total_bins_estimate > randomness_threshold and len(bins_remain_cap) > 1
    
    if should_add_randomness:
        # Add small random perturbation to break ties and explore alternatives
        noise_scale = 0.01 * initial_capacity  # Small noise relative to bin capacity (matching parent #2)
        random_noise = np.random.uniform(-noise_scale, noise_scale, size=bins_remain_cap.shape)
        scores = base_scores + np.where(feasible, random_noise, 0.0)
    else:
        # Use pure Best Fit heuristic when few bins are active
        scores = base_scores
    
    return scores