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
    Introduce randomness controlled by problem scale: for early-stage packing (few bins), use deterministic Best Fit;
    later, inject stochasticity to escape local minima in bin utilization patterns.
    
    This approach combines the effectiveness of Best Fit heuristic with controlled randomness to explore alternative
    packing configurations when many bins are already in use, potentially leading to better global packing efficiency.
    
    Args:
        item: Size of the item to be placed
        bins_remain_cap: Array of remaining capacities for available bins
        
    Returns:
        Array of priority scores for each bin (higher score means higher priority)
    """
    # Calculate how many bins are currently in use (bins with capacity less than initial max capacity)
    initial_capacity = np.max(bins_remain_cap)  # Assuming all bins started with same capacity
    active_bins_count = len(bins_remain_cap)  # Number of bins that can fit the current item
    
    # Use a threshold based on the number of available bins to decide when to introduce randomness
    # When there are fewer bins in play, stick to deterministic Best Fit
    # When there are many bins in play, add some randomness to escape local optima
    total_bins_estimate = len(bins_remain_cap) + int(initial_capacity * 0.5)  # Rough estimate of total bins initially created
    randomness_threshold = 0.3  # Start adding randomness when 30% of bins are active
    
    scores = np.zeros_like(bins_remain_cap, dtype=float)
    
    # Determine feasibility
    feasible = bins_remain_cap >= item
    
    # Apply Best Fit as base heuristic (higher priority to bins with less remaining space)
    base_scores = np.where(feasible, -bins_remain_cap, -np.inf)
    
    # Decide whether to add randomness based on how many bins are in play
    should_add_randomness = active_bins_count / total_bins_estimate > randomness_threshold
    
    if should_add_randomness and len(bins_remain_cap) > 1:
        # Add small random perturbation to break ties and explore alternatives
        noise_scale = 0.01 * initial_capacity  # Small noise relative to bin capacity
        random_noise = np.random.uniform(-noise_scale, noise_scale, size=bins_remain_cap.shape)
        scores = base_scores + np.where(feasible, random_noise, 0.0)
    else:
        # Use pure Best Fit heuristic when few bins are active
        scores = base_scores
    
    return scores