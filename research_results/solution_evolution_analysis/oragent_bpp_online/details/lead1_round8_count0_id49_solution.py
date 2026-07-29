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
    Adaptive priority function that enhances Best Fit with controlled randomness based on
    packing state to escape local optima while maintaining the effectiveness of Best Fit.
    
    The algorithm monitors how many bins are actively being used and introduces controlled
    randomness when many bins are in play to explore alternative packing configurations.
    This builds on the proven effectiveness of Best Fit while adding exploration capability
    when the packing becomes complex with many active bins.
    
    Implementation considers:
    - Maintain Best Fit as core heuristic (proven effectiveness)
    - Add controlled randomness based on number of active bins
    - Build on parent solution's insight about when to introduce exploration
    
    Args:
        item: Size of the item to be placed
        bins_remain_cap: Array of remaining capacities for available bins
        
    Returns:
        Array of priority scores for each bin (higher score means higher priority)
    """
    # Calculate feasibility mask
    feasible = bins_remain_cap >= item
    
    # If no bins are feasible, return zeros (shouldn't happen in normal operation)
    if not np.any(feasible):
        return np.zeros_like(bins_remain_cap, dtype=float)
    
    # Calculate base Best Fit scores: prioritize bins with smallest remaining capacity after placement
    # This tightly packs items which is generally most effective
    base_scores = np.where(feasible, -(bins_remain_cap - item), -np.inf)
    
    # Estimate total number of bins originally created based on max capacity
    initial_capacity = np.max(bins_remain_cap) if len(bins_remain_cap) > 0 else 1.0
    
    # Count how many bins have been used so far (bins with capacity less than initial capacity)
    # Since bins_remain_cap represents remaining capacity, bins with remaining < initial capacity
    # have been used. We can estimate usage by looking at how many bins have been modified.
    # A simpler approach: count how many bins are currently being considered (feasible bins)
    # as a proxy for active packing complexity
    active_bins_count = np.sum(feasible)
    
    # Use a threshold based on the number of available bins to decide when to introduce randomness
    # When there are fewer bins in play, stick to deterministic Best Fit
    # When there are many bins in play, add some randomness to escape local optima
    # Using similar logic to parent solution but adapted
    total_bins_estimate = len(bins_remain_cap)  # Rough estimate of bins currently available
    randomness_threshold = 0.25  # Start adding randomness when 25% of feasible bins are present
    
    # Determine whether to add randomness based on how many bins are in play
    should_add_randomness = active_bins_count / max(total_bins_estimate, 1) > randomness_threshold
    
    if should_add_randomness and active_bins_count > 1:
        # Add small random perturbation to break ties and explore alternatives
        # Use a noise scale relative to the initial bin capacity
        noise_scale = 0.01 * initial_capacity  # Small noise relative to bin capacity (matching parent)
        random_noise = np.random.uniform(-noise_scale, noise_scale, size=bins_remain_cap.shape)
        scores = base_scores + np.where(feasible, random_noise, 0.0)
    else:
        # Use pure Best Fit heuristic when few bins are active
        scores = base_scores
    
    return scores