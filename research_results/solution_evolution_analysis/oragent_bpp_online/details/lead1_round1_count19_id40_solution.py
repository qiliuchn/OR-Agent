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
from typing import List

def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """
    Priority heuristic that adaptively combines multiple-fit and Best Fit components
    using a multiplicative formulation to avoid manual weight tuning.
    
    Implementation idea: Replace the additive combination of multiple-fit and Best Fit 
    components with a multiplicative or conditional scoring formulation that inherently 
    suppresses Best Fit when multiple-fit signals are strong, and vice versa. This uses
    priority = multiple_fit_score * (1 + α * normalized_best_fit_component) where the 
    Best Fit influence is scaled by how close the post-placement capacity is to zero.
    This avoids the need for manually tuned linear weights and creates a more adaptive 
    interaction between the two heuristics.
    
    Args:
        item: Size of the item to place
        bins_remain_cap: NumPy array of remaining bin capacities
        
    Returns:
        NumPy array of priority scores for each bin
    """
    # Calculate post-placement remaining capacities
    post_placement_caps = bins_remain_cap - item
    
    # Initialize scores
    scores = np.full_like(bins_remain_cap, -np.inf, dtype=float)
    
    # Only consider bins that can accommodate the item
    feasible_bins = bins_remain_cap >= item
    
    if not np.any(feasible_bins):
        return scores  # All bins remain with -inf scores
    
    feasible_post_caps = post_placement_caps[feasible_bins]
    feasible_scores = np.zeros_like(feasible_post_caps, dtype=float)
    
    # Estimate common item sizes based on the current item
    current_item = item
    estimated_common_sizes = [current_item, current_item * 0.9, current_item * 1.1, 
                              current_item * 0.5, current_item * 1.5, current_item * 0.75, current_item * 1.25]
    
    # Filter out non-positive sizes
    estimated_common_sizes = [size for size in estimated_common_sizes if size > 0]
    
    # Calculate how close the remaining capacity is to being a multiple of common sizes
    multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    for common_size in estimated_common_sizes:
        # Calculate distances to the nearest multiple of common_size
        multipliers = feasible_post_caps / common_size
        lower_mult = np.floor(multipliers)
        upper_mult = lower_mult + 1
        
        dist_to_lower = np.abs(feasible_post_caps - lower_mult * common_size)
        dist_to_upper = np.abs(feasible_post_caps - upper_mult * common_size)
        
        min_distances = np.minimum(dist_to_lower, dist_to_upper)
        
        # Add to the score (higher score for closer matches)
        multiple_fit_score += 1.0 / (1.0 + min_distances)
    
    # Calculate Best Fit component: prefer bins with less remaining space
    best_fit_component = -feasible_post_caps  # Higher score for less remaining space
    
    # Normalize best fit component to avoid overwhelming the multiple fit score
    max_abs_best_fit = np.max(np.abs(best_fit_component))
    if max_abs_best_fit > 0:
        normalized_best_fit = best_fit_component / max_abs_best_fit
    else:
        normalized_best_fit = best_fit_component
    
    # Create a normalized Best Fit component that scales based on how full the bin will be
    # After placing the item, bins that will be fuller should get higher Best Fit influence
    # We'll use the remaining capacity after placement to determine how full the bin will be
    typical_capacity = 100.0  # Standard bin capacity
    
    # Calculate how full each bin will be after placement (in terms of how much is used)
    # Since bins start full at 100, after placing an item of size s in a bin with r remaining capacity,
    # the bin will have used capacity of (100 - r) + s, so the utilization is ((100-r) + s) / 100
    used_capacity_after_placement = (typical_capacity - bins_remain_cap[feasible_bins]) + item
    utilization_after_placement = used_capacity_after_placement / typical_capacity
    
    # Use the utilization to scale the best fit influence: higher utilization means higher best fit influence
    # But we want to be careful not to make it too aggressive
    best_fit_scaling_factor = np.clip(utilization_after_placement, 0.1, 1.0)
    
    # Multiplicative combination: multiple_fit_score * (1 + alpha * best_fit_component)
    # Where alpha is scaled by how much the bin will be filled
    base_alpha = 0.04  # Reduced base weight for best fit component to fine-tune balance
    adaptive_alpha = base_alpha * best_fit_scaling_factor
    
    # Apply the multiplicative combination
    feasible_scores = multiple_fit_score * (1 + adaptive_alpha * normalized_best_fit)
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores