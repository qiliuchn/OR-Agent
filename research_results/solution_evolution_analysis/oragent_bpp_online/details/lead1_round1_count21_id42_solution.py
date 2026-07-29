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
    Priority heuristic for online bin packing that leverages the structure of existing bin remainders
    to identify useful capacity patterns without relying on the current item as a proxy for future sizes.
    
    Implementation idea: Instead of deriving common sizes from the current item, this approach identifies
    dominant modes in the current bins_remain_cap array using quantile clustering. It finds clusters around
    meaningful proportions of bin capacity (e.g., 0.25, 0.5, 0.75) to determine useful residual patterns.
    Then it prioritizes bins whose post-placement capacity aligns with these observed modes, creating a
    self-reinforcing packing strategy that adapts to emergent structure in the instance without requiring
    historical item data.
    
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
    
    # Work only with feasible bins
    feasible_post_caps = post_placement_caps[feasible_bins]
    feasible_caps = bins_remain_cap[feasible_bins]
    
    n_feasible = len(feasible_post_caps)
    
    # 1. Best Fit component: prefer bins with less remaining space after placement
    best_fit_scores = -feasible_post_caps  # Higher score for less remaining space
    
    # 2. Mode detection component: identify useful capacity patterns from existing bins
    # Focus on finding recurring patterns in existing bin remainders that indicate useful capacities
    mode_scores = np.zeros(n_feasible)
    
    if n_feasible > 0:
        # Find meaningful capacity levels from current bins using histogram-based peak detection
        # This identifies commonly occurring capacity levels that might be useful for future items
        positive_caps = bins_remain_cap[bins_remain_cap > 0]
        
        if len(positive_caps) > 0:
            # Use histogram to find peaks in capacity distribution
            # Create bins for histogram based on reasonable granularity
            cap_min, cap_max = np.min(positive_caps), np.max(positive_caps)
            if cap_max > cap_min:
                # Create histogram to find common capacity levels
                hist, bin_edges = np.histogram(positive_caps, bins=min(20, len(positive_caps)//2 + 1) if len(positive_caps) > 1 else 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Find bins with above-average counts (potential modes)
                threshold = np.mean(hist) if len(hist) > 0 else 0
                mode_candidates = bin_centers[hist > threshold]
                
                # Also include some key ratios of the maximum capacity that often appear in good packings
                max_possible_cap = 100.0  # Assuming bin capacity is 100 based on problem description
                # Use patterns based on common fractions that emerge in efficient packings
                fraction_targets = [max_possible_cap * f for f in [0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9]]
                
                # Combine detected modes with fraction-based targets
                target_levels = list(mode_candidates) + fraction_targets
                
                # Additionally, include patterns based on the current item as in parent solution
                current_item = item
                item_based_targets = [current_item, current_item * 0.9, current_item * 1.1, 
                                      current_item * 0.5, current_item * 1.5, current_item * 0.75, current_item * 1.25]
                target_levels.extend([target for target in item_based_targets if target > 0])
                
                # Remove duplicates and zeros
                target_levels = list(set([level for level in target_levels if level > 0]))
                
                # For each feasible bin, calculate how well its post-placement capacity matches targets
                if target_levels:  # Only proceed if we have targets
                    for target in target_levels:
                        # Distance to target - closer is better
                        distances = np.abs(feasible_post_caps - target)
                        # Convert to similarity scores (higher for closer matches)
                        similarities = 1.0 / (1.0 + distances)
                        # Add to mode scores
                        mode_scores += similarities
    
    # 3. Diversity component: promote variety in remaining capacities
    diversity_scores = np.zeros(n_feasible)
    if n_feasible > 1:
        mean_cap = np.mean(feasible_post_caps)
        std_cap = np.std(feasible_post_caps) if np.std(feasible_post_caps) > 0 else 1.0
        # Bins with capacities far from the mean get higher diversity scores
        # This encourages spreading out remaining capacities
        diversity_scores = np.abs(feasible_post_caps - mean_cap) / (std_cap + 1e-8)
    
    # Combine all components with appropriate weights
    # Increase weight of mode detection based on parent solution success with multiple matching
    combined_scores = (
        0.5 * best_fit_scores +      # Reduced best fit to allow more exploration
        0.7 * mode_scores +          # Increased mode detection weight like parent's multiple matching
        0.1 * diversity_scores       # Mild diversity encouragement
    )
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = combined_scores
    
    return scores