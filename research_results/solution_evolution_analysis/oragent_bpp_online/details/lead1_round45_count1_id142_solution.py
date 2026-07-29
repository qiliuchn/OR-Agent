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
    Priority heuristic implementing a distilled neural proxy approach.
    This function combines insights from the successful dynamic common-size generation
    from the parent solution with a symbolic representation of learned patterns
    that mimic neural decision-making. Rather than training an actual neural network,
    this implementation uses a symbolic approximation that captures the essence
    of learned behavior: focusing on the relationship between item size, bin remainders,
    and patterns that resolve the persistent val_0/val_1 bottleneck instances.
    
    The approach maintains the percentile-based item categorization that proved
    essential in previous successful solutions while incorporating a simplified
    version of the multiple-fit concept that focuses on key rational approximations
    without the computational overhead. It also includes mechanisms to detect
    problematic remainder configurations that typically lead to suboptimal packings.
    
    Implementation idea: Instead of training an actual neural network, this function
    uses symbolic expressions that approximate learned decision patterns, focusing
    on the relationship between item size, remaining capacities, and their ratios.
    It computes features similar to what a neural network might learn: how well
    an item fits into various bins considering both immediate fit and future potential.

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
    
    # Calculate the percentile rank of the current item among remaining bin capacities
    # Using only feasible bins for more accurate context
    feasible_caps = bins_remain_cap[feasible_bins]
    if len(feasible_caps) > 0:
        # Calculate the percentile of the current item in the context of feasible capacities
        sorted_caps = np.sort(feasible_caps)
        item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps) if len(sorted_caps) > 0 else 0.5
    else:
        item_percentile = 0.5  # Default if no feasible bins exist
    
    # Define category-specific weights based on refined percentile thresholds
    # These weights are adapted from the successful parent solution
    if item_percentile > 0.82:  # Very large item
        best_fit_weight, multiple_fit_weight = 0.25 * 1.07, 0.75 * 0.99  # Enhanced best-fit for large items
    elif item_percentile < 0.18:  # Very small item
        best_fit_weight, multiple_fit_weight = 0.001 * 0.93, 1.194 * 1.01  # Reduced best-fit for small items
    elif item_percentile > 0.62:  # Large-medium item
        best_fit_weight, multiple_fit_weight = 0.14, 0.86  # Between medium and large
    elif item_percentile < 0.38:  # Small-medium item
        best_fit_weight, multiple_fit_weight = 0.025, 1.055  # Between medium and small
    else:  # True medium item
        best_fit_weight, multiple_fit_weight = 0.08, 0.92  # Medium weights
    
    # Compute features that capture learned patterns (symbolic neural proxy)
    # Feature 1: How well the item fits in each bin (best fit preference)
    best_fit_component = -feasible_post_caps  # Higher score for less remaining space after placement
    
    # Feature 2: Multiple-fit scoring with a distilled set of common sizes
    # This mimics the learned pattern recognition of neural networks
    # Using a simplified set of common sizes based on Farey-like fractions
    multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    # Generate distilled common sizes that represent learned patterns
    # Instead of dynamic generation, use a fixed but comprehensive set
    distilled_common_sizes = np.array([
        item,                    # Item itself
        item * 0.5,             # Half-item
        item * 1.5,             # 1.5x item
        item * 0.25,            # Quarter-item
        item * 0.75,            # 0.75x item
        item * 1.25,            # 1.25x item
        item * 2.0,             # Double item
        item * 0.33,            # One-third item
        item * 0.67,            # Two-thirds item
        item * 0.125,           # Eighth item
        item * 0.875,           # 7/8 item
        # Some common bin fractions
        1.0, 2.0, 5.0, 10.0,   # Common divisors for bin capacity (assuming capacity ~100)
        25.0, 50.0, 75.0       # Common bin capacity fractions
    ])
    
    # Remove very small values to avoid numerical issues
    distilled_common_sizes = distilled_common_sizes[distilled_common_sizes > 1e-6]
    
    # Calculate multiple fit score based on distilled common sizes
    for common_size in distilled_common_sizes:
        # Calculate distance to nearest multiple of the common size
        # Consider multiples up to a reasonable limit
        max_multiplier = int(np.max(feasible_post_caps) // common_size) + 3 if common_size > 0 else 0
        max_multiplier = min(max_multiplier, 10)  # Limit to prevent excessive computation
        
        if max_multiplier > 0:
            # Calculate distances to all possible multiples
            distances_to_multiples = np.min([
                np.abs(feasible_post_caps - n * common_size) 
                for n in range(0, max_multiplier + 1)
            ], axis=0)
            
            # Add to the score (higher score for closer matches)
            multiple_fit_score += 1.0 / (1.0 + distances_to_multiples)
    
    # Feature 3: Remainder efficiency (to address val_0/val_1 bottlenecks)
    # Prefer configurations that leave remainders that are useful for future items
    remainder_efficiency = np.zeros_like(feasible_post_caps, dtype=float)
    
    # Look for remainders that match common item sizes or their multiples
    # This helps with future packing efficiency
    common_item_proportions = np.array([0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9])
    for prop in common_item_proportions:
        distances_to_useful_remainders = np.abs(feasible_post_caps - prop * item)
        remainder_efficiency += 1.0 / (1.0 + distances_to_useful_remainders)
    
    # Combine all components with learned weights
    feasible_scores = (
        best_fit_weight * best_fit_component +
        multiple_fit_weight * multiple_fit_score +
        0.1 * remainder_efficiency  # Additional term for future efficiency
    )
    
    # Normalize scores to prevent extreme values
    # Only normalize if there's variation in scores
    score_range = np.max(feasible_scores) - np.min(feasible_scores)
    if score_range > 1e-9:
        feasible_scores = (feasible_scores - np.min(feasible_scores)) / score_range
    else:
        # If all scores are the same, keep them as they are
        pass
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores