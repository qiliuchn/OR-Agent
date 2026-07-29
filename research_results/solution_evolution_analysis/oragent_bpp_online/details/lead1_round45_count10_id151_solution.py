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
    Hybrid priority function that applies computationally expensive logic only for high-leverage items
    (e.g., item > 0.4 or when fewer than 3 feasible bins exist), using on-the-fly Farey sequence 
    generation up to adaptive order (based on current bin dispersion) to identify near-optimal 
    rational fits, while defaulting to a simplified, fast-path version of the Node 92 heuristic 
    for routine placements—thereby preserving average-case efficiency while targeting the 
    val_0/val_1 bottlenecks where they matter most.
    
    The approach implements a dual-path strategy: for normal items, it uses the efficient 
    percentile-based logic from successful parent solutions; for critical items (large relative 
    to bins or few feasible bins), it employs sophisticated Farey sequence analysis to find
    optimal rational fits that may resolve challenging packing scenarios causing the val_0/val_1
    bottlenecks.

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
    feasible_caps = bins_remain_cap[feasible_bins]
    
    # Calculate bin dispersion for adaptive weighting in both paths
    cap_dispersion = np.std(feasible_caps) / (np.mean(feasible_caps) + 1e-9) if len(feasible_caps) > 0 else 0
    
    # Determine if this is a high-leverage scenario requiring expensive computation
    # High leverage if item is large relative to typical bin capacity or few feasible bins
    item_ratio = item / np.mean(bins_remain_cap) if len(bins_remain_cap) > 0 else 0
    is_high_leverage = item_ratio > 0.4 or len(feasible_post_caps) < 3
    
    if is_high_leverage:
        # Apply computationally expensive Farey sequence analysis for high-leverage cases
        
        # Calculate item percentile among feasible capacities
        sorted_caps = np.sort(feasible_caps)
        item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps) if len(sorted_caps) > 0 else 0.5
        
        # Use adaptive Farey sequence based on current bin dispersion
        farey_order = max(3, min(10, int(5 + 5 * cap_dispersion)))  # Adaptive order between 3 and 10
        
        # Generate Farey sequence up to adaptive order
        farey_sequence = []
        for q in range(1, farey_order + 1):
            for p in range(0, q + 1):
                if np.gcd(p, q) == 1:  # Only include reduced fractions
                    farey_sequence.append(p / q)
        farey_sequence = np.array(list(set(farey_sequence)))  # Remove duplicates
        
        # Create common sizes based on item and Farey fractions of remaining capacities
        common_sizes = []
        for fraction in farey_sequence:
            common_sizes.extend([
                item,
                item * fraction,
                item * (1 + fraction),
                item * (1 - fraction) if fraction < 1 else item,
                np.mean(feasible_caps) * fraction if len(feasible_caps) > 0 else item
            ])
        common_sizes = np.array(common_sizes)
        common_sizes = common_sizes[common_sizes > 1e-6]  # Remove very small values
        
        # Compute detailed scoring components
        best_fit_component = -feasible_post_caps  # Higher score for less remaining space after placement
        
        # Multiple-fit scoring using Farey-based common sizes
        multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
        
        for common_size in common_sizes:
            # Calculate distance to nearest multiple of the common size
            max_multiplier = int(np.max(feasible_post_caps) // common_size) + 3 if common_size > 0 else 0
            max_multiplier = min(max_multiplier, 8)  # Limit computation
            
            if max_multiplier > 0:
                distances_to_multiples = np.min([
                    np.abs(feasible_post_caps - n * common_size) 
                    for n in range(0, max_multiplier + 1)
                ], axis=0)
                
                # Add to the score (higher score for closer matches)
                multiple_fit_score += 1.0 / (1.0 + distances_to_multiples)
        
        # Weight determination based on item percentile
        if item_percentile > 0.82:  # Very large item
            best_fit_weight, multiple_fit_weight = 0.25 * 1.07, 0.75 * 0.99
        elif item_percentile < 0.18:  # Very small item
            best_fit_weight, multiple_fit_weight = 0.001 * 0.93, 1.194 * 1.01
        elif item_percentile > 0.62:  # Large-medium item
            best_fit_weight, multiple_fit_weight = 0.14, 0.86
        elif item_percentile < 0.38:  # Small-medium item
            best_fit_weight, multiple_fit_weight = 0.025, 1.055
        else:  # True medium item
            best_fit_weight, multiple_fit_weight = 0.08, 0.92
        
        # For high-leverage cases, adjust weights based on bin dispersion to handle different scenarios
        if cap_dispersion > 0.3:  # High dispersion in bin capacities
            # When bins have very different remaining capacities, prioritize best fit
            best_fit_weight *= 1.1
            multiple_fit_weight *= 0.9
        else:  # Low dispersion - bins are similar
            # When bins are similar, prioritize multiple fit patterns
            best_fit_weight *= 0.9
            multiple_fit_weight *= 1.1
        
        # Combine components for high-leverage case
        feasible_scores = best_fit_weight * best_fit_component + multiple_fit_weight * multiple_fit_score
        
    else:
        # Fast-path: Simplified Node 92 heuristic for routine placements
        # Calculate item percentile among all bins
        sorted_all_caps = np.sort(bins_remain_cap)
        item_percentile = np.searchsorted(sorted_all_caps, item) / len(sorted_all_caps) if len(sorted_all_caps) > 0 else 0.5
        
        # Simple scoring components
        best_fit_component = -feasible_post_caps  # Higher score for less remaining space after placement
        
        # Simplified multiple-fit scoring using basic common sizes
        basic_common_sizes = np.array([
            item, item * 0.5, item * 1.5, item * 0.25, item * 0.75, item * 1.25,
            item * 2.0, np.mean(feasible_caps) * 0.5 if len(feasible_caps) > 0 else item
        ])
        basic_common_sizes = basic_common_sizes[basic_common_sizes > 1e-6]
        
        multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
        
        for common_size in basic_common_sizes:
            distances_to_multiples = np.min([
                np.abs(feasible_post_caps - n * common_size) 
                for n in range(0, 4)  # Limited range for speed
            ], axis=0)
            
            multiple_fit_score += 1.0 / (1.0 + distances_to_multiples)
        
        # Weight determination based on item percentile
        if item_percentile > 0.82:  # Very large item
            best_fit_weight, multiple_fit_weight = 0.25, 0.75
        elif item_percentile < 0.18:  # Very small item
            best_fit_weight, multiple_fit_weight = 0.001, 1.194
        elif item_percentile > 0.62:  # Large-medium item
            best_fit_weight, multiple_fit_weight = 0.14, 0.86
        elif item_percentile < 0.38:  # Small-medium item
            best_fit_weight, multiple_fit_weight = 0.025, 1.055
        else:  # True medium item
            best_fit_weight, multiple_fit_weight = 0.08, 0.92
        
        # For fast-path, also adapt based on bin dispersion
        if cap_dispersion > 0.3:  # High dispersion in bin capacities
            best_fit_weight *= 1.05
            multiple_fit_weight *= 0.95
        else:  # Low dispersion - bins are similar
            best_fit_weight *= 0.95
            multiple_fit_weight *= 1.05
        
        # Combine components for fast-path case
        feasible_scores = best_fit_weight * best_fit_component + multiple_fit_weight * multiple_fit_score
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores