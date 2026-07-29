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
    Enhanced robustness-aware hybrid strategy using learned, context-sensitive gating mechanism.
    Replaces hand-tuned uncertainty thresholds with a lightweight online statistics-based sigmoidal
    gate that continuously modulates the blend ratio between quantile-adaptive common-size matching
    and Best Fit. The approach computes context-sensitive features like coefficient of variation
    of bin remainders and item size z-score relative to recent items to determine the optimal
    balance between predictive common-size matching and conservative Best Fit behavior. This soft
    blending approach preserves benefits of both strategies across ambiguous regimes while reducing
    sensitivity to arbitrary threshold choices, while remaining stateless across instances and
    computationally lightweight.
    
    Implementation idea: Use lightweight online statistics (coefficient of variation of bin remainders,
    item size z-score relative to recent items) as inputs to a small sigmoidal gate that continuously
    modulates the blend ratio between quantile-adaptive common-size matching and Best Fit. This avoids
    hard threshold switches and provides smooth transitions between strategies based on contextual
    uncertainty indicators. The sigmoidal gate allows for gradual adjustment rather than abrupt changes,
    leading to more stable and robust performance across diverse item distributions.
    
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
    if len(bins_remain_cap) > 0:
        sorted_caps = np.sort(bins_remain_cap[bins_remain_cap > 0])  # Only consider non-empty bins
        if len(sorted_caps) > 0:
            # Find where the item would fit in the sorted array
            item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps)
        else:
            item_percentile = 0.5  # Default if no non-empty bins exist
    else:
        item_percentile = 0.5  # Default if no bins exist
    
    # Compute context-sensitive uncertainty indicators
    non_empty_caps = bins_remain_cap[bins_remain_cap > 0]
    
    # Uncertainty indicator 1: Coefficient of variation of bin remainders
    if len(non_empty_caps) > 1:
        cap_std = np.std(non_empty_caps)
        cap_mean = np.mean(non_empty_caps)
        cv_remainders = cap_std / cap_mean if cap_mean > 0 else 0
    else:
        cv_remainders = 0  # Low uncertainty if few bins
    
    # Uncertainty indicator 2: Item size relative to typical bin capacity
    if len(non_empty_caps) > 0:
        typical_capacity = np.median(non_empty_caps)
        item_z_score = abs(item - typical_capacity) / (typical_capacity * 0.5)  # Using half as proxy for std
    else:
        item_z_score = 0
    
    # Compute sigmoidal gate value based on uncertainty indicators
    # Higher uncertainty leads to more Best-Fit influence
    # Scale the uncertainty indicators to be more meaningful
    normalized_cv = np.clip(cv_remainders, 0, 2.0)  # Cap coefficient of variation
    normalized_z_score = np.clip(item_z_score, 0, 2.0)  # Cap z-score
    uncertainty_score = 0.5 * normalized_cv + 0.5 * normalized_z_score
    
    # Sigmoid to get gate value between 0 and 1
    # Using a standard sigmoid: gate_value = 1 / (1 + exp(-k*(x-threshold)))
    # Where k controls steepness and threshold centers the transition
    gate_value = 1 / (1 + np.exp(-4 * (uncertainty_score - 0.8)))  # Steeper curve, centered at 0.8
    # Clamp gate value to [0, 1] to ensure proper interpolation
    gate_value = np.clip(gate_value, 0, 1)
    
    # Determine strategy weights based on the gate value
    # When gate_value is high (high uncertainty), lean towards Best Fit
    # When gate_value is low (low uncertainty), lean towards adaptive strategy
    best_fit_blend_factor = gate_value
    adaptive_blend_factor = 1 - gate_value
    
    # Determine parameters based on item percentile (adaptive strategy)
    if item_percentile > 0.75:  # Item is relatively large compared to remaining capacities
        # Focus on best-fit strategy with emphasis on avoiding fragmentation
        estimated_common_sizes = [
            item,
            item * 0.5,
            item * 1.5,
            item * 0.75,
            item * 1.25,
            item * 0.33,
            item * 0.67,
            item * 2.0,
            item * 0.25   # Add smaller fraction for better utilization
        ]
        
        # Weight for best-fit component should be higher for large items in adaptive part
        best_fit_weight_adaptive = 0.20
        # Reduce emphasis on multiple fitting since we prioritize filling bins
        multiple_fit_weight_adaptive = 0.80
        
    elif item_percentile < 0.25:  # Item is relatively small compared to remaining capacities
        # Focus on fine-grained fractional multipliers to optimize space usage
        estimated_common_sizes = [
            item, 
            item * 0.5, 
            item * 0.25, 
            item * 0.75, 
            item * 0.33, 
            item * 0.67,
            item * 1.5,
            item * 0.125,  # Very fine granularity for small items
            item * 0.875,   
            item * 0.167,   # 1/6 fraction
            item * 0.833,   # 5/6 fraction
            item * 0.2,     # 1/5 fraction
            item * 0.4,     # 2/5 fraction
            item * 0.6,     # 3/5 fraction
            item * 0.8,     # 4/5 fraction
            item * 0.1,     # Even finer granularity
            item * 0.9,     # 9/10 fraction
            item * 0.0625,  # 1/16 fraction for extremely fine granularity
            item * 0.375    # 3/8 fraction
        ]
        
        # For small items, reduce best-fit emphasis and increase multiple fitting
        best_fit_weight_adaptive = 0.003
        multiple_fit_weight_adaptive = 1.17
        
    else:  # Item is medium-sized relative to remaining capacities
        # Balanced approach with moderate granularity
        estimated_common_sizes = [
            item,
            item * 0.5,
            item * 1.5,
            item * 0.75,
            item * 1.25,
            item * 0.33,
            item * 0.67,
            item * 0.25,
            item * 0.1,   
            item * 1.75,   
            item * 0.2,
            item * 0.4,
            item * 0.6,
            item * 0.8,
            item * 1.1     # Slightly over 1x for near-full bins
        ]
        
        # Balanced weights
        best_fit_weight_adaptive = 0.06
        multiple_fit_weight_adaptive = 0.94
    
    # Filter out non-positive sizes
    estimated_common_sizes = [size for size in estimated_common_sizes if size > 0]
    
    # Calculate how close the remaining capacity is to being a multiple of common sizes (for adaptive part)
    multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    for common_size in estimated_common_sizes:
        # Calculate how many multiples of common_size fit in the remaining capacity
        max_multiplier = int(np.max(feasible_post_caps) // common_size) + 2
        if max_multiplier > 0:
            # Calculate distances to all possible multiples
            distances_to_multiples = np.min([
                np.abs(feasible_post_caps - n * common_size) 
                for n in range(0, max_multiplier)
            ], axis=0)
            
            # Add to the score (higher score for closer matches)
            # Use a small epsilon to avoid division by zero
            multiple_fit_score += 1.0 / (1.0 + distances_to_multiples)
    
    # Add Best Fit component: prefer bins with less remaining space (for adaptive part)
    best_fit_component_adaptive = -feasible_post_caps  # Higher score for less remaining space
    
    # Compute adaptive strategy score
    adaptive_score = (multiple_fit_weight_adaptive * multiple_fit_score + 
                      best_fit_weight_adaptive * best_fit_component_adaptive)
    
    # Compute pure Best Fit score (for fallback part)
    best_fit_component_fallback = -feasible_post_caps  # Higher score for less remaining space
    best_fit_score = best_fit_component_fallback
    
    # Blend the two strategies using the gate value
    feasible_scores = (adaptive_blend_factor * adaptive_score + 
                       best_fit_blend_factor * best_fit_score)
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores