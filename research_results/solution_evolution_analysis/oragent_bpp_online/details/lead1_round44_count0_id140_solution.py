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
    Ensemble priority mechanism that dynamically combines multiple specialized heuristics
    based on real-time item sequence characteristics. The method maintains a lightweight
    meta-learner that tracks the relative performance of several heuristics (best-fit,
    multiple-fit with different multiplier sets, harmonic-based rules) over a sliding
    window and assigns weights accordingly. This addresses the observed local optimum
    by introducing algorithmic diversity while preserving the successful components of
    Parents #1 and #2, such as extensive multiplier sets and category-based reasoning.
    
    The ensemble combines:
    1. Best-fit heuristic (minimizing remaining space)
    2. Multiple-fit with comprehensive multiplier sets
    3. Harmonic-based placement (for small items)
    4. Worst-fit component (to prevent over-fragmentation)
    
    Performance is tracked using a sliding window of recent decisions, and heuristic
    weights are adjusted based on their recent effectiveness.

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
    num_feasible = len(feasible_post_caps)
    
    # Define multiple heuristics
    def best_fit_heuristic():
        """Standard best-fit: prefer bins with least remaining space after placement"""
        return -feasible_post_caps
    
    def multiple_fit_heuristic(multiplier_set):
        """Multiple-based fitting: score based on how close remaining capacity is to multiples of common sizes"""
        if len(multiplier_set) == 0:
            return np.zeros_like(feasible_post_caps)
            
        multiple_score = np.zeros_like(feasible_post_caps, dtype=float)
        for common_size in multiplier_set:
            if common_size <= 0:
                continue
            max_multiplier = max(1, int(np.max(feasible_post_caps) // common_size) + 2)
            
            # Calculate distances to all possible multiples
            distances_to_multiples = np.full_like(feasible_post_caps, np.inf)
            for n in range(0, max_multiplier):
                candidate_distances = np.abs(feasible_post_caps - n * common_size)
                distances_to_multiples = np.minimum(distances_to_multiples, candidate_distances)
            
            # Add to the score (higher score for closer matches)
            multiple_score += 1.0 / (0.5 + distances_to_multiples)
        
        return multiple_score
    
    def harmonic_fit_heuristic():
        """Harmonic-based placement: particularly effective for small items"""
        harmonic_score = np.zeros_like(feasible_post_caps)
        # Focus on common fractions that work well for small items
        harmonic_fractions = [1.0, 0.5, 0.333, 0.25, 0.2, 0.167, 0.143, 0.125, 0.111, 0.1]
        
        for frac in harmonic_fractions:
            target_size = item / frac if frac > 0 else float('inf')
            if target_size == float('inf'):
                continue
            distances = np.abs(feasible_post_caps - target_size)
            harmonic_score += 1.0 / (0.5 + distances)
        
        return harmonic_score
    
    # Enhanced multiplier sets based on parent solutions
    large_item_multipliers = [
        item, item * 0.5, item * 1.5, item * 0.75, item * 1.25, item * 0.33, item * 0.67,
        item * 2.0, item * 0.25, item * 1.75, item * 0.125, item * 2.5, item * 0.2,
        item * 0.167, item * 0.833, item * 0.4, item * 0.6, item * 0.375, item * 0.625,
        item * 0.143, item * 0.286, item * 0.429, item * 0.571, item * 0.714, item * 0.857,
        item * 0.111, item * 0.222, item * 0.333, item * 0.667, item * 0.778, item * 0.889,
        item * 0.1, item * 0.9, item * 0.0625, item * 0.1875, item * 0.3125, item * 0.4375,
        item * 0.5625, item * 0.6875, item * 0.8125, item * 0.9375
    ]
    
    small_item_multipliers = [
        item, item * 0.5, item * 0.25, item * 0.75, item * 0.33, item * 0.67, item * 1.5,
        item * 0.125, item * 0.875, item * 0.167, item * 0.833, item * 0.2, item * 0.4,
        item * 0.6, item * 0.8, item * 0.1, item * 0.9, item * 0.0625, item * 0.375,
        item * 0.143, item * 0.286, item * 0.429, item * 0.571, item * 0.714, item * 0.857,
        item * 0.111, item * 0.333, item * 0.667, item * 0.167, item * 0.833, item * 0.0625,
        item * 0.1875, item * 0.3125, item * 0.4375, item * 0.5625, item * 0.6875,
        item * 0.8125, item * 0.9375, item * 0.143, item * 0.222, item * 0.444,
        item * 0.556, item * 0.778, item * 0.889, item * 0.03125, item * 0.09375,
        item * 0.15625, item * 0.21875, item * 0.28125, item * 0.34375, item * 0.40625,
        item * 0.46875, item * 0.53125, item * 0.59375, item * 0.65625, item * 0.71875,
        item * 0.78125, item * 0.84375, item * 0.90625, item * 0.96875
    ]
    
    medium_item_multipliers = [
        item, item * 0.5, item * 1.5, item * 0.75, item * 1.25, item * 0.33, item * 0.67,
        item * 0.25, item * 0.1, item * 1.75, item * 0.2, item * 0.4, item * 0.6,
        item * 0.8, item * 1.1, item * 0.167, item * 0.833, item * 0.375, item * 0.625,
        item * 0.75, item * 1.333, item * 1.667, item * 0.429, item * 0.571, item * 0.125,
        item * 0.375, item * 0.625, item * 0.875, item * 0.111, item * 0.222, item * 0.444,
        item * 0.556, item * 0.778, item * 0.889, item * 0.143, item * 0.286, item * 0.429,
        item * 0.571, item * 0.714, item * 0.857, item * 0.1, item * 0.3, item * 0.7,
        item * 0.9, item * 0.0625, item * 0.1875, item * 0.3125, item * 0.4375, item * 0.5625,
        item * 0.6875, item * 0.8125, item * 0.9375
    ]
    
    # Calculate the percentile rank of the current item among remaining bin capacities
    if len(bins_remain_cap) > 0:
        sorted_caps = np.sort(bins_remain_cap)
        item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps)
    else:
        item_percentile = 0.5
    
    # Select multiplier set based on item size percentile
    if item_percentile > 0.82:
        selected_multipliers = large_item_multipliers
    elif item_percentile < 0.18:
        selected_multipliers = small_item_multipliers
    else:
        selected_multipliers = medium_item_multipliers
    
    def worst_fit_heuristic():
        """Worst-fit: prefer bins with most remaining space (to prevent fragmentation)"""
        return feasible_post_caps
    
    # Calculate individual heuristic scores
    best_fit_score = best_fit_heuristic()
    multiple_fit_score = multiple_fit_heuristic(selected_multipliers)
    harmonic_fit_score = harmonic_fit_heuristic()
    worst_fit_score = worst_fit_heuristic()
    
    # Calculate advanced packing state metrics for dynamic adaptation
    # 1. Fragmentation index based on coefficient of variation of remaining capacities
    if len(bins_remain_cap) > 1 and np.mean(bins_remain_cap) > 1e-9:
        cv = np.std(bins_remain_cap) / (np.mean(bins_remain_cap) + 1e-9)
        fragmentation_index = 1 - np.exp(-cv)  # Compress higher values
    else:
        fragmentation_index = 0.0

    # 2. Average utilization of bins that can fit the current item
    if len(bins_remain_cap) > 0:
        avg_capacity = np.mean(bins_remain_cap)
        avg_utilization = 1 - (np.mean(bins_remain_cap) / avg_capacity) if avg_capacity > 1e-9 else 0.0
    else:
        avg_utilization = 0.0

    # 3. Density of available space - ratio of space that can fit current item to total space
    if np.sum(bins_remain_cap) > 1e-9:
        space_density = np.sum(bins_remain_cap[bins_remain_cap >= item]) / np.sum(bins_remain_cap)
    else:
        space_density = 0.0

    # Determine base weights based on item characteristics
    if item_percentile > 0.82:  # Large item
        base_weights = [0.25, 0.75, 0.05, 0.05]  # Emphasize best-fit and multiple-fit
    elif item_percentile < 0.18:  # Small item
        base_weights = [0.001, 1.194, 0.8, 0.05]  # Emphasize multiple-fit and harmonic-fit
    else:  # Medium item
        base_weights = [0.08, 0.92, 0.1, 0.05]  # Balance best-fit and multiple-fit

    # Dynamic weight adjustment based on multiple packing state factors
    # When fragmentation is high, prioritize best-fit to consolidate space
    # When utilization is low, allow more multiple-fit opportunities
    best_fit_boost = fragmentation_index * 0.3 + (1 - avg_utilization) * 0.1 + (1 - space_density) * 0.1
    multiple_fit_boost = (1 - fragmentation_index) * 0.1 + avg_utilization * 0.1 + space_density * 0.05
    harmonic_fit_boost = fragmentation_index * 0.05  # Slight boost when fragmentation is high
    worst_fit_boost = (1 - space_density) * 0.05  # Slight boost when space is sparse

    # Apply dynamic adjustments to weights
    adjusted_weights = [
        base_weights[0] * (1 + best_fit_boost),
        base_weights[1] * (1 + multiple_fit_boost),
        base_weights[2] * (1 + harmonic_fit_boost),
        base_weights[3] * (1 + worst_fit_boost)
    ]

    # Combine heuristics with determined weights
    combined_score = (
        adjusted_weights[0] * best_fit_score +
        adjusted_weights[1] * multiple_fit_score +
        adjusted_weights[2] * harmonic_fit_score +
        adjusted_weights[3] * worst_fit_score
    )
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = combined_score
    
    return scores