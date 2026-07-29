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
    Priority heuristic that implements a context-sensitive scoring kernel which dynamically
    switches between best-fit dominance and multiple-fit emphasis based on the entropy
    and spread of the current bin remainder distribution. This implementation reintegrates
    the core dynamic common-size generation mechanism from the high-performing root solution
    (Node 92, score 2014.60) into the current context-sensitive framework of Node 119.
    Specifically, replaces the manually constructed dynamic_common_sizes with a parametric
    distribution-fitting approach applied to feasible post-placement remainders. This hybrid
    retains the entropy/CV-based adaptive switching between best-fit and multiple-fit while
    restoring the more principled, data-driven common size selection that contributed to the
    original 2014.6 performance. The implementation fits the distribution only on positive
    remainders and samples candidate multipliers from high-density regions.

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
        # Calculate the percentile of the current item in the context of remaining capacities
        sorted_caps = np.sort(bins_remain_cap)
        item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps)
    else:
        item_percentile = 0.5  # Default if no bins exist
    
    # Define category-specific base weights based on refined percentile thresholds
    if item_percentile > 0.82:  # Very large item
        base_best_fit_weight, base_multiple_fit_weight = 0.30 * 1.07, 0.70 * 0.99  # Enhanced best-fit for large items
    elif item_percentile < 0.18:  # Very small item
        base_best_fit_weight, base_multiple_fit_weight = 0.0005 * 0.93, 1.199 * 1.01  # Reduced best-fit for small items
    elif item_percentile > 0.62:  # Large-medium item
        base_best_fit_weight, base_multiple_fit_weight = 0.16, 0.84  # Between medium and large
    elif item_percentile < 0.38:  # Small-medium item
        base_best_fit_weight, base_multiple_fit_weight = 0.020, 1.060  # Between medium and small
    else:  # True medium item
        base_best_fit_weight, base_multiple_fit_weight = 0.07, 0.93  # Medium weights
    
    # Compute context-sensitive adjustments based on entropy and CV of feasible remainders
    if len(feasible_post_caps) > 1:
        # Calculate Shannon entropy of the remainder distribution
        # Normalize the remainders to form a probability distribution
        pos_caps = feasible_post_caps[feasible_post_caps > 0]
        if len(pos_caps) > 0:
            # Normalize to sum to 1 for probability interpretation
            norm_caps = pos_caps / np.sum(pos_caps)
            # Remove zeros to avoid log(0)
            norm_caps = norm_caps[norm_caps > 0]
            if len(norm_caps) > 0:
                shannon_entropy = -np.sum(norm_caps * np.log(norm_caps + 1e-9))  # Add small value to avoid log(0)
                # Normalize entropy by log(number of positive caps) to get value in [0, 1]
                max_possible_entropy = np.log(len(norm_caps)) if len(norm_caps) > 1 else 1
                normalized_entropy = shannon_entropy / (max_possible_entropy + 1e-9)
            else:
                normalized_entropy = 0.0
        else:
            normalized_entropy = 0.0
            
        # Calculate coefficient of variation (CV)
        mean_remainder = np.mean(feasible_post_caps)
        std_remainder = np.std(feasible_post_caps)
        if mean_remainder > 1e-9:  # Avoid division by zero
            cv = std_remainder / mean_remainder
        else:
            cv = 0.0
    else:
        # Single feasible bin case - high certainty, low entropy
        normalized_entropy = 0.0
        cv = 0.0
    
    # Determine context-sensitive adjustment
    # When entropy is low (clustered remainders), emphasize best-fit to avoid fragmentation
    # When entropy is high (diverse remainders), emphasize multiple-fit to exploit opportunities
    entropy_factor = 2.0 * normalized_entropy - 1.0  # Maps [0,1] to [-1,1]
    
    # Adjust weights based on entropy - increase the sensitivity to entropy
    if normalized_entropy < 0.25:  # Very low entropy - highly clustered remainders
        # Strongly emphasize best-fit to avoid creating more fragmented bins
        best_fit_emphasis = 2.0
        multiple_fit_emphasis = 0.5
    elif normalized_entropy > 0.75:  # Very high entropy - very diverse remainders
        # Strongly emphasize multiple-fit to take advantage of different opportunities
        best_fit_emphasis = 0.5
        multiple_fit_emphasis = 1.5
    else:  # Medium entropy - balanced approach with slight adaptation
        # Linear interpolation between low and high entropy behavior
        mid_point = 0.25
        high_point = 0.75
        if normalized_entropy <= mid_point:
            # Interpolate between neutral and low entropy
            ratio = (normalized_entropy) / mid_point
            best_fit_emphasis = 1.0 + 1.0 * (1 - ratio)  # From 2.0 to 1.0
            multiple_fit_emphasis = 1.0 - 0.5 * (1 - ratio)  # From 0.5 to 1.0
        else:
            # Interpolate between medium and high entropy
            ratio = (normalized_entropy - mid_point) / (high_point - mid_point)
            best_fit_emphasis = 1.0 - 0.5 * ratio  # From 1.0 to 0.5
            multiple_fit_emphasis = 1.0 + 0.5 * ratio  # From 1.0 to 1.5
    
    # Apply context-sensitive adjustments
    adjusted_best_fit_weight = base_best_fit_weight * best_fit_emphasis
    adjusted_multiple_fit_weight = base_multiple_fit_weight * multiple_fit_emphasis
    
    # Generate dynamic common sizes using distribution fitting approach
    if len(feasible_post_caps) > 0:
        # Filter positive remainders
        pos_caps = feasible_post_caps[feasible_post_caps > 1e-9]
        
        if len(pos_caps) > 0:
            # Use KDE-inspired approach to identify common sizes
            # Find unique values and their nearby clusters
            unique_caps = np.unique(pos_caps)
            
            # Create a set of common sizes by looking at the actual distribution
            # We'll sample from the distribution of remainders to find representative values
            if len(unique_caps) >= 2:
                # Create common sizes by considering quantiles and differences
                # Generate quantiles as potential common sizes
                quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
                quantile_vals = np.quantile(pos_caps, quantiles)
                
                # Also add the modes/dense regions of the distribution
                # We'll use a simple clustering approach to identify dense regions
                sorted_caps = np.sort(pos_caps)
                
                # Look for gaps and identify potential common sizes
                # Take actual values from the distribution as common sizes
                common_size_candidates = np.concatenate([
                    quantile_vals,
                    unique_caps[:min(10, len(unique_caps))]  # Top 10 unique values
                ])
                
                # Add some derived values based on the distribution
                mean_val = np.mean(pos_caps)
                std_val = np.std(pos_caps)
                
                # Add mean-based values
                derived_vals = np.array([
                    mean_val,
                    mean_val - 0.5 * std_val,
                    mean_val + 0.5 * std_val,
                    mean_val - std_val,
                    mean_val + std_val,
                    np.median(pos_caps)
                ])
                
                # Include item-related values that might be relevant
                item_related = np.array([item, item * 0.5, item * 1.5, item * 0.75, item * 1.25])
                
                # Combine all
                dynamic_common_sizes = np.concatenate([
                    common_size_candidates,
                    derived_vals,
                    item_related
                ])
                
                # Remove duplicates and very small values
                dynamic_common_sizes = np.unique(dynamic_common_sizes)
                dynamic_common_sizes = dynamic_common_sizes[dynamic_common_sizes > 1e-9]
            else:
                # Fallback if we have few unique values
                dynamic_common_sizes = np.array([pos_caps[0], item, item * 0.5, item * 1.5, 100.0])
        else:
            # If no positive caps, use fallback
            dynamic_common_sizes = np.array([item, item * 0.5, item * 1.5, 100.0])
    else:
        dynamic_common_sizes = np.array([item, item * 0.5, item * 1.5, 100.0])
    
    # Calculate multiple fit score based on dynamic common sizes
    multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    # Process each common size efficiently
    for common_size in dynamic_common_sizes:
        if common_size > 1e-9:  # Avoid division by very small numbers
            # Calculate distance to nearest multiple of the common size
            # Consider multiples up to a reasonable limit
            max_multiplier = int(np.max(feasible_post_caps) // common_size) + 2
            max_multiplier = min(max_multiplier, 20)  # Increase back to original limit
            
            if max_multiplier > 0:
                # Calculate distances to all possible multiples efficiently
                multipliers = np.arange(0, max_multiplier)
                multiples = multipliers[:, np.newaxis] * common_size  # Shape (n_multipliers, 1)
                distances = np.abs(feasible_post_caps - multiples)  # Broadcasting
                min_distances = np.min(distances, axis=0)  # Min distance for each cap
                
                # Add to the score (higher score for closer matches)
                multiple_fit_score += 1.0 / (1.0 + min_distances)
    
    # Add Best Fit component: prefer bins with less remaining space
    best_fit_component = -feasible_post_caps  # Higher score for less remaining space
    
    # Combine all components with context-adjusted weights
    feasible_scores = (
        adjusted_multiple_fit_weight * multiple_fit_score + 
        adjusted_best_fit_weight * best_fit_component
    )
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores