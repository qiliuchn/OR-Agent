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
    Enhanced adaptive priority function that incorporates entropy-based fragmentation detection
    to dynamically adjust the balance between best-fit and multiple-based scoring.
    
    Implementation idea: This function builds upon Node 30's adaptive blending by adding a
    real-time entropy calculation of the remaining capacity distribution to modulate
    heuristic weights. The entropy of feasible bins' post-placement capacities quantifies
    fragmentation risk: low entropy (many similar residuals) triggers increased emphasis
    on multiple-based scoring to promote diverse residuals, while high entropy allows
    best-fit to dominate. This complements the item-size-based weighting with a system-level
    state measure to better manage bin fragmentation throughout the packing process.
    
    Args:
        item: Size of the item to place
        bins_remain_cap: NumPy array of remaining bin capacities
        
    Returns:
        NumPy array of priority scores for each bin
    """
    # Calculate relevant state features
    if len(bins_remain_cap) > 0:
        max_capacity = np.max(bins_remain_cap) if np.any(bins_remain_cap > 0) else item * 2
    else:
        max_capacity = item * 2
    
    # Feature: Item size relative to bin capacity (detects large vs small items)
    item_capacity_ratio = item / max_capacity
    
    # Calculate post-placement remaining capacities
    post_placement_caps = bins_remain_cap - item
    
    # Initialize scores
    scores = np.full_like(bins_remain_cap, -np.inf, dtype=float)
    
    # Identify feasible bins (those that can accommodate the item)
    feasible_bins = bins_remain_cap >= item
    
    if not np.any(feasible_bins):
        return scores  # All bins remain with -inf scores
    
    # Extract feasible capacities
    feasible_post_caps = post_placement_caps[feasible_bins]
    
    # Calculate post-placement remaining capacities for the feasible bins
    feasible_post_caps_original = post_placement_caps[feasible_bins]
    
    # Calculate entropy of the remaining capacity distribution to measure fragmentation risk
    # First, normalize the capacities to create a probability distribution
    positive_caps = feasible_post_caps_original[feasible_post_caps_original > 0]  # Only consider positive capacities
    if len(positive_caps) > 0:
        # Create a normalized distribution by dividing by sum
        cap_sum = np.sum(positive_caps)
        if cap_sum > 0:
            prob_dist = positive_caps / cap_sum
            # Calculate entropy H = -sum(p * log(p)) where p > 0
            non_zero_probs = prob_dist[prob_dist > 0]
            entropy = -np.sum(non_zero_probs * np.log(non_zero_probs + 1e-10))  # Add small epsilon to prevent log(0)
            # Normalize entropy by max possible entropy (log(n)) to keep in [0, 1] range
            max_possible_entropy = np.log(len(positive_caps)) if len(positive_caps) > 1 else 1e-10
            normalized_entropy = entropy / max_possible_entropy if max_possible_entropy > 0 else 0
        else:
            normalized_entropy = 0
    else:
        # When all remaining capacities are zero or no feasible bins exist beyond those already checked
        normalized_entropy = 0
    
    # Keep the original feasible post placement caps for later calculations
    feasible_post_caps = feasible_post_caps_original
    
    # Component 1: Best-fit scoring (higher score for less remaining space after placement)
    best_fit_scores = -feasible_post_caps  # Higher score for less remaining space
    
    # Component 2: Multiple-based scoring (enhanced version)
    # Use a broader range of estimated common sizes based on the parent solution approach
    estimated_common_sizes = [item, item * 0.9, item * 1.1, item * 0.5, item * 1.5, item * 0.75, item * 1.25]
    multiple_scores = np.zeros_like(feasible_post_caps, dtype=float)
    
    for common_size in estimated_common_sizes:
        if common_size > 0:
            # Calculate distance to nearest multiple
            quotients = feasible_post_caps / common_size
            rounded_quotients = np.round(quotients)
            distances_to_multiples = np.abs(feasible_post_caps - rounded_quotients * common_size)
            # Add to the score (higher score for closer matches)
            multiple_scores += 1.0 / (1.0 + distances_to_multiples)
    
    # Dynamic weight calculation based on both item characteristics and entropy
    # Base weights determined by item-to-capacity ratio
    if item_capacity_ratio > 0.5:
        base_w_bestfit = 0.8
        base_w_multiple = 0.2
    elif item_capacity_ratio > 0.2:
        base_w_bestfit = 0.6
        base_w_multiple = 0.4
    else:
        base_w_bestfit = 0.3
        base_w_multiple = 0.7
    
    # Adjust weights based on entropy to control fragmentation
    # Low entropy indicates high fragmentation risk (similar residual capacities)
    # So we increase the weight of multiple-based scoring to promote diverse residuals
    entropy_influence = 1.0 - normalized_entropy  # Higher influence when entropy is low
    entropy_adjustment = 0.2 * entropy_influence  # Max adjustment of 0.2
    
    # Adjust weights: decrease best-fit weight and increase multiple weight when entropy is low
    adjusted_w_bestfit = max(0.1, base_w_bestfit - entropy_adjustment)
    adjusted_w_multiple = min(0.9, base_w_multiple + entropy_adjustment)
    
    # Ensure weights sum to 1.0
    total_weight = adjusted_w_bestfit + adjusted_w_multiple
    if total_weight > 0:
        adjusted_w_bestfit /= total_weight
        adjusted_w_multiple /= total_weight
    
    # Normalize components to similar scales before combining
    def safe_normalize(arr):
        std_val = np.std(arr)
        if std_val > 0:
            mean_val = np.mean(arr)
            return (arr - mean_val) / (std_val + 1e-8)
        return arr - np.mean(arr)
    
    best_fit_scores_norm = safe_normalize(best_fit_scores)
    multiple_scores_norm = safe_normalize(multiple_scores)
    
    # Combine components with adaptive weights adjusted by entropy
    combined_scores = adjusted_w_bestfit * best_fit_scores_norm + adjusted_w_multiple * multiple_scores_norm
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = combined_scores
    
    return scores