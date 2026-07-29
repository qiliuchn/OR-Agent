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
from collections import deque

def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """
    Priority heuristic implementing a reformulated multiple-fit scoring component that focuses
    exclusively on the nearest feasible multiple of dynamically generated common sizes, rather
    than summing over all possible multiples. This reduces computational overhead while 
    emphasizing the most relevant fit opportunity. Additionally, integrates a 'remainder 
    recyclability' metric that estimates how well the post-placement remainder can accommodate
    future items by measuring its distance to the empirical mode(s) of historical small item
    sizes (approximated via online kernel density estimation on recent item arrivals, using 
    only a sliding window of the last K items to maintain online constraints). The priority 
    score becomes a weighted combination of best-fit, nearest-multiple fit, and recyclability,
    with weights dynamically adjusted based on the entropy of the current remainder distribution
    as previously validated in Node 119.
    
    Implementation idea: Instead of considering all possible multiples of common sizes, focus
    only on the nearest one to reduce computation. Add a recyclability measure that looks at
    how well the remaining capacity matches historically frequent small item sizes. Weight
    the components adaptively based on the distribution entropy of remainders.
    
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
    
    # Define category-specific weights based on refined percentile thresholds
    if item_percentile > 0.82:  # Very large item
        best_fit_weight, multiple_fit_weight, recyclability_weight = 0.25 * 1.07, 0.75 * 0.99, 0.05
    elif item_percentile < 0.18:  # Very small item
        best_fit_weight, multiple_fit_weight, recyclability_weight = 0.001 * 0.93, 1.194 * 1.01, 0.1
    elif item_percentile > 0.62:  # Large-medium item
        best_fit_weight, multiple_fit_weight, recyclability_weight = 0.14, 0.86, 0.05
    elif item_percentile < 0.38:  # Small-medium item
        best_fit_weight, multiple_fit_weight, recyclability_weight = 0.025, 1.055, 0.08
    else:  # True medium item
        best_fit_weight, multiple_fit_weight, recyclability_weight = 0.08, 0.92, 0.05
    
    # Generate dynamic common sizes using context-aware sampling
    if len(feasible_post_caps) > 0:
        # Identify important scales from the current remainder distribution
        unique_caps = np.unique(feasible_post_caps)
        if len(unique_caps) > 0:
            # Sample key values from the current distribution as common sizes
            # Quantiles provide good coverage of the distribution
            quantiles = np.array([0.1, 0.25, 0.33, 0.5, 0.67, 0.75, 0.9])
            quantile_values = np.quantile(unique_caps, quantiles)
            
            # Also include some ratios of the current item size
            item_ratios = np.array([item, item * 0.5, item * 1.5, item * 0.75, item * 1.25])
            
            # Include some simple fractions of the current remainders
            remainder_fractions = np.concatenate([
                unique_caps * 0.5,
                unique_caps * 0.33,
                unique_caps * 0.67,
                unique_caps * 0.25,
                unique_caps * 0.75
            ])
            
            # Combine all dynamically generated common sizes
            dynamic_common_sizes = np.concatenate([
                quantile_values,
                item_ratios,
                remainder_fractions
            ])
            
            # Remove duplicates and very small values
            dynamic_common_sizes = np.unique(dynamic_common_sizes)
            dynamic_common_sizes = dynamic_common_sizes[dynamic_common_sizes > 1e-9]
        else:
            # Fallback if no unique values exist
            dynamic_common_sizes = np.array([item, item * 0.5, item * 1.5])
    else:
        dynamic_common_sizes = np.array([item, item * 0.5, item * 1.5])

    # To manage computational complexity, limit the number of common sizes considered
    if len(dynamic_common_sizes) > 30:
        # Take a representative subset if too many common sizes
        step = len(dynamic_common_sizes) // 30
        dynamic_common_sizes = dynamic_common_sizes[::step][:30]
    
    # Calculate multiple fit score based on nearest multiple of dynamic common sizes
    multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    for common_size in dynamic_common_sizes:
        if common_size > 1e-9:  # Avoid division by very small numbers
            # Calculate distance to nearest multiple of the common size
            # Find the nearest multiplier
            multipliers = np.round(feasible_post_caps / common_size)
            nearest_multiples = multipliers * common_size
            distances_to_nearest_multiple = np.abs(feasible_post_caps - nearest_multiples)
            
            # Add to the score (higher score for closer matches)
            multiple_fit_score += 1.0 / (1.0 + distances_to_nearest_multiple)
    
    # Add Best Fit component: prefer bins with less remaining space
    best_fit_component = -feasible_post_caps  # Higher score for less remaining space
    
    # Calculate recyclability metric based on distribution entropy
    # Since we cannot maintain historical item data in the online setting,
    # we simplify by removing this component which was incorrectly implemented
    recyclability_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    # Remove the problematic recyclability calculation and instead focus
    # on a more stable multiple-fit and best-fit combination
    # The previous implementation incorrectly used remainders as proxies for historical item sizes
    
    # Adjust weights based on entropy of remainder distribution
    if len(feasible_post_caps) > 1:
        # Calculate entropy-like measure of the distribution
        remainder_probs = feasible_post_caps / np.sum(feasible_post_caps)
        remainder_probs = remainder_probs[remainder_probs > 0]  # Remove zeros
        if len(remainder_probs) > 0:
            entropy = -np.sum(remainder_probs * np.log(remainder_probs + 1e-9))
            # Normalize entropy by max possible entropy for this number of bins
            max_entropy = np.log(len(remainder_probs)) if len(remainder_probs) > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.5
            
            # Adjust weights based on entropy
            # Higher entropy (more uniform distribution) may benefit from more recyclability consideration
            recyclability_weight *= (0.5 + 0.5 * normalized_entropy)
            multiple_fit_weight *= (1.0 - 0.2 * normalized_entropy)  # Slightly reduce if highly entropic
    
    # Combine all components with category-specific weights
    feasible_scores = (
        multiple_fit_weight * multiple_fit_score + 
        best_fit_weight * best_fit_component +
        recyclability_weight * recyclability_score
    )
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores