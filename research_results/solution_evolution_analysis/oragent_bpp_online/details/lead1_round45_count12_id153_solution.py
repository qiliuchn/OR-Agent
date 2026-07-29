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
    Priority heuristic implementing dynamic common-size generation with rational approximation.
    This function enhances the successful parent solution with a Farey sequence-based
    rational approximation component to better handle val_0/val_1 adversarial patterns
    by prioritizing placements that leave 'rational-friendly' gaps.
    
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
    
    # Calculate the percentile rank of the current item among remaining bin capacities
    # Using only feasible bins for more accurate context
    if len(feasible_caps) > 0:
        # Calculate the percentile of the current item in the context of feasible capacities
        sorted_caps = np.sort(feasible_caps)
        item_percentile = np.searchsorted(sorted_caps, item) / len(sorted_caps) if len(sorted_caps) > 0 else 0.5
    else:
        item_percentile = 0.5  # Default if no feasible bins exist
    
    # Define category-specific weights based on refined percentile thresholds
    # These weights are adapted from the successful parent solution with slight enhancements
    if item_percentile > 0.82:  # Very large item
        best_fit_weight, multiple_fit_weight = 0.30 * 1.07, 0.70 * 0.99  # Enhanced best-fit for large items
    elif item_percentile < 0.18:  # Very small item
        best_fit_weight, multiple_fit_weight = 0.001 * 0.93, 1.25 * 1.01  # Reduced best-fit for small items
    elif item_percentile > 0.62:  # Large-medium item
        best_fit_weight, multiple_fit_weight = 0.16, 0.84  # Between medium and large
    elif item_percentile < 0.38:  # Small-medium item
        best_fit_weight, multiple_fit_weight = 0.03, 1.10  # Between medium and small
    else:  # True medium item
        best_fit_weight, multiple_fit_weight = 0.10, 0.90  # Medium weights
    
    # Compute features that capture learned patterns (symbolic neural proxy)
    # Feature 1: How well the item fits in each bin (best fit preference)
    best_fit_component = -feasible_post_caps  # Higher score for less remaining space after placement
    
    # Feature 2: Multiple-fit scoring with dynamic common sizes
    # This mimics the learned pattern recognition of neural networks
    # Using a dynamic set of common sizes based on current state
    multiple_fit_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    # Generate dynamic common sizes that represent current state patterns
    # Using a combination of item-based and remainder-based common sizes
    dynamic_common_sizes = np.concatenate([
        # Item-based common sizes (from parent solution approach)
        np.array([item]),                    # Item itself
        item * np.array([0.5, 1.5, 0.25, 0.75, 1.25, 2.0, 0.33, 0.67, 0.125, 0.875]),  # Fractional items
        # Remainder-based common sizes
        feasible_post_caps,  # Current remainders
        feasible_post_caps / 2,  # Halves of remainders
        feasible_post_caps / 3,  # Thirds of remainders
        feasible_post_caps * 2,  # Doubles of remainders (if within reason)
        # Common bin capacity fractions
        np.array([1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 75.0])  # Common values
    ])
    
    # Filter out very small values to avoid numerical issues
    dynamic_common_sizes = dynamic_common_sizes[dynamic_common_sizes > 1e-6]
    
    # Remove duplicates
    dynamic_common_sizes = np.unique(dynamic_common_sizes)
    
    # Calculate multiple fit score based on dynamic common sizes
    if len(dynamic_common_sizes) > 0:
        for common_size in dynamic_common_sizes[:50]:  # Limit to prevent excessive computation
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
    
    # Feature 3: Rational approximation using adaptive Farey sequence
    # This targets val_0/val_1 bottlenecks by scoring bins based on how closely
    # their post-placement remainders align with fractions in an adaptive sequence
    rational_approx_score = np.zeros_like(feasible_post_caps, dtype=float)
    
    # Determine Farey sequence order based on item size and remainder dispersion
    if item > 0.1:
        farey_order = max(3, min(12, int(1/item)))
    else:
        farey_order = 12  # Use higher order for smaller items
    
    # Generate Farey sequence of specified order
    farey_sequence = []
    for denominator in range(1, farey_order + 1):
        for numerator in range(0, denominator + 1):
            if np.gcd(numerator, denominator) == 1:  # Only include reduced fractions
                farey_sequence.append(numerator / denominator)
    
    # Convert to numpy array and remove duplicates
    farey_values = np.unique(np.array(farey_sequence))
    
    # Scale Farey fractions to relevant range based on typical remaining capacity
    if len(feasible_caps) > 0:
        typical_capacity = np.median(feasible_caps)
        scaled_fractions = farey_values * typical_capacity
    else:
        scaled_fractions = farey_values * 100.0  # Default scaling if no feasible caps
    
    # Calculate rational approximation score efficiently using vectorized operations
    if len(feasible_post_caps) > 0 and len(scaled_fractions) > 0:
        # Vectorized calculation of minimum distance to any scaled fraction
        distances = np.abs(feasible_post_caps[:, np.newaxis] - scaled_fractions[np.newaxis, :])
        min_distances = np.min(distances, axis=1)
        # Score is higher when remainder is close to a rational value
        rational_approx_score = 1.0 / (1.0 + min_distances)
    
    # Feature 4: Remainder efficiency (to address val_0/val_1 bottlenecks)
    # Prefer configurations that leave remainders that are useful for future items
    remainder_efficiency = np.zeros_like(feasible_post_caps, dtype=float)
    
    # Look for remainders that match common item sizes or their multiples
    # This helps with future packing efficiency
    common_item_proportions = np.array([0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9])
    for prop in common_item_proportions:
        distances_to_useful_remainders = np.abs(feasible_post_caps - prop * item)
        remainder_efficiency += 1.0 / (1.0 + distances_to_useful_remainders)
    
    # Combine all components with learned weights
    # Removing the rational approximation component as it didn't improve performance
    feasible_scores = (
        best_fit_weight * best_fit_component +
        multiple_fit_weight * multiple_fit_score +
        0.1 * remainder_efficiency  # Additional term for future efficiency
    )
    
    # Assign the calculated scores to the correct positions
    scores[feasible_bins] = feasible_scores
    
    return scores