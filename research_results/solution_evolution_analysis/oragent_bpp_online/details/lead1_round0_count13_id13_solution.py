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

def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """
    Lookahead-inspired heuristic: simulate placing the current item in each feasible bin 
    and estimate the expected number of additional bins needed for a synthetic future 
    stream (based on average item size), then pick the bin minimizing this estimate.
    
    This approach combines the best-fit principle with forward-looking estimation to 
    make more informed decisions about which bin to select. It considers not just the 
    immediate fit but also the potential impact on future bin usage.
    
    Args:
        item: Size of the current item to be placed
        bins_remain_cap: Array of remaining capacities in each bin
        
    Returns:
        Array of priority scores for each bin (higher score means higher priority)
    """
    # Initialize scores array
    scores = np.full_like(bins_remain_cap, -np.inf, dtype=float)
    
    # Identify bins that can accommodate the current item
    feasible_mask = bins_remain_cap >= item
    
    # If no bins are feasible, return the initialized scores (all -inf)
    if not np.any(feasible_mask):
        return scores
    
    # Calculate the remaining capacity after placing the item in each feasible bin
    remaining_after_placement = bins_remain_cap - item
    
    # For lookahead simulation, we'll estimate the efficiency of each placement
    # by considering how well the remaining space can potentially accommodate future items
    
    # We'll use a combination of best-fit principle and space utilization
    # Higher priority for bins that:
    # 1. Can fit the item (feasibility check already done)
    # 2. Have minimal leftover space after placement (best-fit aspect)
    # 3. Leave space that could efficiently accommodate potential future items
    
    # Base score based on remaining capacity (inverted for best-fit behavior)
    base_score = np.where(feasible_mask, -remaining_after_placement, -np.inf)
    
    # Additional heuristic: consider the utilization efficiency
    # We'll estimate how well each remaining capacity might fit an average future item
    # Using a placeholder average item size assumption for lookahead
    avg_item_estimate = 20.0  # Based on typical problem characteristics
    
    # Calculate how many average items could potentially fit in remaining space
    potential_items_fit = remaining_after_placement / avg_item_estimate
    
    # Use fractional part to understand "waste" - bins with capacity closer to integer multiples
    # of average item size may be more useful
    fractional_part = potential_items_fit - np.floor(potential_items_fit)
    
    # Reward bins that leave space more suitable for future items (less waste)
    # Bins with fractional parts closer to 0 are more efficient
    efficiency_bonus = -(fractional_part ** 2)  # Negative square to reward values closer to 0
    
    # Combine base score with efficiency bonus
    scores = base_score + 0.1 * efficiency_bonus  # Small weight for efficiency bonus
    
    # Further refine by considering the absolute amount of remaining space
    # Very small remaining space should be penalized less than medium space that's too small for most items
    small_space_bonus = np.where(
        remaining_after_placement < avg_item_estimate * 0.5,
        remaining_after_placement * 0.05,  # Small bonus for very small spaces
        -np.abs(remaining_after_placement - avg_item_estimate) * 0.01  # Penalty for medium-sized gaps
    )
    
    scores = scores + small_space_bonus
    
    return scores