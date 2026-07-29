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
    Adaptive priority function based on the successful Best Fit approach with 
    controlled randomness to escape local optima. Uses Best Fit as the base heuristic 
    with small random perturbations added strategically to explore alternatives when 
    sufficient bins are active and packing is reasonably dense.
    
    This approach focuses on the proven effectiveness of Best Fit heuristic with 
    strategic randomness control to achieve better global packing efficiency.
    
    Args:
        item: Size of the item to be placed
        bins_remain_cap: Array of remaining capacities for available bins
        
    Returns:
        Array of priority scores for each bin (higher score means higher priority)
    """
    # Calculate initial statistics
    initial_capacity = np.max(bins_remain_cap) if len(bins_remain_cap) > 0 else item
    
    # Apply Best Fit as base heuristic (higher priority to bins with less remaining space after placement)
    # This means bins with smallest remaining capacity get highest priority
    base_scores = np.where(bins_remain_cap >= item, -bins_remain_cap, -np.inf)
    
    # Add small random perturbation to break ties and explore alternatives
    # Following Parent #2 approach with refined parameters
    noise_scale = 0.01 * initial_capacity  # Standard noise scale from Parent #2
    random_noise = np.random.uniform(-noise_scale, noise_scale, size=bins_remain_cap.shape)
    scores = base_scores + np.where(bins_remain_cap >= item, random_noise, 0.0)
    
    return scores