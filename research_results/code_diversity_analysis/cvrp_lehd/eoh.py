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

import torch

def heuristics(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    """
    Generates an attention bias matrix using a simplified savings-based approach.
    
    Args:
        distance_matrix: Tensor of shape (n, n) representing distances between nodes
        demands: Tensor of shape (n,) representing normalized customer demands
    
    Returns:
        Tensor of shape (n, n) with attention biases for each edge
    """
    n = distance_matrix.shape[0]
    
    # Compute the classical Clarke-Wright savings for each edge
    depot_distances = distance_matrix[0, :]  # [n]
    savings = depot_distances.unsqueeze(0) + depot_distances.unsqueeze(1) - distance_matrix
    
    # Demand compatibility factor - higher when demands complement each other well
    demand_compatibility = (1 - torch.abs(demands.unsqueeze(0) - demands.unsqueeze(1))) * \
                          (1 - torch.clamp(demands.unsqueeze(0) + demands.unsqueeze(1) - 1, min=0))
    
    # Marginal benefit combining savings and demand compatibility
    marginal_benefit = savings * demand_compatibility
    
    # Capacity violation penalty
    capacity_violation = torch.where(
        demands.unsqueeze(0) + demands.unsqueeze(1) > 1.0,
        torch.ones_like(distance_matrix) * -10.0,
        torch.zeros_like(distance_matrix)
    )
    
    # Combine components
    heuristic = marginal_benefit + capacity_violation
    
    # Apply depot-specific adjustments
    # Encourage leaving depot for high-demand nodes
    heuristic[0, 1:] *= (1 + demands[1:])
    # Encourage returning to depot when load is high
    heuristic[1:, 0] *= (1 + 0.5 * demands[1:])
    
    # Zero out diagonal elements (no self loops)
    heuristic.fill_diagonal_(0.0)
    
    # Normalize to [-1, 1] range
    min_val = heuristic.min()
    max_val = heuristic.max()
    range_val = max_val - min_val
    if range_val > 0:
        heuristic = 2 * (heuristic - min_val) / range_val - 1
    else:
        heuristic = torch.zeros_like(heuristic)
    
    return heuristic