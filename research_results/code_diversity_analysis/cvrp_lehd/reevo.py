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
    A minimal yet effective heuristic function that balances:
    1. Distance efficiency (inverse distance)
    2. Demand considerations for capacity planning
    3. Depot connectivity for proper route formation
    """
    n = distance_matrix.shape[0]
    
    # Avoid division by zero
    eps = 1e-8
    
    # Base inverse distance heuristic (higher for closer nodes)
    inv_distances = 1.0 / (distance_matrix + eps)
    inv_distances.fill_diagonal_(0.0)
    
    # Demand cost: penalize connections between high-demand nodes to preserve capacity
    demand_cost = demands.unsqueeze(0) + demands.unsqueeze(1)
    
    # Depot connectivity bonuses to ensure routes start and end properly
    depot_idx = 0
    depot_bonus = torch.zeros_like(inv_distances)
    depot_bonus[depot_idx, 1:] = 2.0  # Strong incentive for routes to start at depot
    depot_bonus[1:, depot_idx] = 1.2  # Moderate incentive for routes to return to depot
    
    # Combine components with balanced weights
    combined_heuristic = inv_distances * 2.5 - demand_cost * 0.5 + depot_bonus
    
    # Apply tanh for stable, bounded output
    combined_heuristic = torch.tanh(combined_heuristic * 0.4)
    
    # Slightly amplify depot connection preferences
    combined_heuristic[depot_idx, 1:] *= 1.1
    combined_heuristic[1:, depot_idx] *= 1.1
    
    # Ensure diagonal remains zero
    combined_heuristic.fill_diagonal_(0.0)
    
    return combined_heuristic