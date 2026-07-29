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
    Generate attention bias matrix based on distance and demand considerations.
    
    Args:
        distance_matrix: Tensor of shape (n, n) representing distances between nodes
        demands: Tensor of shape (n,) representing normalized customer demands
    
    Returns:
        Tensor of shape (n, n) with heuristic scores for each edge
    """
    n = distance_matrix.shape[0]
    device = distance_matrix.device
    
    # Avoid division by zero
    eps = 1e-8
    
    # Inverse distance heuristic - closer nodes are better
    inv_distances = 1.0 / (distance_matrix + eps)
    
    # Create demand matrices
    demands_i = demands.unsqueeze(1).expand(-1, n)  # Demand at source
    demands_j = demands.unsqueeze(0).expand(n, -1)  # Demand at destination
    
    # Distance savings heuristic (Clarke-Wright type)
    depot_distances = distance_matrix[0, :].unsqueeze(1) + distance_matrix[0, :].unsqueeze(0)
    savings = depot_distances - distance_matrix
    
    # Mask for non-depot to non-depot connections for savings
    savings_mask = torch.ones_like(savings)
    savings_mask[0, :] = 0
    savings_mask[:, 0] = 0
    savings = savings * savings_mask
    
    # Capacity utilization heuristic - encourage connecting nodes that approach capacity without exceeding
    total_demand = demands_i + demands_j
    # Prefer pairs whose combined demand is close to but under capacity limit
    capacity_efficiency = torch.where(
        total_demand <= 1.0,
        torch.clamp(1.0 - torch.abs(0.9 - total_demand), min=0.0),  # Target slightly higher capacity utilization
        -20.0 * total_demand  # Heavier penalty if over capacity
    )
    
    # Ratio of combined demand to distance - higher ratios are more attractive
    demand_to_distance_ratio = (demands_i + demands_j + eps) / (distance_matrix + eps)
    
    # Spatial clustering heuristic - prefer connecting nearby customers when both are far from depot
    depot_dist_i = distance_matrix[0, :].unsqueeze(1).expand(-1, n)
    depot_dist_j = distance_matrix[0, :].unsqueeze(0).expand(n, -1)
    # When both nodes are far from depot relative to their mutual distance, connecting them makes sense
    spatial_preference = torch.where(
        (depot_dist_i + depot_dist_j) > 2.0 * distance_matrix,
        inv_distances * 3.0,
        inv_distances * 0.2
    )
    
    # Demand similarity heuristic - prefer connecting nodes with similar demand sizes
    demand_similarity = 1.0 / (torch.abs(demands_i - demands_j) + 0.2)
    
    # Local density heuristic - penalize connecting nodes in very dense regions
    avg_node_distance = torch.mean(distance_matrix, dim=1, keepdim=True)
    local_density_penalty = 1.0 / (avg_node_distance + avg_node_distance.t() + eps)
    
    # Nearest neighbor heuristic - prioritize connecting to nearest neighbors
    k_nearest_mask = torch.zeros_like(inv_distances)
    k = min(5, n-1)  # Number of nearest neighbors to consider
    _, top_k_indices = torch.topk(inv_distances, k=k+1, dim=1)  # +1 to exclude self
    for i in range(n):
        k_nearest_mask[i, top_k_indices[i, 1:k+1]] = 1.0  # Exclude self (first index)
        
    nearest_neighbor_bonus = k_nearest_mask * inv_distances * 2.0
    
    # Balance demand heuristic - prefer connecting nodes with complementary demands
    demand_complementarity = torch.where(
        total_demand <= 1.0,
        torch.sqrt(demands_i * demands_j) * torch.clamp(0.8 - torch.abs(0.6 - total_demand), min=0.0),
        torch.zeros_like(total_demand)
    )
    
    # Depot-specific logic: enhance attraction to depot for high-demand customers
    depot_attraction = torch.zeros_like(inv_distances)
    depot_attraction[0, 1:] = demands[1:] / (distance_matrix[0, 1:] + eps) * 3.0
    depot_attraction[1:, 0] = demands[1:] / (distance_matrix[1:, 0] + eps) * 3.0
    
    # Path extension heuristic - prefer connecting to nodes that would continue a route efficiently
    # Prioritize connections that don't create early return to depot unless necessary
    path_continuation = torch.where(
        (demands_i > 0) & (demands_j > 0),  # Both are customers
        inv_distances * 1.5,  # Boost connections between customers
        inv_distances * 0.6   # Lower priority for depot connections unless needed
    )
    
    # Combine heuristics with different weights
    result = (
        inv_distances * 1.2 +                    # Basic proximity
        savings * 1.8 +                          # Increased savings weight
        torch.relu(capacity_efficiency) * inv_distances * 3.0 +  # Capacity efficiency with distance consideration
        demand_to_distance_ratio * 1.2 +         # Demand-to-distance ratio weight
        spatial_preference * 0.8 +               # Spatial heuristic
        demand_similarity * 0.7 +                # Demand similarity
        local_density_penalty * 0.2 +            # Local density consideration
        nearest_neighbor_bonus * 1.0 +           # Nearest neighbor bonus
        demand_complementarity * 1.2 +           # Complementary demand preference
        depot_attraction * 1.5 +                 # Depot attraction for high-demand customers
        path_continuation * 0.6                   # Path continuation heuristic
    )
    
    # Apply heavy penalty for over-capacity combinations
    over_capacity_mask = (demands_i + demands_j > 1.0).float()
    result = result * (1 - over_capacity_mask) + result * (over_capacity_mask * -200.0)
    
    # Zero out diagonal (self-loops not allowed)
    result.fill_diagonal_(0)
    
    # Zero out depot to depot connection explicitly
    result[0, 0] = 0
    
    # Normalize to prevent extreme values while preserving relative relationships
    max_abs = result.abs().max()
    if max_abs > eps:
        result = result / max_abs * 8.0  # Increased normalization factor
    
    return result