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
    Heuristic function to guide attention mechanism in CVRP.
    
    Args:
        distance_matrix: n x n tensor of distances between nodes
        demands: n-dim tensor of customer demands (normalized by vehicle capacity)
    
    Returns:
        n x n tensor of attention biases (positive for promising edges, negative for undesirable ones)
    """
    device = distance_matrix.device
    n = distance_matrix.size(0)
    
    # Avoid division by zero
    eps = 1e-8
    
    # Invert distances so closer nodes have higher values
    inv_distances = 1.0 / (distance_matrix + eps)
    
    # Mask out self loops
    mask = (1 - torch.eye(n, device=device))
    inv_distances = inv_distances * mask
    
    # Calculate demand density around each node (excluding depot)
    demands_no_depot = demands.clone()
    demands_no_depot[0] = 0  # Zero out depot demand for density calculation
    
    # Weighted sum of demands based on proximity (using Gaussian-like kernel)
    mean_distance = torch.mean(distance_matrix[distance_matrix > 0])  # Exclude zeros for mean
    distance_weights = torch.exp(-distance_matrix / (mean_distance + eps))
    demand_density = torch.matmul(distance_weights, demands_no_depot)
    
    # Create factors for i and j nodes
    demand_density_i = demand_density.unsqueeze(1).expand(-1, n)
    demand_density_j = demand_density.unsqueeze(0).expand(n, -1)
    avg_demand_density = (demand_density_i + demand_density_j) / 2.0
    
    # Consider individual demands as well as densities
    demands_i = demands.unsqueeze(1).expand(-1, n)
    demands_j = demands.unsqueeze(0).expand(n, -1)
    avg_demands = (demands_i + demands_j) / 2.0
    
    # Compute nearest neighbors for each node (excluding depot initially)
    # Find k nearest neighbors for each node (k=3)
    k = min(3, n-1)
    _, nearest_neighbor_indices = torch.topk(distance_matrix, k+1, dim=1, largest=False)  # Include self
    nearest_neighbor_indices = nearest_neighbor_indices[:, 1:]  # Exclude self
    
    # Create a mask for being among top-k nearest neighbors
    neighbor_mask = torch.zeros_like(distance_matrix)
    batch_idx = torch.arange(n, device=device).unsqueeze(1).expand(-1, k)
    neighbor_mask[batch_idx, nearest_neighbor_indices] = 1.0
    # Make symmetric
    neighbor_mask = torch.maximum(neighbor_mask, neighbor_mask.transpose(0, 1))
    
    # Base heuristic: combine inverse distances with demand information
    base_heuristic = inv_distances * (1.0 + 0.3 * avg_demand_density + 0.5 * avg_demands)
    
    # Enhanced heuristic for nearby nodes with good demand properties
    enhanced_nearby = neighbor_mask * inv_distances * (1.0 + 0.4 * avg_demand_density + 0.6 * avg_demands)
    
    # Combine base and enhanced heuristic
    heuristic = 0.7 * base_heuristic + 0.3 * enhanced_nearby
    
    # Special consideration for depot connections
    depot_connections = torch.zeros_like(heuristic)
    depot_connections[0, :] = 1.0  # From depot
    depot_connections[:, 0] = 1.0  # To depot
    depot_connections[0, 0] = 0.0  # No self-loop
    
    # Depot connections are critical, especially to high-demand nodes
    depot_to_high_demand = (demands_j * depot_connections) * inv_distances
    depot_from_high_demand = (demands_i * depot_connections) * inv_distances
    depot_bonus = 0.3 * depot_to_high_demand + 0.3 * depot_from_high_demand
    
    heuristic = heuristic + depot_bonus
    
    # Penalize edges that would likely violate capacity constraints
    # When connecting two high-demand nodes directly (not involving depot)
    non_depot_mask = 1.0 - depot_connections
    high_demand_penalty = torch.relu(demands_i + demands_j - 1.0)  # Penalty when sum > 1.0
    heuristic = heuristic - 1.0 * high_demand_penalty * non_depot_mask  # Only apply to non-depot edges
    
    # Reward edges that connect nodes with complementary demands (sum close to 1.0 without exceeding)
    # This promotes efficient use of vehicle capacity
    demand_sum = demands_i + demands_j
    complementary_reward = torch.relu(1.0 - torch.abs(demand_sum - 1.0))  # Peak at sum=1.0
    complementary_reward = torch.where(demand_sum <= 1.0, complementary_reward, torch.zeros_like(complementary_reward))
    heuristic = heuristic + 0.3 * complementary_reward * inv_distances * non_depot_mask
    
    # Reward connecting nodes that have high combined demand density but are not too far
    combined_density = avg_demand_density * inv_distances
    density_bonus = torch.relu(combined_density - torch.quantile(combined_density.flatten(), 0.8)) * non_depot_mask
    heuristic = heuristic + 0.2 * density_bonus
    
    # Strategic heuristic: Prioritize connecting high-demand nodes that are close to each other
    # but only when their combined demand doesn't exceed capacity
    high_demand_close_bonus = torch.where(
        (demands_i + demands_j <= 1.0) & (demands_i >= 0.3) & (demands_j >= 0.3),
        inv_distances * 0.5,
        torch.zeros_like(inv_distances)
    )
    heuristic = heuristic + high_demand_close_bonus * non_depot_mask
    
    # Encourage connections between nodes with moderate demands that are nearby
    moderate_demands = torch.where(
        (demands_i > 0.1) & (demands_i < 0.7) & (demands_j > 0.1) & (demands_j < 0.7),
        torch.ones_like(inv_distances),
        torch.zeros_like(inv_distances)
    )
    moderate_bonus = 0.15 * inv_distances * neighbor_mask * moderate_demands * non_depot_mask
    heuristic = heuristic + moderate_bonus
    
    # Apply mask again after adjustments
    heuristic = heuristic * mask
    
    # Normalize to prevent overwhelming the learned parameters
    max_val = torch.max(torch.abs(heuristic))
    if max_val > eps:
        heuristic = heuristic / max_val * 6.0  # Scale to moderate range
    
    return heuristic