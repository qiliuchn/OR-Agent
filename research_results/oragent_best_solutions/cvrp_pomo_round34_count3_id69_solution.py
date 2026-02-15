import torch

def heuristics(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    """
    Implementation idea: 
    Enhance depot connectivity with return-aware efficiency: instead of treating 
    depot-to-node and node-to-depot symmetrically, model round-trip efficiency 
    as (demand_i) / (d_0i + d_i0 + η) to capture asymmetric routing costs, and 
    bias edges that enable efficient 'in-and-out' depot access. This better 
    reflects real route structure in CVRP where routes begin and end at the depot.
    Building on parent solutions, we incorporate scale-invariant normalization 
    and bounded fusion while emphasizing depot-round-trip efficiency as a key 
    differentiable component that guides the neural policy toward routes that 
    naturally begin and end at the depot with efficient demand fulfillment.
    
    Args:
        distance_matrix: A tensor of shape (n, n) representing distances between all pairs of nodes
        demands: A tensor of shape (n,) representing the demand at each node
        
    Returns:
        A tensor of shape (n, n) with heuristic values for each edge
    """
    n = distance_matrix.size(0)
    eps = 1e-8
    
    # Use median-based normalization instead of max to reduce outlier sensitivity
    # Only consider upper triangular part (excluding diagonal) to avoid double counting
    dist_values = distance_matrix[torch.triu(torch.ones_like(distance_matrix), diagonal=1) == 1]
    median_dist = torch.median(dist_values) if dist_values.numel() > 0 else torch.mean(dist_values)
    
    # If median is too small, fall back to mean to avoid extreme scaling
    norm_factor = median_dist if median_dist > 1e-6 else torch.mean(dist_values)
    norm_distances = distance_matrix / (norm_factor + eps)
    
    # Calculate demand pressure for each node
    # Use a more stable proximity measure than pure inverse distance
    # Cap the inverse distances to prevent extreme values
    max_inv_dist = 1.0 / (norm_distances + eps)
    capped_inv_dist = torch.clamp(max_inv_dist, max=100.0)  # Cap to prevent extreme values
    # Zero out diagonal to avoid self-influence
    capped_inv_dist.fill_diagonal_(0)
    
    # Calculate weighted sum of neighbor demands
    neighbor_influence = torch.matmul(capped_inv_dist, demands.unsqueeze(1)).squeeze(1)
    # Total demand pressure = node's own demand + neighbor influences
    demand_pressure = demands + neighbor_influence
    
    # Create a demand pressure matrix
    pressure_matrix = demand_pressure[:, None] + demand_pressure[None, :]
    
    # Calculate a capacity feasibility score for each edge
    # Lower scores indicate higher risk of capacity violation
    capacity_penalty = pressure_matrix * norm_distances
    
    # Create a distance-demand ratio component (similar to parent solution)
    demand_sum = demands[:, None] + demands[None, :]
    # Normalize to similar scale as other components
    distance_demand_component = -norm_distances * (1 + 0.5 * demand_sum)
    
    # Normalize this component to prevent dominance
    dd_max = torch.max(torch.abs(distance_demand_component))
    distance_demand_component = distance_demand_component / (dd_max + eps) if dd_max > 0 else distance_demand_component
    
    # Depot-specific considerations - enhanced with return-aware efficiency
    # Encourage connections from depot to nodes with reasonable demand
    depot_connectivity = torch.zeros_like(distance_matrix)
    
    # Calculate depot round-trip efficiency: demand_i / (d_0i + d_i0 + eta)
    # This captures the efficiency of going from depot to node and back
    depot_to_node_distances = norm_distances[0, :]  # d_0i for all i
    node_to_depot_distances = norm_distances[:, 0]  # d_i0 for all i
    # For each potential edge (i,j), calculate the round-trip efficiency if it were part of a depot round-trip
    # Actually, let's focus on individual node round-trips: d_0i + d_i0 for each node i
    node_round_trip_distances = depot_to_node_distances + node_to_depot_distances  # Shape: (n,)
    # Calculate efficiency for visiting each node: demand_i / (d_0i + d_i0)
    node_depot_efficiency = demands / (node_round_trip_distances + eps)  # Shape: (n,)
    
    # Enhance depot connectivity with round-trip efficiency
    # Bias edges that are part of efficient depot round trips
    depot_to_nodes = node_depot_efficiency[None, :]  # From depot to each node based on round-trip efficiency
    nodes_to_depot = node_depot_efficiency[:, None]  # To depot from each node based on round-trip efficiency
    
    depot_connectivity[0, :] = depot_to_nodes[0, :]  # Connections from depot
    depot_connectivity[:, 0] = nodes_to_depot[:, 0]  # Connections to depot
    
    # Also add the round-trip efficiency as an additional bias for general edges
    # Create a matrix where each element (i,j) represents some combination of round-trip efficiency
    depot_connectivity += torch.outer(node_depot_efficiency, node_depot_efficiency) * 0.1  # Scale factor to prevent overwhelming
    
    # Normalize depot connectivity
    depot_max = torch.max(torch.abs(depot_connectivity))
    depot_connectivity = depot_connectivity / (depot_max + eps) if depot_max > 0 else depot_connectivity
    
    # Demand efficiency term: ratio of combined demand to distance
    demand_efficiency = (demands[:, None] + demands[None, :]) / (norm_distances + eps)
    
    # Normalize efficiency term
    eff_max = torch.max(torch.abs(demand_efficiency))
    demand_efficiency = demand_efficiency / (eff_max + eps) if eff_max > 0 else demand_efficiency
    
    # Depot access efficiency - how efficiently a node can be accessed from/to depot
    # Now incorporating the round-trip concept
    depot_access_efficiency = torch.zeros_like(distance_matrix)
    depot_access_efficiency[0, :] = demands[:] / (norm_distances[0, :] + eps)  # From depot
    depot_access_efficiency[:, 0] = demands[:] / (norm_distances[:, 0] + eps)  # To depot
    
    # Normalize depot access efficiency
    access_max = torch.max(torch.abs(depot_access_efficiency))
    depot_access_efficiency = depot_access_efficiency / (access_max + eps) if access_max > 0 else depot_access_efficiency
    
    # Combine components with balanced weights
    # Adjust weights to potentially improve scaling across problem sizes
    alpha = 0.18   # Weight for distance-demand ratio
    beta = 0.22    # Weight for capacity penalty (negative as it's a penalty) - slightly increased importance
    gamma = 0.32   # Increased weight for depot connectivity (with round-trip efficiency)
    delta = 0.28   # Weight for demand efficiency
    
    heuristic_matrix = (
        alpha * distance_demand_component -
        beta * capacity_penalty +
        gamma * depot_connectivity +
        delta * demand_efficiency
    )
    
    # Enhance depot connections with the round-trip efficiency concept
    # Add specific biases for depot access that consider round-trip costs
    depot_round_trip_costs = depot_to_node_distances[:, None] + node_to_depot_distances[None, :]
    # Higher heuristic values for edges that are part of low-cost round trips
    round_trip_bonus = 0.05 / (depot_round_trip_costs + eps)
    heuristic_matrix += round_trip_bonus
    
    # Prevent self-loops by setting diagonal to large negative value
    diag_indices = torch.arange(n)
    heuristic_matrix[diag_indices, diag_indices] = -1e6
    
    # Apply final bounded activation to ensure numerical stability
    # Using tanh to bound to [-1, 1] then scale to [-100, 100] range
    heuristic_matrix = torch.tanh(heuristic_matrix) * 100.0
    
    # Ensure no NaN or inf values
    heuristic_matrix = torch.nan_to_num(heuristic_matrix, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return heuristic_matrix