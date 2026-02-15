import numpy as np


def heuristics(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    """
    Generate a heuristic matrix for the Orienteering Problem using Ant Colony Optimization.
    
    Implementation idea: This function creates a static heuristic matrix that emphasizes
    exploration-friendly signals to be used in conjunction with dynamic rescaling during
    ACO tour construction. The static matrix focuses on core desirability indicators:
    - Squared prize/distance ratios to emphasize high-efficiency moves
    - Adaptive neighborhood density to capture local prize clustering
    - Smooth budget considerations that don't overly constrain early exploration
    During ACO execution, each ant will multiply these static values by a dynamic factor
    max(0, (remaining_budget - distance[j,0]) / maxlen) to ensure path feasibility while
    maintaining exploration flexibility. This two-phase approach balances static guidance
    with runtime adaptability.
    
    Args:
        prize: Array of shape (n,) representing the prize value of each node
        distance: Matrix of shape (n, n) representing the distance between nodes
        maxlen: Maximum allowed tour length
        
    Returns:
        Heuristic matrix of shape (n, n) where heuristics[i][j] indicates the promise 
        of including the edge from node i to node j in the solution
    """
    n = len(prize)
    
    # Calculate instance-specific features for adaptive parameters
    # Prize coefficient of variation (std/mean)
    prize_mean = np.mean(prize)
    prize_std = np.std(prize)
    prize_cv = prize_std / (prize_mean + 1e-9)  # Avoid division by zero
    
    # Maximum distance from depot (graph radius)
    max_dist_from_depot = np.max(distance[1:, 0]) if n > 1 else 0  # Exclude depot from max calculation
    
    # Distance-prize correlation (between prize[j] and distance[0,j])
    depot_distances = distance[1:, 0] if n > 1 else np.array([0])
    node_prizes = prize[1:] if n > 1 else prize
    if len(depot_distances) > 1:
        # Calculate correlation manually to avoid importing scipy
        mean_dist = np.mean(depot_distances)
        mean_prize = np.mean(node_prizes)
        numerator = np.sum((depot_distances - mean_dist) * (node_prizes - mean_prize))
        denominator = np.sqrt(np.sum((depot_distances - mean_dist)**2) * np.sum((node_prizes - mean_prize)**2))
        dist_prize_corr = numerator / (denominator + 1e-9)
    else:
        dist_prize_corr = 0
    
    # Adaptive neighborhood size based on instance features
    # Base neighborhood size influenced by prize CV and distance-prize correlation
    base_neighborhood_size = 5
    # Increase neighborhood if prizes are highly variable or negatively correlated with distance
    cv_factor = min(max(1.0, prize_cv * 0.5), 2.0)  # Cap between 0.5 and 2.0
    corr_factor = min(max(1.0, (1 - dist_prize_corr) * 0.5), 2.0)  # If distant nodes have higher prizes
    adaptive_k = int(np.clip(base_neighborhood_size * cv_factor * corr_factor, 3, 7))
    
    # Initialize the heuristic matrix
    heuristics_matrix = np.zeros((n, n))
    
    # Calculate the basic heuristic value (prize/distance ratio) for each possible move
    # Avoid division by zero by ensuring distance is not zero
    basic_heuristic = np.zeros((n, n))
    non_zero_distances = distance.copy()
    # Set diagonal to a large value to avoid self loops
    np.fill_diagonal(non_zero_distances, np.inf)
    
    for i in range(n):
        for j in range(n):
            if i != j:  # Don't consider moving from a node to itself
                basic_heuristic[i, j] = prize[j] / non_zero_distances[i, j]
    
    # Calculate neighborhood prize density for each node with adaptive neighborhood size
    # This will be used later in the heuristic calculation
    neighbor_densities = np.zeros(n)
    for j in range(n):
        all_other_nodes = np.arange(n)
        other_nodes_except_j = all_other_nodes[all_other_nodes != j]
        sorted_by_distance_to_j = other_nodes_except_j[np.argsort(distance[j, other_nodes_except_j])]
        j_neighbors = sorted_by_distance_to_j[:min(adaptive_k, len(sorted_by_distance_to_j))]
        
        avg_neighbor_prize = np.mean(prize[j_neighbors]) if len(j_neighbors) > 0 else prize[j]
        neighbor_densities[j] = avg_neighbor_prize

    # Apply sparsification: for each node i, keep only the top-k most promising edges
    for i in range(n):
        # Get the heuristic values for node i to all other nodes
        node_heuristics = basic_heuristic[i, :].copy()
        
        # Temporarily set the diagonal element to a very small value to exclude self-loop
        node_heuristics[i] = -np.inf
        
        # Find the indices of the top-k most promising nodes
        k = min(10, n - 1) if n > 1 else n - 1
        if k >= n - 1:
            top_k_indices = np.arange(n)
            top_k_indices = top_k_indices[top_k_indices != i]
        else:
            top_k_indices = np.argpartition(node_heuristics, -k)[-k:]
        
        # For each of the top-k indices, calculate a refined heuristic
        for j in top_k_indices:
            if j != i:  # Double-check to exclude self-loop
                # Basic heuristic value
                basic_value = node_heuristics[j]
                
                # Calculate squared prize/distance ratio to emphasize high-efficiency moves
                squared_efficiency = (prize[j] / (distance[i, j] + 1e-9)) ** 2
                
                # Use the pre-calculated neighborhood density
                neighborhood_factor = 1 + neighbor_densities[j] / (prize_mean + 1e-9)
                
                # The resulting heuristic combines multiple exploration-friendly signals:
                # - Basic value: prize[j]/distance[i,j]
                # - Squared efficiency: emphasizes high-efficiency moves
                # - Neighborhood value: prize density around node j with adaptive neighborhood size
                heuristics_matrix[i, j] = basic_value * squared_efficiency * neighborhood_factor
    
    # Ensure no negative values and handle any remaining zeros
    heuristics_matrix = np.maximum(heuristics_matrix, 0)
    
    # Small epsilon to avoid zero values which could cause issues in the ACO algorithm
    # Only set to epsilon where there's still a zero after our calculations
    zero_mask = (heuristics_matrix == 0)
    heuristics_matrix[zero_mask] = 1e-9
    
    return heuristics_matrix