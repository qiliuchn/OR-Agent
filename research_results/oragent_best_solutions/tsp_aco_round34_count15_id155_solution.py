import numpy as np

def heuristics(distance_matrix: np.ndarray) -> np.ndarray:
    """
    TSP-ACO Heuristic using power-law distance weighting with bidirectional local significance and soft topological affinity.
    
    Implementation idea: Optimize the computational efficiency of the soft topological affinity calculation by 
    reformulating it as a matrix operation. Specifically, compute all softmax neighbor distributions in parallel 
    using broadcasting: for each city i, calculate P_i = exp(-D_i / τ_i) where D_i is the i-th row of the 
    distance matrix, then normalize each row to get a stochastic matrix P. The topological affinity matrix can 
    then be computed as the cosine similarity between rows of P using efficient linear algebra: 
    Topo = (P @ P.T) / (||P||_2 @ ||P||_2.T), where ||P||_2 is the L2 norm of each row. This reduces 
    complexity from O(n³) to O(n²) while preserving the differentiable, symmetric nature of the original 
    formulation, enabling application to larger TSP instances without performance degradation.
    
    Args:
        distance_matrix: A square matrix where element (i,j) represents the distance between city i and city j
        
    Returns:
        A heuristic matrix of the same shape as the input, where higher values indicate 
        more attractive edges for the ants to traverse
    """
    n_cities = distance_matrix.shape[0]
    
    # Set k as max(5, n/10) for k-nearest neighbors
    k = max(5, n_cities // 10)
    
    # Apply power-law transformation with beta=4 based on parent solution analysis
    beta = 4
    epsilon = 1e-10
    
    # Add small epsilon to avoid division by zero, particularly on diagonal
    distances_with_epsilon = distance_matrix + epsilon * np.eye(n_cities)
    
    # Compute the base heuristic: 1 / distance^beta
    base_heuristic = 1 / (distances_with_epsilon ** beta)
    
    # Create bidirectional context weights with sigmoid-based rank decay
    context_weights = np.zeros_like(distance_matrix)
    
    # For each city, precompute sorted indices to avoid repeated sorting
    sorted_indices_per_city = []
    for i in range(n_cities):
        sorted_indices = np.argsort(distance_matrix[i])
        sorted_indices_per_city.append(sorted_indices)
    
    # For each city, calculate bidirectional context weights
    for i in range(n_cities):
        sorted_indices_i = sorted_indices_per_city[i]
        min_dist_i = distance_matrix[i, sorted_indices_i[1]] if n_cities > 1 else 1.0  # Distance to nearest neighbor
        
        for j in range(n_cities):
            if i == j:
                continue
                
            # Find the rank of city j in city i's sorted neighbor list
            rank_ij = int(np.where(sorted_indices_i == j)[0][0])
            
            # Find the rank of city j in city i's sorted neighbor list (for i from j's perspective)
            sorted_indices_j = sorted_indices_per_city[j]
            rank_ji = int(np.where(sorted_indices_j == i)[0][0])
            
            # Compute average mutual rank
            avg_rank = (rank_ij + rank_ji) / 2.0
            
            # Apply sigmoid-based decay based on average rank
            # Using tau = k/4 as in the parent solution
            tau = k / 4.0
            sigmoid_input = (k - avg_rank) / tau
            avg_rank_decay = 1.0 / (1.0 + np.exp(-sigmoid_input))
            
            # Calculate bidirectional local significance: consider both cities' local significance
            min_dist_j = distance_matrix[j, sorted_indices_j[1]] if n_cities > 1 else 1.0
            local_significance_i = min_dist_i / (distance_matrix[i, j] + epsilon)
            local_significance_j = min_dist_j / (distance_matrix[i, j] + epsilon)
            
            # Combine the factors similar to parent solution for better performance
            # This includes a baseline context that gets modulated by the rank decay
            combined_factor = (0.8 + local_significance_i + local_significance_j) / 2.8
            
            # Apply the rank decay to the combined factor to ensure proper scaling
            context_weights[i, j] = avg_rank_decay * combined_factor
    
    # Efficiently compute the soft topological affinity using matrix operations
    topo_affinity = np.zeros_like(distance_matrix)
    
    # Calculate temperature parameter for each city as median of k-nearest neighbor distances
    temperatures = np.zeros(n_cities)
    for i in range(n_cities):
        sorted_dists_i = np.sort(distance_matrix[i])
        k_nearest_dists = sorted_dists_i[1:k+1]  # Skip self-distance at index 0
        temperatures[i] = np.median(k_nearest_dists)  # Using median as in parent solution
        
        # Avoid division by zero in case of very small temperatures
        if temperatures[i] < epsilon:
            temperatures[i] = epsilon
    
    # Calculate softmax-weighted neighbor distributions for all cities in parallel
    # Normalize distances by temperature for each row
    temp_matrix = np.tile(temperatures, (n_cities, 1))  # Shape: (n_cities, n_cities)
    normalized_distances = -distance_matrix / temp_matrix  # Element-wise division
    
    # Create a mask to zero out self-connections (diagonal elements)
    mask = 1 - np.eye(n_cities)
    exp_weights = np.exp(normalized_distances) * mask  # Zero out diagonal
    
    # Normalize each row to get probability distributions
    row_sums = np.sum(exp_weights, axis=1, keepdims=True)
    
    # Avoid division by zero - handle case where all neighbors have same distance
    # In such cases, assign uniform probability to all non-diagonal elements
    zero_sum_mask = (row_sums == 0)
    P = np.zeros_like(exp_weights)
    
    # For rows with non-zero sums, do normal normalization
    P[~zero_sum_mask.flatten(), :] = (exp_weights / row_sums)[~zero_sum_mask.flatten(), :]
    
    # For rows with zero sums, assign uniform probability to all off-diagonal elements
    for idx in range(n_cities):
        if zero_sum_mask[idx, 0]:
            P[idx, :idx] = 1.0 / (n_cities - 1)
            P[idx, idx+1:] = 1.0 / (n_cities - 1)
    
    # Compute cosine similarity matrix using efficient matrix operations
    # Dot product of rows
    dot_products = P @ P.T  # Shape: (n_cities, n_cities)
    
    # L2 norms of each row
    norms = np.linalg.norm(P, axis=1, keepdims=True)  # Shape: (n_cities, 1)
    norm_products = norms @ norms.T  # Shape: (n_cities, n_cities)
    
    # Avoid division by zero for cosine similarity
    norm_products[norm_products == 0] = 1
    topo_affinity = dot_products / norm_products
    
    # Ensure diagonal is zero for topological affinity (no self-similarity)
    np.fill_diagonal(topo_affinity, 0)
    
    # Add small value to avoid zero values that might cause issues in later computations
    topo_affinity = topo_affinity + epsilon
    
    # Combine base heuristic with context-aware weights and topological affinity
    heuristic_matrix = base_heuristic * context_weights * topo_affinity
    
    # Ensure diagonal is zero (no self-loops)
    np.fill_diagonal(heuristic_matrix, 0)
    
    return heuristic_matrix