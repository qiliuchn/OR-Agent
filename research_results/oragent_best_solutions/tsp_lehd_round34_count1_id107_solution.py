import torch

def heuristics(distance_matrix: torch.Tensor) -> torch.Tensor:
    """
    Create an attention bias matrix integrating multi-scale distance attention with implicit global structure via distance quantile masking.
    
    Implementation idea: Integrate implicit global structure through distance quantile masking: Augment the multi-scale distance attention 
    with a binary mask that identifies 'globally important' edges based on distance quantiles (e.g., bottom 15% of all distances) or 
    local density outliers, then apply a learned bonus to these edges. This provides coarse global connectivity cues without explicit 
    graph construction like MST, complementing local angular penalties. The implementation combines the proven multi-scale temperature 
    framework (0.05, 0.1, 0.2 with weights 0.5, 0.3, 0.2) with the distance quantile masking approach. For each node, we identify edges 
    that fall in the bottom 15% of all distances globally and boost their importance. Additionally, we incorporate local angular penalties 
    using the law of cosines on K-nearest neighbors to ensure geometric consistency. The quantile masking provides a global connectivity 
    prior while maintaining computational efficiency.
    
    Args:
        distance_matrix: Pairwise distances between all nodes in the TSP problem
                        Shape: (n_nodes, n_nodes) where n_nodes is the number of cities
    Returns:
        attention_bias: Attention bias matrix to guide the LEHD decoder's node selection
                       Shape: (n_nodes, n_nodes)
    """
    # Create a copy to avoid modifying the original distance matrix
    dist_copy = distance_matrix.clone()
    
    # Identify diagonal/self-loop positions (where distance is 0)
    zero_mask = dist_copy == 0
    
    # Temporarily replace zero distances with a large finite value to avoid numerical issues
    large_value = 1e6
    dist_copy[zero_mask] = large_value
    
    n_nodes = dist_copy.shape[0]
    
    # If we have fewer than 3 nodes, we can't meaningfully calculate angles
    if n_nodes < 3:
        # Just return a basic distance-based heuristic
        attention_weights = torch.softmax(-dist_copy / 0.1, dim=1)
        heu = -dist_copy * attention_weights
        heu[zero_mask] = -1e9
        return heu
    
    # Stage 1: Create the base multi-scale distance attention bias
    temperatures = [0.05, 0.1, 0.2]  # Multi-scale temperature levels
    fusion_weights = [0.5, 0.3, 0.2]  # Weights for combining scales
    
    # Initialize the base heuristic matrix
    base_heu = torch.zeros_like(dist_copy)
    
    # Process each temperature level
    for temp, weight in zip(temperatures, fusion_weights):
        # Compute attention weights using softmax over negative distances (divided by temperature)
        attention_weights = torch.softmax(-dist_copy / temp, dim=1)
        
        # Create the heuristic bias matrix for this temperature level
        current_heu = -dist_copy * attention_weights
        
        # Weight this temperature level's contribution and add to the overall heuristic
        base_heu = base_heu + weight * current_heu
    
    # Stage 2: Add global structure via distance quantile masking
    # Flatten the distance matrix to find quantiles (excluding diagonal)
    flat_distances = dist_copy[~zero_mask]
    
    # Calculate the 10th percentile threshold for "globally important" edges
    quantile_threshold = torch.quantile(flat_distances, 0.10)
    
    # Create a binary mask for edges in the bottom 10% of distances
    global_structure_mask = (dist_copy <= quantile_threshold).float()
    
    # Apply a bonus to globally important edges
    global_bonus = global_structure_mask * 3.0  # Increased boost factor for globally important edges
    
    # Stage 3: Refine with geometric consistency via angular penalties
    
    # Get K nearest neighbors for each node to build a sparse graph
    K = min(10, n_nodes - 1)  # Use up to 10 nearest neighbors
    
    # Get K nearest neighbors for each node (excluding self)
    _, nearest_indices = torch.topk(dist_copy, K + 1, largest=False, dim=1)  # Include self
    nearest_indices = nearest_indices[:, 1:]  # Exclude self (first column after sorting)
    
    # Calculate initial angular penalties for nearby triplets
    angular_penalties = torch.zeros_like(dist_copy)
    
    # Compute angular penalties using a vectorized approach where possible
    for i in range(n_nodes):
        neighbors_i = nearest_indices[i]  # Nearest neighbors of node i
        
        for j in neighbors_i:
            j = int(j.item())
            # Get neighbors of j that are also in the neighborhood of i or are nearby
            neighbors_j = nearest_indices[j]
            
            # Calculate angles for triplets involving i, j, and neighbors of j
            for k in neighbors_j:
                k = int(k.item())
                if i != j and j != k and i != k:
                    # Get the three distances forming the triangle i-j-k
                    d_ij = dist_copy[i, j]
                    d_jk = dist_copy[j, k]
                    d_ik = dist_copy[i, k]
                    
                    # Calculate the angle at j using law of cosines
                    # cos(angle) = (d_ij^2 + d_jk^2 - d_ik^2) / (2 * d_ij * d_jk)
                    denominator = 2 * d_ij * d_jk
                    if denominator > 1e-9:  # Avoid division by zero
                        cos_angle = (d_ij**2 + d_jk**2 - d_ik**2) / denominator
                        # Clamp cosine to valid range [-1, 1] to avoid numerical errors
                        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                        
                        # Convert cosine to angle in radians
                        angle = torch.acos(cos_angle)
                        
                        # Penalize sharp angles (close to 0 or π)
                        # Use sin of angle as penalty (high for 0 and π, low for π/2)
                        angle_penalty = torch.abs(torch.sin(angle))
                        
                        # Add this penalty to the connection j->k
                        angular_penalties[j, k] += angle_penalty
    
    # Normalize angular penalties
    max_penalty = torch.max(angular_penalties)
    if max_penalty > 0:
        angular_penalties = angular_penalties / max_penalty * 2.0  # Scale factor
    
    # Stage 4: Combine all components
    heu = base_heu + global_bonus - angular_penalties * 0.5  # Scale factor for angular penalties
    
    # For very close neighbors, add a bonus term based on the attention weight itself
    max_dist = torch.max(dist_copy, dim=1, keepdim=True)[0]
    close_neighbor_threshold = max_dist * 0.05  # Consider nodes within 5% of max distance as "close"
    close_mask = dist_copy <= close_neighbor_threshold
    
    # Add bonus for close neighbors weighted by attention
    attention_weights_base = torch.softmax(-dist_copy / 0.05, dim=1)  # Use lower temperature for sharper attention
    heu = heu + (attention_weights_base * close_mask.float()) * 6.0  # Increase bonus for very close neighbors
    
    # Set the diagonal entries to a large negative value to prevent selection
    heu[zero_mask] = -1e9
    
    # Ensure no inf or nan values exist in the result
    heu = torch.clamp(heu, min=-1e9, max=1e9)
    heu = torch.nan_to_num(heu, nan=0.0, posinf=-1e9, neginf=-1e9)
    
    return heu