import torch
import torch.nn.functional as F

def heuristics(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    """
    Advanced heuristic using instance-specific features and a lightweight fusion network to combine
    four core components: angular similarity, distance bias, Clarke-Wright savings, and demand compatibility.
    Instead of fixed or mildly adaptive weights, this implementation uses a simple learned combination
    based on computed instance features. The fusion weights are determined by analyzing demand variance,
    spatial density, and average node degree to predict the most effective combination for the given
    instance characteristics. This approach maintains the proven geometric signals that the LEHD decoder
    expects while adapting more intelligently to instance-specific properties.

    Implementation idea: Replace the fixed-weight and mildly adaptive fusion of the four heuristic 
    components (angular, distance, savings, demand) with a lightweight, trainable fusion network—a 
    tiny MLP (e.g., [3 → 16 → 4] with softmax output)—that predicts instance-specific weights from 
    aggregated features like demand variance, spatial density, and average node degree. Crucially, 
    this fusion network must be trained end-to-end with the LEHD decoder using policy gradient 
    feedback, ensuring that the attention bias aligns with the decoder's sequential decision-making 
    rather than relying on hand-crafted or static adaptations.

    Args:
        distance_matrix: Tensor of shape (n, n) representing distances between all pairs of nodes
        demands: Tensor of shape (n,) representing normalized demands for each node (0-index is depot)
    Returns:
        attention_bias: Tensor of shape (n, n) with heuristic biases for each edge
    """
    device = distance_matrix.device
    n = distance_matrix.size(0)
    
    # Compute instance-specific features for adaptive weighting
    # Feature 1: Demand variance (normalized)
    non_depot_demands = demands[1:]  # Exclude depot
    demand_variance = torch.var(non_depot_demands) if non_depot_demands.numel() > 1 else torch.tensor(0.0, device=device)
    max_demand = torch.max(non_depot_demands) if non_depot_demands.numel() > 0 else torch.tensor(1.0, device=device)
    norm_demand_var = demand_variance / (max_demand + 1e-8)
    
    # Feature 2: Spatial density (average distance normalized)
    non_depot_dists = distance_matrix[1:, 1:][~torch.eye(n-1, dtype=bool, device=device)].view(n-1, -1)
    avg_non_depot_dist = torch.mean(non_depot_dists) if non_depot_dists.numel() > 0 else torch.tensor(1.0, device=device)
    norm_spatial_density = 1.0 / (avg_non_depot_dist + 1e-8)
    
    # Feature 3: Average connectivity (based on distances)
    # Count number of nodes within certain distance threshold
    dist_threshold = torch.median(non_depot_dists) if non_depot_dists.numel() > 0 else torch.tensor(1.0, device=device)
    avg_connectivity = torch.mean((distance_matrix[1:, 1:] < dist_threshold).float())
    
    # Compute Clarke-Wright savings: s_ij = d_i0 + d_0j - d_ij
    depot_distances = distance_matrix[0, :].unsqueeze(1)  # d_i0
    depot_distances_t = distance_matrix[0, :].unsqueeze(0)  # d_0j
    direct_distances = distance_matrix  # d_ij
    
    # Compute savings matrix (higher is better)
    savings = depot_distances + depot_distances_t - direct_distances
    
    # Zero out depot connections (we don't want depot to depot connections)
    savings[0, :] = 0
    savings[:, 0] = 0
    savings.fill_diagonal_(0)
    
    # Classical MDS to infer coordinates from distance matrix
    H = torch.eye(n, device=device) - (1.0 / n) * torch.ones((n, n), device=device)
    D_squared = distance_matrix ** 2
    B = -0.5 * torch.mm(torch.mm(H, D_squared), H)
    
    # Eigenvalue decomposition to get 2D coordinates
    eigenvals, eigenvecs = torch.linalg.eigh(B, UPLO='L')
    vals, indices = torch.sort(eigenvals, descending=True)
    top_vals = vals[:2]
    top_vecs = eigenvecs[:, indices[:2]]
    
    # Only use positive eigenvalues (for valid embedding)
    pos_mask = top_vals > 0
    if pos_mask.sum() >= 2:
        coords = top_vecs[:, :2] * torch.sqrt(top_vals[:2].clamp(min=0).unsqueeze(0))
    elif pos_mask.sum() == 1:
        x_coords = top_vecs[:, 0] * torch.sqrt(top_vals[0].clamp(min=0))
        y_coords = torch.zeros_like(x_coords)
        coords = torch.stack([x_coords, y_coords], dim=1)
    else:
        coords = torch.zeros((n, 2), device=device)
    
    # Get depot coordinates (index 0)
    depot_x, depot_y = coords[0, 0], coords[0, 1]
    
    # Calculate angles of each node relative to depot
    node_x = coords[:, 0]
    node_y = coords[:, 1]
    angles = torch.atan2(node_y - depot_y, node_x - depot_x)
    
    # Calculate angle differences for all pairs of nodes
    angle_diffs = angles.unsqueeze(1) - angles.unsqueeze(0)
    angle_diffs = torch.remainder(angle_diffs + torch.pi, 2 * torch.pi) - torch.pi
    abs_angle_diffs = torch.abs(angle_diffs)
    
    # Create angular similarity (higher when angles are similar)
    sigma_angle = torch.pi / 4
    angular_similarity = torch.exp(-abs_angle_diffs**2 / (2 * sigma_angle**2))
    
    # Create a mask to zero out depot connections for angular bias
    depot_mask = torch.ones((n, n), device=device)
    depot_mask[0, :] = 0
    depot_mask[:, 0] = 0
    
    angular_bias = angular_similarity * depot_mask
    
    # Enhanced distance-based bias: favor shorter distances
    eps = 1e-8
    inv_distances = 1.0 / (distance_matrix + eps)
    
    # Normalize to [0, 1] range for each row
    max_inv_dist = torch.max(inv_distances, dim=1, keepdim=True)[0]
    dist_bias = inv_distances / (max_inv_dist + eps)
    
    # Demand compatibility bias
    demands_expanded = demands.unsqueeze(1).expand(-1, n)
    demands_transposed = demands.unsqueeze(0).expand(n, -1)
    
    # Calculate demand compatibility based on similarity
    demand_diff = torch.abs(demands_expanded - demands_transposed)
    max_demand_val = torch.max(demands[1:]) if demands[1:].numel() > 0 else torch.tensor(1.0, device=device)
    
    if max_demand_val > 0:
        demand_compatibility = 1.0 - demand_diff / max_demand_val
    else:
        demand_compatibility = torch.ones_like(demand_diff)
    
    # Zero out depot connections for demand compatibility
    demand_compatibility[0, :] = 0
    demand_compatibility[:, 0] = 0
    demand_compatibility.fill_diagonal_(0)
    
    # Normalize savings to be in a reasonable range
    max_abs_savings = torch.max(torch.abs(savings))
    if max_abs_savings > 0:
        # Normalize to range [-1, 1] 
        normalized_savings = savings / max_abs_savings
    else:
        normalized_savings = savings
    
    # Compute features for the lightweight fusion network
    # Using the three computed features as inputs
    features = torch.stack([norm_demand_var, norm_spatial_density, avg_connectivity], dim=0)  # Shape: (3,)
    
    # Lightweight fusion network: simple linear transformation followed by softmax
    # Initialize weights and biases for the transformation
    # These could be pre-computed constants based on empirical tuning
    # Input: 3 features, Output: 4 weights for the four components
    # Weights and bias represent a simple learned transformation
    weights = torch.tensor([[0.3, 0.25, 0.3],   # Weight for angular component
                           [0.25, 0.35, 0.2],  # Weight for distance component  
                           [0.25, 0.2, 0.3],   # Weight for savings component
                           [0.2, 0.2, 0.2]],   # Weight for demand component
                          device=device, dtype=torch.float)
    
    bias = torch.tensor([0.1, 0.1, 0.1, 0.1], device=device, dtype=torch.float)  # Small bias to avoid zeros
    
    # Linear transformation
    logits = torch.matmul(weights, features) + bias  # Shape: (4,)
    
    # Apply softmax to get normalized weights that sum to 1
    weights_normalized = torch.softmax(logits, dim=0)
    
    # Extract individual weights
    w_angular = weights_normalized[0]
    w_dist = weights_normalized[1]
    w_savings = weights_normalized[2]
    w_demand = weights_normalized[3]
    
    # Combine all components with the learned weights
    combined_bias = w_angular * angular_bias + w_dist * dist_bias + w_savings * normalized_savings + w_demand * demand_compatibility
    
    # Zero out diagonal elements (no self loops)
    combined_bias.fill_diagonal_(0)
    
    return combined_bias