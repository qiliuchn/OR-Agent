import torch

def heuristics(distance_matrix: torch.Tensor) -> torch.Tensor:
    """
    Create heuristic attention bias.

    Formula: heu_ij = - log(dis_ij) if j is the top-K nearest neighbor of i, else - dis_ij

    This creates a bias that encourages the model to attend to nearby nodes,
    mimicking human intuition in TSP solving.
    """
    # Avoid log(0) by replacing zeros with large number
    distance_matrix[distance_matrix == 0] = 1e5
    K = 100  # Number of nearest neighbors to consider

    # Compute top-k nearest neighbors (smallest distances)
    values, indices = torch.topk(distance_matrix, k=K, largest=False, dim=1)

    # Start with negative distances (discourages distant nodes)
    heu = -distance_matrix.clone()

    # Create a mask where topk indices are True and others are False
    topk_mask = torch.zeros_like(distance_matrix, dtype=torch.bool)
    topk_mask.scatter_(1, indices, True)

    # Apply -log(d_ij) only to the top-k elements (encourages nearby nodes)
    # log makes nearby nodes have stronger positive bias
    heu[topk_mask] = -torch.log(distance_matrix[topk_mask])
    
    return heu