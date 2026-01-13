import torch

def heuristics(distance_matrix: torch.Tensor) -> torch.Tensor:
    """
    heu_ij = - log(dis_ij) if j is the top-K nearest neighbor of i, else - dis_ij
    """
    K = 100
    
    distance_matrix[distance_matrix == 0] = 1e5
    
    # Compute top-k nearest neighbors (smallest distances)
    values, indices = torch.topk(distance_matrix, k=K, largest=False, dim=1)
    heu = -distance_matrix.clone()
    # Create a mask where topk indices are True and others are False
    topk_mask = torch.zeros_like(distance_matrix, dtype=torch.bool)
    topk_mask.scatter_(1, indices, True)
    # Apply -log(d_ij) only to the top-k elements
    heu[topk_mask] = -torch.log(distance_matrix[topk_mask])
    return heu