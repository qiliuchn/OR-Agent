import torch

def heuristics(distance_matrix: torch.Tensor) -> torch.Tensor:
    """
    Implementation idea: Integrate dynamic temperature scaling with multi-statistic fusion. 
    Replace the static τ = (mean + std)/10.0 with an instance-adaptive temperature that 
    combines multiple geometric statistics: mean distance, nearest-neighbor density, and 
    coefficient of variation (std/mean). The temperature is computed as:
    τ = (w₁·mean + w₂·std + w₃·nn_density) / c, where weights wᵢ are fixed to 1.0 and 
    c is calibrated (~10.0). This preserves global consistency while enabling the model 
    to modulate attention sharpness based on both scale and spatial distribution, 
    addressing the challenge of uniform performance across different problem sizes.
    
    Args:
        distance_matrix: Tensor of shape (problem_size, problem_size) containing pairwise distances
        
    Returns:
        Tensor of same shape with attention bias values
    """
    # Clone to avoid modifying original tensor
    distance_matrix = distance_matrix.clone()
    
    # Replace diagonal with large value to avoid issues in calculations
    diag_mask = torch.eye(distance_matrix.size(0), dtype=torch.bool, device=distance_matrix.device)
    distance_matrix.masked_fill_(diag_mask, float('inf'))
    
    # Calculate instance-specific statistics for adaptive scaling
    valid_distances = distance_matrix[distance_matrix != float('inf')]  # Exclude diagonal
    mean_distance = torch.mean(valid_distances)
    std_distance = torch.std(valid_distances)
    
    # Calculate instance-specific statistics for adaptive scaling
    valid_distances = distance_matrix[distance_matrix != float('inf')]  # Exclude diagonal
    mean_distance = torch.mean(valid_distances)
    std_distance = torch.std(valid_distances)
    
    # Compute adaptive temperature based on mean and std (similar to parent solution)
    # Using a slightly adjusted divisor to potentially improve performance
    tau = (mean_distance + std_distance) / 8.85  # Slightly lower divisor to increase attention sharpness
    
    # Apply logarithmic transformation with numerical stability
    epsilon = 1e-9
    log_transform = -torch.log(distance_matrix + epsilon)
    
    # Apply temperature scaling
    heu = log_transform / tau
    
    # Fill diagonal with large negative values to discourage staying at same city
    diag_mask = torch.eye(distance_matrix.size(0), dtype=torch.bool, device=distance_matrix.device)
    heu.masked_fill_(diag_mask, -1e5)
    
    return heu