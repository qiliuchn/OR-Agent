import numpy as np
from scipy.optimize import linprog

def heuristics(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Compute heuristic values for items in the multiple knapsack problem using Ant Colony Optimization.
    
    Implementation idea: Create a static proxy for dynamic construction-phase awareness by precomputing
    a composite heuristic that incorporates both global LP relaxation guidance and local efficiency 
    metrics weighted by constraint tightness information. Rather than implementing truly dynamic 
    heuristics (which would require modifying the ACO framework), we simulate construction-phase 
    awareness by computing base heuristics (LP solution and efficiency ratio) and adjusting them
    based on per-item constraint flexibility measures. This captures the essence of remaining capacity
    awareness in a static form suitable for the existing ACO implementation.
    
    Args:
        prize: Array of shape (n,) representing the prize/value of each item
        weight: Array of shape (n, m) representing the weight of each item in each constraint dimension
        
    Returns:
        Array of shape (n,) representing the heuristic desirability of each item
    """
    n, m = weight.shape
    
    # Calculate the maximum weight across all constraint dimensions for each item
    max_weight_per_item = np.max(weight, axis=1)
    
    # To avoid division by zero, replace zeros with a small positive value
    max_weight_per_item = np.where(max_weight_per_item == 0, 1e-9, max_weight_per_item)
    
    # Calculate heuristic as prize divided by max weight across constraints
    efficiency_heuristic = prize / max_weight_per_item
    
    # Normalize the efficiency heuristic to prevent numerical issues
    efficiency_heuristic_norm = efficiency_heuristic / (np.max(efficiency_heuristic) + 1e-9)
    
    # Set up the LP problem
    c = -prize  # Negate because linprog minimizes, but we want to maximize
    
    # Inequality constraints: A_ub * x <= b_ub represents weight constraints
    A_ub = weight.T  # Transpose so each row corresponds to a constraint dimension
    b_ub = np.ones(m)  # Each constraint has capacity 1
    
    # Bounds for variables: 0 <= x_i <= 1
    bounds = [(0, 1) for _ in range(n)]
    
    try:
        # Solve the LP relaxation
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            # Extract the solution
            lp_solution = result.x
            
            # Normalize LP solution to ensure it's on a comparable scale with efficiency heuristic
            lp_solution_norm = lp_solution / (np.max(lp_solution) + 1e-9) if np.max(lp_solution) > 0 else lp_solution
            
            # Calculate constraint utilization: sum of weights for each constraint dimension
            constraint_utilization = np.sum(weight, axis=0)  # Shape (m,)
            
            # Calculate constraint slack factors - higher values indicate more available capacity
            # Add a small epsilon to prevent division by zero
            constraint_slack_factor = 1.0 / (constraint_utilization + 1e-9)
            
            # Weight each item's heuristic by the average slack across dimensions it uses
            item_constraint_weights = np.sum(weight * constraint_slack_factor, axis=1)  # Shape (n,)
            # Normalize the constraint weights
            item_constraint_weights = item_constraint_weights / (np.max(item_constraint_weights) + 1e-9)
            
            # Combine LP solution, efficiency heuristic, and constraint awareness
            # Using weighted combination that maintains balance while adding constraint information
            base_combined = 0.5 * lp_solution_norm + 0.5 * efficiency_heuristic_norm
            # Use constraint weights as an additive component to maintain the balance
            combined_heuristic = base_combined + 0.1 * item_constraint_weights
            
            # Add a small constant to avoid zero values which can cause issues in ACO
            final_heuristic = combined_heuristic + 1e-9
        else:
            # If LP fails, fall back to the original efficiency-based heuristic approach
            # But still incorporate constraint awareness as additive term
            item_constraint_weights = np.sum(weight * (1.0 / (np.sum(weight, axis=0) + 1e-9)), axis=1)
            item_constraint_weights = item_constraint_weights / (np.max(item_constraint_weights) + 1e-9)
            final_heuristic = efficiency_heuristic_norm + 0.1 * item_constraint_weights + 1e-9
    except Exception:
        # If there's an error with LP solving, fall back to efficiency heuristic with constraint awareness
        item_constraint_weights = np.sum(weight * (1.0 / (np.sum(weight, axis=0) + 1e-9)), axis=1)
        item_constraint_weights = item_constraint_weights / (np.max(item_constraint_weights) + 1e-9)
        final_heuristic = efficiency_heuristic_norm + 0.1 * item_constraint_weights + 1e-9
    
    return final_heuristic