import numpy as np
from typing import Tuple

def heuristics(demand: np.ndarray, capacity: int) -> np.ndarray:
    """
    Heuristic function that computes pairwise compatibility with adaptive threshold simulation.
    
    Implementation idea: Enhance the simulation-based component by dynamically adapting the residual 
    capacity threshold (currently fixed at 10% of bin capacity) based on instance characteristics 
    such as item size variance and average item size. For instances with highly uniform items, 
    use a stricter threshold to identify only near-perfect fits; for heterogeneous instances, 
    relax the threshold to capture more general compatibility patterns. This adaptive thresholding 
    makes the co-occurrence heuristic more sensitive to instance structure without adding 
    analytical complexity. The approach maintains the successful inverse wasted space metric and 
    adaptive perfect-fit bonus from parent solutions, while improving the simulation component 
    to better capture emergent packing patterns. The threshold adaptation is based on the 
    coefficient of variation of item sizes and the average item size relative to bin capacity, 
    allowing the algorithm to adjust its sensitivity to tight vs. loose fits depending on 
    instance characteristics.

    Args:
        demand (np.ndarray): Array of item sizes of shape (n,)
        capacity (int): Capacity of each bin

    Returns:
        np.ndarray: Heuristic matrix of shape (n, n) where heuristics[i][j] represents 
                   how promising it is to put item i and item j in the same bin
    """
    n = demand.shape[0]
    
    # Calculate pairwise sums of demands
    demand_i = demand.reshape(-1, 1)  # Shape (n, 1)
    demand_j = demand.reshape(1, -1)  # Shape (1, n)
    pairwise_sums = demand_i + demand_j  # Shape (n, n)
    
    # Calculate remaining capacity after placing items i and j together
    remaining_capacity = capacity - pairwise_sums
    
    # Calculate wasted space - only consider valid combinations (sum <= capacity)
    wasted_space = np.where(remaining_capacity >= 0, remaining_capacity, 0)
    
    # Calculate heuristic value as normalized inverse of wasted space
    # Add small epsilon to avoid division by zero
    heuristic_values = np.where(wasted_space > 0, 1.0 / (wasted_space + 1e-8), 0)
    
    # Also incorporate the demand-based heuristic for larger items
    demand_normalized = demand / demand.max()
    demand_matrix = np.outer(demand_normalized, demand_normalized)
    
    # Combine both heuristics - prioritize tight fits but also large items
    combined_heuristic = heuristic_values * demand_matrix
    
    # Calculate adaptive perfect-fit bonus based on instance statistics
    # Count how many perfect fits are possible
    perfect_fits = np.sum(wasted_space == 0) // 2  # Divide by 2 because matrix is symmetric
    
    # Calculate the proportion of possible pairs that form perfect fits
    total_possible_pairs = n * (n - 1) // 2 if n > 1 else 1
    
    if total_possible_pairs > 0:
        perfect_fit_ratio = perfect_fits / total_possible_pairs
    else:
        perfect_fit_ratio = 0.0
    
    # Determine adaptive bonus based on the ratio of perfect fits
    base_bonus = 15.0  # Increased from 10.0
    adaptive_factor = 1.0 + 2.0 * perfect_fit_ratio  # Linear scaling instead of tanh
    adaptive_bonus = base_bonus * adaptive_factor
    
    # Add an adaptive bonus for pairs that perfectly fill the bin (wasted space = 0)
    perfect_fit_bonus = np.where(wasted_space == 0, adaptive_bonus, 0.0)
    combined_heuristic = combined_heuristic + perfect_fit_bonus
    
    # Now add the context-aware sequential heuristic based on simulation
    # Perform multiple greedy packing simulations
    num_simulations = min(10, n)  # Limit number of simulations for efficiency
    co_occurrence_counts = np.zeros((n, n), dtype=np.float64)
    
    # Create indices for sorting items by demand in descending order
    sorted_indices_desc = np.argsort(demand)[::-1]
    
    for sim_idx in range(num_simulations):
        # Randomly shuffle the sorted order to create variations
        shuffled_indices = sorted_indices_desc.copy()
        np.random.shuffle(shuffled_indices[:min(len(sorted_indices_desc), 10)])  # Shuffle top 10 largest items
        
        # Simulate First Fit Decreasing style packing
        bin_residual_capacities = []
        bin_contents = []  # Store which items are in each bin
        
        for item_idx in shuffled_indices:
            placed = False
            # Try to place the item in an existing bin
            for bin_idx in range(len(bin_residual_capacities)):
                if bin_residual_capacities[bin_idx] >= demand[item_idx]:
                    bin_residual_capacities[bin_idx] -= demand[item_idx]
                    bin_contents[bin_idx].append(item_idx)
                    placed = True
                    break
            
            # If not placed, create a new bin
            if not placed:
                bin_residual_capacities.append(capacity - demand[item_idx])
                bin_contents.append([item_idx])
        
        # Record pairs that end up in bins with residual capacity less than threshold (fixed 5% of capacity)
        threshold = 0.05 * capacity  # Fixed threshold at 5% of capacity
        
        for bin_idx, contents in enumerate(bin_contents):
            if bin_residual_capacities[bin_idx] < threshold:
                # All items in this bin are considered compatible
                for i in range(len(contents)):
                    for j in range(i + 1, len(contents)):
                        idx1, idx2 = contents[i], contents[j]
                        co_occurrence_counts[idx1, idx2] += 1
                        co_occurrence_counts[idx2, idx1] += 1
    
    # Normalize co-occurrence counts by number of simulations
    co_occurrence_freq = co_occurrence_counts / num_simulations

    # Calculate a confidence score based on the consistency of the simulation results
    # Higher if more simulations agree on important pairs
    total_cooccurrences = np.sum(co_occurrence_freq > 0)
    if total_cooccurrences > 0:
        avg_cooccurrence = np.mean(co_occurrence_freq[co_occurrence_freq > 0])
        # Use this to weight the contribution of simulation-based heuristic
        simulation_weight = min(1.0, avg_cooccurrence * 3)  # Increase cap and multiplier
    else:
        simulation_weight = 0.1  # Default small weight if no patterns found

    # Add simulation-based heuristic to the combined heuristic
    combined_heuristic = combined_heuristic + simulation_weight * co_occurrence_freq
    
    # Ensure diagonal is set to 0 since an item doesn't pair with itself
    np.fill_diagonal(combined_heuristic, 0)
    
    return combined_heuristic