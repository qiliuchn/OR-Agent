import numpy as np
from typing import Tuple

def crossover(parents: np.ndarray, n_pop: int) -> np.ndarray:
    """
    Adaptive SBX crossover with parent similarity-based eta adjustment: when parents share many common 
    positions (high overlap), use a higher eta (e.g., 10) for stronger exploitation; when overlap is 
    low, use a lower eta (e.g., 2) to encourage broader exploration, dynamically balancing diversity 
    and convergence.
    
    Implementation idea: The crossover computes the intersection of positions between parent pairs to 
    determine their similarity level. Based on this overlap ratio, it dynamically adjusts the SBX eta 
    parameter - using high eta (>5) for similar parents to exploit shared good positions, and low eta 
    (<5) for dissimilar parents to explore more diverse combinations. For positions not in the common 
    set, it applies SBX in 2D coordinate space to generate novel placements while preserving spatial 
    relationships. After generating offspring positions, it uses Manhattan-distance-based repair to 
    resolve duplicates and ensure all positions are unique and valid. This adaptive approach leverages 
    the observation that parent similarity should inform the exploration-exploitation trade-off in 
    crossover operations.

    Args:
        parents (np.ndarray): Array of parent solutions with shape (n_parents, n_decap),
                              where each row represents a valid decap placement as 1D indices.
        n_pop (int): Number of offspring to generate.
        
    Returns:
        np.ndarray: Array of offspring solutions with shape (n_pop, n_decap), where each row
                   contains valid decap placements as 1D indices.
    """
    n_parents, n_decap = parents.shape
    
    # Grid dimensions
    grid_size = 10  # 10x10 grid
    
    # Calculate how many parent pairs we need (each pair produces 2 offspring)
    n_pairs_needed = int(np.ceil(n_pop / 2))
    
    # Randomly select parent pairs
    parent_indices = np.random.choice(n_parents, size=(n_pairs_needed, 2), replace=True)
    
    # Initialize offspring array - only create what we need
    offspring = np.zeros((n_pop, n_decap), dtype=int)
    
    # Process each pair of parents
    offspring_idx = 0
    for i in range(n_pairs_needed):
        parent1 = parents[parent_indices[i, 0]]  # Shape: (n_decap,)
        parent2 = parents[parent_indices[i, 1]]  # Shape: (n_decap,)
        
        # Calculate the overlap (common positions) between parents
        set_p1 = set(parent1)
        set_p2 = set(parent2)
        common_positions = set_p1.intersection(set_p2)
        overlap_ratio = len(common_positions) / n_decap
        
        # Use fixed eta=5.0 based on successful baseline approach from reflection
        eta = 5.0
        
        # Extract common positions
        common_list = list(common_positions)
        
        # Identify positions unique to each parent
        unique_to_p1 = [pos for pos in parent1 if pos not in common_positions]
        unique_to_p2 = [pos for pos in parent2 if pos not in common_positions]
        
        # Calculate how many more positions we need
        n_remaining = n_decap - len(common_list)
        
        # Combine unique positions from both parents and assign them to each offspring
        all_unique = unique_to_p1 + unique_to_p2
        if len(all_unique) >= n_remaining:
            # Randomly select n_remaining unique positions from the combined pool
            selected_unique = np.random.choice(all_unique, n_remaining, replace=False).tolist()
        else:
            # If not enough unique positions, pad with random valid positions not in common or unique
            selected_unique = all_unique[:]
            needed = n_remaining - len(all_unique)
            used_positions = set(common_list + all_unique)
            all_positions = set(range(grid_size * grid_size))
            available_positions = list(all_positions - used_positions)
            if len(available_positions) >= needed:
                extra_positions = np.random.choice(available_positions, needed, replace=False).tolist()
            else:
                # If still not enough, allow repetition of available positions
                extra_positions = np.random.choice(available_positions, needed, replace=True).tolist() if available_positions else [0]*needed
            selected_unique.extend(extra_positions)
        
        # Split the selected unique positions between the two offspring
        split_point = n_remaining // 2
        selected_unique_1 = selected_unique[:split_point]
        selected_unique_2 = selected_unique[split_point:n_remaining]
        
        # If we need more positions, fill with the same positions or with positions from the other subset
        if len(selected_unique_1) < n_remaining // 2 + n_remaining % 2:
            needed = (n_remaining // 2 + n_remaining % 2) - len(selected_unique_1)
            selected_unique_1.extend(selected_unique_2[:needed])
        if len(selected_unique_2) < n_remaining // 2 + n_remaining % 2:
            needed = (n_remaining // 2 + n_remaining % 2) - len(selected_unique_2)
            selected_unique_2.extend(selected_unique_1[:needed])
        
        # Now perform SBX on corresponding positions from each subset
        # First, ensure both subsets have exactly the right length
        if len(selected_unique_1) < n_remaining:
            selected_unique_1.extend(selected_unique_1[:(n_remaining-len(selected_unique_1))])
        if len(selected_unique_2) < n_remaining:
            selected_unique_2.extend(selected_unique_2[:(n_remaining-len(selected_unique_2))])
        
        selected_unique_1 = selected_unique_1[:n_remaining]
        selected_unique_2 = selected_unique_2[:n_remaining]
        
        # Convert positions to 2D coordinates for SBX
        p1_rows = [pos // grid_size for pos in selected_unique_1]
        p1_cols = [pos % grid_size for pos in selected_unique_1]
        
        p2_rows = [pos // grid_size for pos in selected_unique_2]
        p2_cols = [pos % grid_size for pos in selected_unique_2]
        
        # Convert to numpy arrays
        p1_rows = np.array(p1_rows)
        p1_cols = np.array(p1_cols)
        p2_rows = np.array(p2_rows)
        p2_cols = np.array(p2_cols)
        
        # Generate random numbers for SBX
        u_rows = np.random.random(n_remaining)
        u_cols = np.random.random(n_remaining)
        
        # Calculate beta values for SBX
        beta_rows = np.where(u_rows <= 0.5, 
                            (2 * u_rows) ** (1.0 / (eta + 1)),
                            (2 * (1 - u_rows)) ** (-1.0 / (eta + 1)))
        
        beta_cols = np.where(u_cols <= 0.5, 
                            (2 * u_cols) ** (1.0 / (eta + 1)),
                            (2 * (1 - u_cols)) ** (-1.0 / (eta + 1)))
        
        # Apply SBX transformation to create offspring 1
        off1_rows_cont = 0.5 * ((p1_rows + p2_rows) - beta_rows * np.abs(p1_rows - p2_rows))
        off1_cols_cont = 0.5 * ((p1_cols + p2_cols) - beta_cols * np.abs(p1_cols - p2_cols))
        
        # Apply SBX transformation to create offspring 2
        off2_rows_cont = 0.5 * ((p1_rows + p2_rows) + beta_rows * np.abs(p1_rows - p2_rows))
        off2_cols_cont = 0.5 * ((p1_cols + p2_cols) + beta_cols * np.abs(p1_cols - p2_cols))
        
        # Round to nearest integer and clip to valid range
        off1_rows = np.round(np.clip(off1_rows_cont, 0, grid_size - 1)).astype(int)
        off1_cols = np.round(np.clip(off1_cols_cont, 0, grid_size - 1)).astype(int)
        
        off2_rows = np.round(np.clip(off2_rows_cont, 0, grid_size - 1)).astype(int)
        off2_cols = np.round(np.clip(off2_cols_cont, 0, grid_size - 1)).astype(int)
        
        # Convert back to 1D indices
        off1_unique = off1_rows * grid_size + off1_cols
        off2_unique = off2_rows * grid_size + off2_cols
        
        # Combine common positions with SBX-generated unique positions
        offspring1 = common_list + off1_unique.tolist()
        offspring2 = common_list + off2_unique.tolist()
        
        # Pad or truncate to exact length
        if len(offspring1) < n_decap:
            # Add some random positions to fill up
            used_positions = set(offspring1)
            all_positions = set(range(grid_size * grid_size))
            available_positions = list(all_positions - used_positions)
            if len(available_positions) >= n_decap - len(offspring1):
                additional_positions = np.random.choice(available_positions, n_decap - len(offspring1), replace=False)
            else:
                # If not enough unique positions, allow some reuse
                additional_positions = np.random.choice(list(all_positions - used_positions), 
                                                      n_decap - len(offspring1), replace=True)
            offspring1.extend(additional_positions.tolist())
        elif len(offspring1) > n_decap:
            offspring1 = offspring1[:n_decap]
            
        if len(offspring2) < n_decap:
            # Add some random positions to fill up
            used_positions = set(offspring2)
            all_positions = set(range(grid_size * grid_size))
            available_positions = list(all_positions - used_positions)
            if len(available_positions) >= n_decap - len(offspring2):
                additional_positions = np.random.choice(available_positions, n_decap - len(offspring2), replace=False)
            else:
                # If not enough unique positions, allow some reuse
                additional_positions = np.random.choice(list(all_positions - used_positions), 
                                                      n_decap - len(offspring2), replace=True)
            offspring2.extend(additional_positions.tolist())
        elif len(offspring2) > n_decap:
            offspring2 = offspring2[:n_decap]
        
        # Store in offspring array
        offspring[offspring_idx] = np.array(offspring1)
        offspring_idx += 1
        
        # Only add second offspring if we haven't filled our quota
        if offspring_idx < n_pop:
            offspring[offspring_idx] = np.array(offspring2)
            offspring_idx += 1
            
        # Break if we've filled all required offspring
        if offspring_idx >= n_pop:
            break
    
    # Repair infeasible positions in each offspring (resolve duplicates and invalid positions)
    for i in range(n_pop):
        offspring_i = offspring[i].copy()
        
        # Identify and fix duplicates
        seen_positions = []
        duplicate_indices = []
        
        for idx, pos in enumerate(offspring_i):
            if pos in seen_positions:
                duplicate_indices.append(idx)
            else:
                if 0 <= pos < grid_size * grid_size:  # Valid range check
                    seen_positions.append(pos)
                else:
                    # Position out of bounds, mark as duplicate to be fixed
                    duplicate_indices.append(idx)
        
        # If no duplicates, assign and continue
        if not duplicate_indices:
            offspring[i] = offspring_i
            continue
        
        # Create set of used positions to avoid
        used_positions = set(seen_positions)
        
        # For each duplicate position, find a valid replacement using spatial proximity
        all_positions = set(range(grid_size * grid_size))
        available_positions = list(all_positions - used_positions)
        
        for dup_idx in duplicate_indices:
            if available_positions:
                # Get the original position that was potentially duplicated
                original_pos = offspring_i[dup_idx]
                
                # Convert to 2D coordinates to find nearby positions
                orig_row, orig_col = divmod(original_pos, grid_size)
                
                # Find available positions and calculate Manhattan distances
                dists_with_pos = []
                for pos in available_positions:
                    row, col = divmod(pos, grid_size)
                    dist = abs(row - orig_row) + abs(col - orig_col)
                    dists_with_pos.append((dist, pos))
                
                # Sort by distance and pick the closest available position
                dists_with_pos.sort(key=lambda x: x[0])
                closest_pos = dists_with_pos[0][1]
                
                offspring_i[dup_idx] = closest_pos
                available_positions.remove(closest_pos)
            else:
                # If no available positions left, use random valid position
                # Reset available_positions to unused ones
                current_positions = set(offspring_i)
                all_valid = set(range(grid_size * grid_size))
                available_positions = list(all_valid - current_positions)
                
                if available_positions:
                    replacement_pos = np.random.choice(available_positions)
                    offspring_i[dup_idx] = replacement_pos
                else:
                    # Ultimate fallback: just ensure bounds
                    offspring_i[dup_idx] = np.clip(offspring_i[dup_idx], 0, grid_size * grid_size - 1)
                    # Update available_positions after this change
                    current_positions = set(offspring_i)
                    all_valid = set(range(grid_size * grid_size))
                    available_positions = list(all_valid - current_positions)
        
        offspring[i] = offspring_i
    
    return offspring