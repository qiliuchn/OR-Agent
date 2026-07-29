import os
import sys
import time
import random
import math
import json
import numpy as np
import pandas as pd
import scipy
import traci
import torch

import numpy as np
from itertools import permutations

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    """
    Select the next node to visit in a Traveling Salesman Problem (TSP) constructive heuristic.
    
    This implementation uses a greedy nearest neighbor approach with strategic lookahead
    and considers multiple factors for better decision making, including advanced tie-breaking
    and dynamic lookahead based on remaining nodes. Enhanced with additional heuristics
    for better performance.
    
    Args:
        current_node (int): The node currently being visited
        destination_node (int): The final destination node in the TSP tour
        unvisited_nodes (set): Set of nodes that haven't been visited yet
        distance_matrix (np.ndarray): NxN matrix where distance_matrix[i][j] is the distance from node i to j
    Returns:
        int: The selected next node to visit
    """
    if len(unvisited_nodes) == 1:
        # If it's the last node, just return it
        return unvisited_nodes.pop()

    # For the second-to-last node, consider the complete remaining path
    if len(unvisited_nodes) == 2:
        nodes_list = list(unvisited_nodes)
        option1_first = nodes_list[0]
        option1_second = nodes_list[1]
        option2_first = nodes_list[1] 
        option2_second = nodes_list[0]
        
        # Calculate total cost for each sequence: current->first->second->destination
        cost1 = (distance_matrix[current_node][option1_first] + 
                 distance_matrix[option1_first][option1_second] + 
                 distance_matrix[option1_second][destination_node])
        cost2 = (distance_matrix[current_node][option2_first] + 
                 distance_matrix[option2_first][option2_second] + 
                 distance_matrix[option2_second][destination_node])
        
        if cost1 < cost2:
            return option1_first
        else:
            return option2_first

    # For 3 nodes remaining, evaluate all possible paths completely
    if len(unvisited_nodes) == 3:
        nodes_list = list(unvisited_nodes)
        best_sequence = None
        best_cost = float('inf')
        
        # Try all possible orderings of the 3 unvisited nodes
        for perm in permutations(nodes_list):
            cost = (distance_matrix[current_node][perm[0]] + 
                    distance_matrix[perm[0]][perm[1]] + 
                    distance_matrix[perm[1]][perm[2]] + 
                    distance_matrix[perm[2]][destination_node])
            
            if cost < best_cost:
                best_cost = cost
                best_sequence = perm
                
        return best_sequence[0]

    # For 4-8 nodes remaining, use deeper lookahead with pruning
    if 4 <= len(unvisited_nodes) <= 8:
        best_node = None
        best_cost = float('inf')
        
        # Try each possible next node
        for candidate in unvisited_nodes:
            remaining_after_candidate = unvisited_nodes - {candidate}
            
            # Use recursive lookahead to estimate best path from here
            def calculate_path_cost(start_node, remaining_set, depth=0, max_depth=2):
                if not remaining_set:
                    return distance_matrix[start_node][destination_node]
                
                # Limit recursion depth to manage complexity
                if depth >= max_depth:
                    # Greedy approximation when we reach max depth
                    total_cost = 0
                    current_pos = start_node
                    temp_remaining = remaining_set.copy()
                    
                    while temp_remaining:
                        next_node = min(temp_remaining, key=lambda x: distance_matrix[current_pos][x])
                        total_cost += distance_matrix[current_pos][next_node]
                        temp_remaining.remove(next_node)
                        current_pos = next_node
                    
                    total_cost += distance_matrix[current_pos][destination_node]
                    return total_cost
                
                min_cost = float('inf')
                # Only consider top few candidates to limit branching
                candidates_for_next = sorted(remaining_set, key=lambda x: distance_matrix[start_node][x])[:max(3, len(remaining_set)//2)]
                
                for next_node in candidates_for_next:
                    new_remaining = remaining_set - {next_node}
                    cost = distance_matrix[start_node][next_node] + \
                           calculate_path_cost(next_node, new_remaining, depth + 1, max_depth)
                    min_cost = min(min_cost, cost)
                
                return min_cost
            
            # Calculate the estimated cost for choosing this candidate
            cost = distance_matrix[current_node][candidate] + \
                   calculate_path_cost(candidate, remaining_after_candidate, 0, 2 if len(unvisited_nodes) > 6 else 3)
            
            if cost < best_cost:
                best_cost = cost
                best_node = candidate
        
        return best_node

    # For 9-15 nodes remaining, use medium lookahead with more sophisticated heuristics
    if 9 <= len(unvisited_nodes) <= 15:
        best_node = None
        best_cost = float('inf')
        
        # Try each possible next node
        for candidate in unvisited_nodes:
            remaining_after_candidate = unvisited_nodes - {candidate}
            
            # Look ahead 2 steps and use greedy heuristic for the rest
            def calculate_path_cost_with_greedy(start_node, remaining_set):
                total_cost = 0
                current_pos = start_node
                temp_remaining = remaining_set.copy()
                
                # Look ahead 2 steps depending on size
                if len(temp_remaining) >= 3:
                    # For first step, try each possibility and then apply greedy
                    min_sub_cost = float('inf')
                    for next_node in temp_remaining:
                        new_remaining = temp_remaining - {next_node}
                        sub_cost = distance_matrix[current_pos][next_node]
                        
                        # Look ahead one more step
                        if len(new_remaining) >= 1:
                            for next_next_node in new_remaining:
                                newer_remaining = new_remaining - {next_next_node}
                                sub_cost_temp = sub_cost + distance_matrix[next_node][next_next_node]
                                
                                # Apply greedy to the rest
                                temp_pos = next_next_node
                                temp_rem = newer_remaining.copy()
                                while temp_rem:
                                    next_step = min(temp_rem, key=lambda x: distance_matrix[temp_pos][x])
                                    sub_cost_temp += distance_matrix[temp_pos][next_step]
                                    temp_rem.remove(next_step)
                                    temp_pos = next_step
                                
                                sub_cost_temp += distance_matrix[temp_pos][destination_node]
                                
                                if sub_cost_temp < min_sub_cost:
                                    min_sub_cost = sub_cost_temp
                        else:
                            # Apply greedy to the rest
                            temp_pos = next_node
                            temp_rem = new_remaining.copy()
                            while temp_rem:
                                next_step = min(temp_rem, key=lambda x: distance_matrix[temp_pos][x])
                                sub_cost += distance_matrix[temp_pos][next_step]
                                temp_rem.remove(next_step)
                                temp_pos = next_step
                            
                            sub_cost += distance_matrix[temp_pos][destination_node]
                            
                            if sub_cost < min_sub_cost:
                                min_sub_cost = sub_cost
                    
                    return min_sub_cost
                else:
                    # For smaller sets, just apply greedy
                    while temp_remaining:
                        next_node = min(temp_remaining, key=lambda x: distance_matrix[current_pos][x])
                        total_cost += distance_matrix[current_pos][next_node]
                        temp_remaining.remove(next_node)
                        current_pos = next_node
                    
                    total_cost += distance_matrix[current_pos][destination_node]
                    return total_cost
            
            cost = distance_matrix[current_node][candidate] + calculate_path_cost_with_greedy(candidate, remaining_after_candidate)
            
            if cost < best_cost:
                best_cost = cost
                best_node = candidate
        
        return best_node

    # For 16-25 nodes remaining, use more detailed lookahead
    if 16 <= len(unvisited_nodes) <= 25:
        best_node = None
        best_cost = float('inf')
        
        # Focus on top candidates to reduce computation
        candidates = sorted(unvisited_nodes, key=lambda x: distance_matrix[current_node][x])[:max(3, len(unvisited_nodes)//3)]
        
        for candidate in candidates:
            remaining_after_candidate = unvisited_nodes - {candidate}
            
            # Look ahead 2 steps with greedy continuation (increased lookahead from previous)
            min_cost = float('inf')
            for next_candidate in remaining_after_candidate:
                new_remaining = remaining_after_candidate - {next_candidate}
                for next_next_candidate in new_remaining:
                    cost = (distance_matrix[current_node][candidate] + 
                            distance_matrix[candidate][next_candidate] +
                            distance_matrix[next_candidate][next_next_candidate])
                    
                    # Estimate the rest using greedy
                    temp_remaining = new_remaining - {next_next_candidate}
                    temp_pos = next_next_candidate
                    
                    while temp_remaining:
                        next_step = min(temp_remaining, key=lambda x: distance_matrix[temp_pos][x])
                        cost += distance_matrix[temp_pos][next_step]
                        temp_remaining.remove(next_step)
                        temp_pos = next_step
                    
                    cost += distance_matrix[temp_pos][destination_node]
                    
                    if cost < min_cost:
                        min_cost = cost
            
            if min_cost < best_cost:
                best_cost = min_cost
                best_node = candidate
        
        return best_node
    
    # For 26-35 nodes remaining, use even more focused lookahead
    if 26 <= len(unvisited_nodes) <= 35:
        best_node = None
        best_cost = float('inf')
        
        # Focus on top 2 candidates to reduce computation
        candidates = sorted(unvisited_nodes, key=lambda x: distance_matrix[current_node][x])[:2]
        
        for candidate in candidates:
            remaining_after_candidate = unvisited_nodes - {candidate}
            
            # Look ahead 1 step with greedy continuation
            min_cost = float('inf')
            for next_candidate in remaining_after_candidate:
                cost = (distance_matrix[current_node][candidate] + 
                        distance_matrix[candidate][next_candidate])
                
                # Estimate the rest using greedy
                temp_remaining = remaining_after_candidate - {next_candidate}
                temp_pos = next_candidate
                
                while temp_remaining:
                    next_step = min(temp_remaining, key=lambda x: distance_matrix[temp_pos][x])
                    cost += distance_matrix[temp_pos][next_step]
                    temp_remaining.remove(next_step)
                    temp_pos = next_step
                
                cost += distance_matrix[temp_pos][destination_node]
                
                if cost < min_cost:
                    min_cost = cost
            
            if min_cost < best_cost:
                best_cost = min_cost
                best_node = candidate
        
        return best_node

    # For 36-45 nodes remaining, use even more focused lookahead
    if 36 <= len(unvisited_nodes) <= 45:
        best_node = None
        best_cost = float('inf')
        
        # Focus on top 2 candidates to reduce computation
        candidates = sorted(unvisited_nodes, key=lambda x: distance_matrix[current_node][x])[:2]
        
        for candidate in candidates:
            remaining_after_candidate = unvisited_nodes - {candidate}
            
            # Look ahead 1 step with greedy continuation
            min_cost = float('inf')
            for next_candidate in remaining_after_candidate:
                cost = (distance_matrix[current_node][candidate] + 
                        distance_matrix[candidate][next_candidate])
                
                # Estimate the rest using greedy
                temp_remaining = remaining_after_candidate - {next_candidate}
                temp_pos = next_candidate
                
                while temp_remaining:
                    next_step = min(temp_remaining, key=lambda x: distance_matrix[temp_pos][x])
                    cost += distance_matrix[temp_pos][next_step]
                    temp_remaining.remove(next_step)
                    temp_pos = next_step
                
                cost += distance_matrix[temp_pos][destination_node]
                
                if cost < min_cost:
                    min_cost = cost
            
            if min_cost < best_cost:
                best_cost = min_cost
                best_node = candidate
        
        return best_node

    # For 46-60 nodes remaining, add another range with focused strategy
    if 46 <= len(unvisited_nodes) <= 60:
        best_node = None
        best_cost = float('inf')
        
        # Focus on top 2 candidates to reduce computation
        candidates = sorted(unvisited_nodes, key=lambda x: distance_matrix[current_node][x])[:2]
        
        for candidate in candidates:
            remaining_after_candidate = unvisited_nodes - {candidate}
            
            # Look ahead 1 step with greedy continuation
            min_cost = float('inf')
            for next_candidate in remaining_after_candidate:
                cost = (distance_matrix[current_node][candidate] + 
                        distance_matrix[candidate][next_candidate])
                
                # Estimate the rest using greedy
                temp_remaining = remaining_after_candidate - {next_candidate}
                temp_pos = next_candidate
                
                while temp_remaining:
                    next_step = min(temp_remaining, key=lambda x: distance_matrix[temp_pos][x])
                    cost += distance_matrix[temp_pos][next_step]
                    temp_remaining.remove(next_step)
                    temp_pos = next_step
                
                cost += distance_matrix[temp_pos][destination_node]
                
                if cost < min_cost:
                    min_cost = cost
            
            if min_cost < best_cost:
                best_cost = min_cost
                best_node = candidate
        
        return best_node

    # For 61-80 nodes remaining, use focused strategy with minimal lookahead
    if 61 <= len(unvisited_nodes) <= 80:
        best_node = None
        best_cost = float('inf')
        
        # Focus on top 1-2 candidates to reduce computation
        candidates = sorted(unvisited_nodes, key=lambda x: distance_matrix[current_node][x])[:min(2, len(unvisited_nodes))]
        
        for candidate in candidates:
            remaining_after_candidate = unvisited_nodes - {candidate}
            
            # Minimal lookahead: just one greedy step
            if remaining_after_candidate:
                next_candidate = min(remaining_after_candidate, key=lambda x: distance_matrix[candidate][x])
                cost = (distance_matrix[current_node][candidate] + 
                        distance_matrix[candidate][next_candidate])
                
                # Estimate the rest using greedy
                temp_remaining = remaining_after_candidate - {next_candidate}
                temp_pos = next_candidate
                
                while temp_remaining:
                    next_step = min(temp_remaining, key=lambda x: distance_matrix[temp_pos][x])
                    cost += distance_matrix[temp_pos][next_step]
                    temp_remaining.remove(next_step)
                    temp_pos = next_step
                
                cost += distance_matrix[temp_pos][destination_node]
                
                if cost < best_cost:
                    best_cost = cost
                    best_node = candidate
            else:
                # If only one node left after candidate, just go directly to destination
                cost = distance_matrix[current_node][candidate] + distance_matrix[candidate][destination_node]
                if cost < best_cost:
                    best_cost = cost
                    best_node = candidate
        
        if best_node is not None:
            return best_node

    # For 81-100 nodes remaining, use focused strategy with minimal lookahead
    if 81 <= len(unvisited_nodes) <= 100:
        best_node = None
        best_cost = float('inf')
        
        # Focus on top 1-2 candidates to reduce computation
        candidates = sorted(unvisited_nodes, key=lambda x: distance_matrix[current_node][x])[:min(2, len(unvisited_nodes))]
        
        for candidate in candidates:
            remaining_after_candidate = unvisited_nodes - {candidate}
            
            # Minimal lookahead: just one greedy step
            if remaining_after_candidate:
                next_candidate = min(remaining_after_candidate, key=lambda x: distance_matrix[candidate][x])
                cost = (distance_matrix[current_node][candidate] + 
                        distance_matrix[candidate][next_candidate])
                
                # Estimate the rest using greedy
                temp_remaining = remaining_after_candidate - {next_candidate}
                temp_pos = next_candidate
                
                while temp_remaining:
                    next_step = min(temp_remaining, key=lambda x: distance_matrix[temp_pos][x])
                    cost += distance_matrix[temp_pos][next_step]
                    temp_remaining.remove(next_step)
                    temp_pos = next_step
                
                cost += distance_matrix[temp_pos][destination_node]
                
                if cost < best_cost:
                    best_cost = cost
                    best_node = candidate
            else:
                # If only one node left after candidate, just go directly to destination
                cost = distance_matrix[current_node][candidate] + distance_matrix[candidate][destination_node]
                if cost < best_cost:
                    best_cost = cost
                    best_node = candidate
        
        if best_node is not None:
            return best_node

    # For larger sets of unvisited nodes, use sophisticated scoring with enhanced features
    # Start with nearest neighbors but expand slightly for tie-breaking
    nearest_distance = min(distance_matrix[current_node][node] for node in unvisited_nodes)
    
    # Get all nodes that are within a small threshold of the nearest distance
    if len(unvisited_nodes) > 100:
        tolerance = 1.18
    elif len(unvisited_nodes) > 80:
        tolerance = 1.16
    elif len(unvisited_nodes) > 60:
        tolerance = 1.14
    elif len(unvisited_nodes) > 45:
        tolerance = 1.12
    elif len(unvisited_nodes) > 35:
        tolerance = 1.10
    elif len(unvisited_nodes) > 25:
        tolerance = 1.08
    else:
        tolerance = 1.06
        
    near_optimal_candidates = [
        node for node in unvisited_nodes 
        if distance_matrix[current_node][node] <= nearest_distance * tolerance
    ]
    
    if len(near_optimal_candidates) > 1:
        # Among near-optimal candidates, use a multi-factor scoring system
        def calculate_score(node):
            # Factor 1: Immediate distance to the node
            immediate_cost = distance_matrix[current_node][node]
            
            # Factor 2: Average distance to other unvisited nodes (connectivity)
            if len(unvisited_nodes) > 1:
                avg_connectivity = sum(distance_matrix[node][other] for other in unvisited_nodes if other != node) / (len(unvisited_nodes) - 1)
            else:
                avg_connectivity = 0
            
            # Factor 3: Minimum distance to any other unvisited node (future flexibility)
            if len(unvisited_nodes) > 1:
                min_future_cost = min(distance_matrix[node][other] for other in unvisited_nodes if other != node)
            else:
                min_future_cost = distance_matrix[node][destination_node]
            
            # Factor 4: Distance to destination (for bias toward home stretch)
            dist_to_dest = distance_matrix[node][destination_node]
            
            # Factor 5: Variance of distances to other unvisited nodes (preference for consistent connectivity)
            if len(unvisited_nodes) > 1:
                distances_to_others = [distance_matrix[node][other] for other in unvisited_nodes if other != node]
                variance_connectivity = np.var(distances_to_others)
            else:
                variance_connectivity = 0
                
            # Factor 6: Max distance to other unvisited nodes (to avoid very bad connections later)
            if len(unvisited_nodes) > 1:
                max_connectivity = max(distance_matrix[node][other] for other in unvisited_nodes if other != node)
            else:
                max_connectivity = 0
            
            # Factor 7: Second nearest unvisited node (to account for potential ties)
            sorted_distances = sorted([distance_matrix[node][other] for other in unvisited_nodes if other != node])
            second_nearest = sorted_distances[1] if len(sorted_distances) > 1 else sorted_distances[0]
            
            # Factor 8: Degree of connectivity - how well connected the node is to others
            # Count nodes that are closer than average distance from this node
            if len(unvisited_nodes) > 1:
                avg_distance_from_node = sum(distance_matrix[node][other] for other in unvisited_nodes if other != node) / (len(unvisited_nodes) - 1)
                well_connected_count = sum(1 for other in unvisited_nodes if other != node and distance_matrix[node][other] <= avg_distance_from_node)
                connectivity_ratio = well_connected_count / (len(unvisited_nodes) - 1)
            else:
                connectivity_ratio = 0
                
            # Factor 9: Centrality measure - how close this node is to all others on average
            if len(unvisited_nodes) > 1:
                centrality = sum(1.0 / (distance_matrix[node][other] + 1e-9) for other in unvisited_nodes if other != node)
            else:
                centrality = 0
            
            # Factor 10: How critical this node is for connecting distant nodes
            # Find the two farthest unvisited nodes and see if this node helps connect them
            if len(unvisited_nodes) > 2:
                unvisited_list = list(unvisited_nodes)
                max_dist_pairs = [(i, j) for i in range(len(unvisited_list)) for j in range(i+1, len(unvisited_list))]
                if max_dist_pairs:
                    max_dist_pair = max(max_dist_pairs, 
                                       key=lambda pair: distance_matrix[unvisited_list[pair[0]]][unvisited_list[pair[1]]])
                    far_node1, far_node2 = unvisited_list[max_dist_pair[0]], unvisited_list[max_dist_pair[1]]
                    bridge_value = distance_matrix[far_node1][node] + distance_matrix[node][far_node2] - distance_matrix[far_node1][far_node2]
                else:
                    bridge_value = 0
            else:
                bridge_value = 0
            
            # Factor 11: Penalty for creating long edges later
            # Consider the maximum distance among remaining nodes
            if len(unvisited_nodes) > 1:
                max_remaining_dist = max(distance_matrix[i][j] 
                                        for i in unvisited_nodes 
                                        for j in unvisited_nodes if i != j)
                # How does visiting this node affect our ability to handle large distances?
                max_from_node = max(distance_matrix[node][j] for j in unvisited_nodes if j != node)
            else:
                max_from_node = 0
                
            # Factor 12: Balance between closeness to current and accessibility to others
            if len(unvisited_nodes) > 1:
                accessibility = sum(1.0 / (distance_matrix[other][node] + 1e-9) for other in unvisited_nodes if other != node)
                balance_metric = (immediate_cost / (avg_connectivity + 1e-9)) * (len(unvisited_nodes) / (accessibility + 1e-9))
            else:
                balance_metric = immediate_cost
                
            # Factor 13: Closeness to the geometric center of remaining nodes
            if len(unvisited_nodes) > 1:
                # Approximate center by averaging coordinates if available, otherwise use mean distances
                center_distance = sum(distance_matrix[node][j] for j in unvisited_nodes if j != node) / (len(unvisited_nodes) - 1)
            else:
                center_distance = 0
                
            # Factor 14: Edge density around this node - how many nearby nodes are there
            if len(unvisited_nodes) > 2:
                avg_dist = sum(distance_matrix[node][j] for j in unvisited_nodes if j != node) / (len(unvisited_nodes) - 1)
                near_count = sum(1 for j in unvisited_nodes if j != node and distance_matrix[node][j] <= avg_dist)
                edge_density = near_count / (len(unvisited_nodes) - 1)
            else:
                edge_density = 0
                
            # Factor 15: Future penalty based on how this choice limits future options
            # Count number of nodes that would become unreachable if we go to this node next and then have to go to destination
            if len(unvisited_nodes) > 1:
                # Calculate how "out of the way" this node would be relative to the destination
                detour_penalty = distance_matrix[current_node][node] + distance_matrix[node][destination_node] - distance_matrix[current_node][destination_node]
                # Ensure this is non-negative
                detour_penalty = max(0, detour_penalty)
            else:
                detour_penalty = 0
                
            # Factor 16: How much this node reduces the overall diameter of remaining nodes
            if len(unvisited_nodes) > 2:
                original_diameter = max(distance_matrix[i][j] for i in unvisited_nodes for j in unvisited_nodes if i != j)
                new_unvisited = unvisited_nodes - {node}
                if len(new_unvisited) > 1:
                    new_diameter = max(distance_matrix[i][j] for i in new_unvisited for j in new_unvisited if i != j)
                    diameter_reduction = original_diameter - new_diameter
                else:
                    diameter_reduction = original_diameter
            else:
                diameter_reduction = 0

            # Factor 17: Angle-based heuristic - how much direction changes when going through this node
            # This helps maintain a more direct path overall
            if len(unvisited_nodes) > 1 and current_node != destination_node:
                # Calculate angle between current->node and node->destination
                direct_dist = distance_matrix[current_node][destination_node]
                path_dist = distance_matrix[current_node][node] + distance_matrix[node][destination_node]
                angular_penalty = max(0, path_dist - direct_dist)
            else:
                angular_penalty = 0

            # Weighted combination of factors
            # Adjust weights based on how many nodes remain
            if len(unvisited_nodes) > 100:
                # Very early in the tour, focus on global connectivity and bridging
                weights = [0.12, 0.10, 0.06, 0.02, 0.04, 0.02, 0.02, 0.02, 0.02, 0.14, 0.04, 0.05, 0.05, 0.03, 0.10, 0.05, 0.33]
            elif len(unvisited_nodes) > 80:
                # Very early in the tour, focus on global connectivity and bridging
                weights = [0.13, 0.11, 0.06, 0.02, 0.04, 0.02, 0.02, 0.02, 0.02, 0.14, 0.04, 0.05, 0.05, 0.03, 0.10, 0.05, 0.32]
            elif len(unvisited_nodes) > 60:
                # Very early in the tour, focus on global connectivity and bridging
                weights = [0.14, 0.12, 0.07, 0.03, 0.05, 0.03, 0.03, 0.03, 0.03, 0.15, 0.05, 0.06, 0.06, 0.04, 0.12, 0.06, 0.15]
            elif len(unvisited_nodes) > 45:
                # Very early in the tour, focus on global connectivity and bridging
                weights = [0.15, 0.12, 0.07, 0.03, 0.05, 0.03, 0.03, 0.03, 0.03, 0.16, 0.05, 0.06, 0.06, 0.04, 0.11, 0.05, 0.12]
            elif len(unvisited_nodes) > 35:
                # Very early in the tour, focus on global connectivity and bridging
                weights = [0.16, 0.13, 0.08, 0.04, 0.06, 0.04, 0.04, 0.04, 0.04, 0.17, 0.06, 0.07, 0.07, 0.05, 0.08, 0.05, 0.08]
            elif len(unvisited_nodes) > 25:
                # Early in the tour, focus more on global connectivity
                weights = [0.17, 0.14, 0.09, 0.05, 0.07, 0.05, 0.05, 0.05, 0.05, 0.18, 0.07, 0.08, 0.06, 0.04, 0.06, 0.04, 0.04]
            elif len(unvisited_nodes) > 20:
                # Early in the tour, focus more on global connectivity
                weights = [0.18, 0.15, 0.09, 0.06, 0.07, 0.06, 0.06, 0.06, 0.05, 0.17, 0.06, 0.06, 0.04, 0.03, 0.04, 0.03, 0.03]
            elif len(unvisited_nodes) > 10:
                # Mid-tour, balance immediate and future costs
                weights = [0.22, 0.15, 0.10, 0.12, 0.07, 0.06, 0.05, 0.05, 0.06, 0.08, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02]
            else:
                # Late in the tour, consider immediate and destination costs more
                weights = [0.32, 0.10, 0.08, 0.20, 0.07, 0.06, 0.05, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00]
            
            score = (weights[0] * immediate_cost + 
                     weights[1] * avg_connectivity + 
                     weights[2] * min_future_cost + 
                     weights[3] * dist_to_dest +
                     weights[4] * variance_connectivity +
                     weights[5] * max_connectivity +
                     weights[6] * second_nearest +
                     weights[7] * (1 - connectivity_ratio) +  # Prefer nodes with good connectivity (low inverse)
                     weights[8] * (1 - centrality / len(unvisited_nodes)) +  # Prefer central nodes
                     weights[9] * bridge_value +  # Prefer nodes that help bridge distant points
                     weights[10] * max_from_node +  # Penalty for nodes that create large max distances
                     weights[11] * balance_metric +  # Balance metric
                     weights[12] * center_distance +  # Closeness to center
                     weights[13] * (1 - edge_density) +  # Prefer nodes with good local connectivity
                     weights[14] * detour_penalty +   # Penalty for unnecessary detours
                     weights[15] * (original_diameter - diameter_reduction if 'original_diameter' in locals() and len(unvisited_nodes) > 2 else 0) +  # Reward diameter reduction
                     weights[16] * angular_penalty)  # Angular penalty for direction changes
            
            return score
        
        return min(near_optimal_candidates, key=calculate_score)
    else:
        # Just return the nearest node if no tie exists
        return min(unvisited_nodes, key=lambda node: distance_matrix[current_node][node])