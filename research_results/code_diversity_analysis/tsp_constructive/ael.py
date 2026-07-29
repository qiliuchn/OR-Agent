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

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    """
    Select the next node to visit in a Traveling Salesman Problem (TSP) constructive heuristic.

    Args:
        current_node (int): The node currently being visited
        destination_node (int): The final destination node in the TSP tour
        unvisited_nodes (set): Set of nodes that haven't been visited yet
        distance_matrix (np.ndarray): NxN matrix where distance_matrix[i][j] is the distance from node i to j

    Returns:
        int: The selected next node to visit
    """
    # If there's only one unvisited node left, visit it
    if len(unvisited_nodes) == 1:
        return unvisited_nodes.pop()
    
    # If there are no unvisited nodes except the destination, go to destination
    if not unvisited_nodes:
        return destination_node
    
    # Calculate some useful metrics
    n_remaining = len(unvisited_nodes)
    
    # For small number of remaining nodes, use exact calculation
    if n_remaining <= 3:
        best_score = float('inf')
        next_node = None
        
        for node in unvisited_nodes:
            # Calculate the exact cost of visiting this node and then completing the tour
            remaining_after_choice = unvisited_nodes - {node}
            completion_cost = calculate_exact_completion_cost(node, remaining_after_choice, distance_matrix, destination_node)
            
            score = distance_matrix[current_node][node] + completion_cost
            
            if score < best_score:
                best_score = score
                next_node = node
        
        return next_node
    else:
        # For larger problems, use a combination of greedy and look-ahead heuristics
        best_score = float('inf')
        next_node = None
        
        # Calculate global statistics for normalization
        avg_distance = np.mean(distance_matrix[distance_matrix > 0])
        max_distance = np.max(distance_matrix)
        
        for candidate_node in unvisited_nodes:
            # Immediate cost
            immediate_cost = distance_matrix[current_node][candidate_node]
            
            # Calculate remaining nodes after choosing this candidate
            remaining_after_choice = unvisited_nodes - {candidate_node}
            
            # Estimate future cost based on remaining nodes
            if len(remaining_after_choice) == 0:
                # Last node before destination
                score = immediate_cost + distance_matrix[candidate_node][destination_node]
            elif len(remaining_after_choice) == 1:
                # Two nodes left: this one and destination
                other_node = next(iter(remaining_after_choice))
                score = (immediate_cost + 
                         distance_matrix[candidate_node][other_node] + 
                         distance_matrix[other_node][destination_node])
            else:
                # Multiple nodes remaining - use greedy look-ahead heuristic
                # Look at the nearest neighbor from the candidate to remaining nodes
                nearest_neighbor = min(remaining_after_choice, 
                                       key=lambda x: distance_matrix[candidate_node][x])
                
                # Estimate the remaining path cost
                estimated_remaining = estimate_remaining_path(candidate_node, remaining_after_choice, distance_matrix, destination_node)
                
                # Use a weighted formula that balances immediate and future costs
                # Weight more toward immediate costs when many nodes remain
                # Weight more toward future costs when few nodes remain
                future_weight = min(0.7, 0.3 + 0.4 * (1 - len(remaining_after_choice)/len(unvisited_nodes)))
                immediate_weight = 1.0 - future_weight
                
                score = immediate_weight * immediate_cost + future_weight * estimated_remaining
                
                # Add penalty for choosing very distant nodes early in the tour
                if n_remaining > len(distance_matrix) // 3 and immediate_cost > avg_distance * 1.5:
                    score *= 1.1  # Penalty for choosing a far node early on
        
            if score < best_score:
                best_score = score
                next_node = candidate_node
        
        return next_node


def calculate_exact_completion_cost(current_node, remaining_nodes, distance_matrix, destination_node):
    """Calculate the exact minimum cost to visit all remaining nodes and return to destination."""
    if not remaining_nodes:
        return distance_matrix[current_node][destination_node]
    
    if len(remaining_nodes) == 1:
        node = remaining_nodes.pop()
        return distance_matrix[current_node][node] + distance_matrix[node][destination_node]
    
    # For up to 3 remaining nodes, evaluate all permutations
    import itertools
    min_cost = float('inf')
    
    for perm in itertools.permutations(remaining_nodes):
        cost = 0
        prev = current_node
        for node in perm:
            cost += distance_matrix[prev][node]
            prev = node
        cost += distance_matrix[prev][destination_node]
        
        if cost < min_cost:
            min_cost = cost
    
    return min_cost


def estimate_remaining_path(current_node, remaining_nodes, distance_matrix, destination_node):
    """
    Estimate the remaining path cost using a greedy nearest neighbor heuristic
    """
    if not remaining_nodes:
        return distance_matrix[current_node][destination_node]
    
    total_dist = 0
    current = current_node
    unvisited = remaining_nodes.copy()
    
    # Use greedy nearest neighbor for the remaining nodes
    while unvisited:
        closest_node = min(unvisited, key=lambda x: distance_matrix[current][x])
        total_dist += distance_matrix[current][closest_node]
        current = closest_node
        unvisited.remove(closest_node)
    
    # Add the final leg back to the destination
    total_dist += distance_matrix[current][destination_node]
    
    return total_dist