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
from typing import Set, List
import itertools

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    """
    Select the next node to visit in a Traveling Salesman Problem (TSP) constructive heuristic.
    
    Implementation idea: Use nearest neighbor approach with simulated 2-opt improvement
    during construction. For each candidate node, we estimate the remaining tour cost
    by creating a greedy path and simulating potential 2-opt improvements without
    actually performing them, which provides lookahead while maintaining efficiency.
    
    Args:
        current_node (int): The node currently being visited
        destination_node (int): The final destination node in the TSP tour
        unvisited_nodes (set): Set of nodes that haven't been visited yet
        distance_matrix (np.ndarray): NxN matrix where distance_matrix[i][j] is the distance from node i to j

    Returns:
        int: The selected next node to visit
    """
    if len(unvisited_nodes) == 1:
        # If only one unvisited node left, just return it
        return next(iter(unvisited_nodes))
    
    # Calculate number of candidates based on available unvisited nodes
    num_candidates = min(5, len(unvisited_nodes))  # Go back to 5 candidates for better selection
    
    # Get nearest neighbors from unvisited nodes
    current_row = distance_matrix[current_node]
    # Sort unvisited nodes by their distance from current node
    sorted_unvisited = sorted(unvisited_nodes, key=lambda node: current_row[node])
    candidate_nodes = sorted_unvisited[:num_candidates]
    
    # Evaluate each candidate with lookahead including simulated 2-opt improvement
    best_candidate = None
    best_score = float('inf')
    
    for candidate in candidate_nodes:
        # Calculate immediate cost: distance from current to candidate
        immediate_cost = distance_matrix[current_node][candidate]
        
        # Calculate lookahead estimate using simulated 2-opt improvement
        temp_unvisited = unvisited_nodes - {candidate}
        
        # Estimate the remaining cost using NN path with simulated 2-opt improvement
        estimated_remaining_cost = estimate_remaining_tour_cost_with_2opt_simulation(
            candidate, destination_node, temp_unvisited, distance_matrix
        )
        
        # Total estimated score combines immediate cost and estimated future cost
        score = immediate_cost + estimated_remaining_cost
        
        if score < best_score:
            best_score = score
            best_candidate = candidate
    
    # Fallback to nearest neighbor if no best candidate was found (should not happen)
    if best_candidate is None:
        best_candidate = min(unvisited_nodes, key=lambda node: distance_matrix[current_node][node])
    
    return best_candidate

def estimate_remaining_tour_cost_with_2opt_simulation(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Estimation of the remaining tour cost using nearest neighbor path with simulated 2-opt improvement.
    Instead of actually performing 2-opt, we simulate the potential for improvement by checking
    for possible beneficial swaps in the greedy path.
    """
    if not unvisited_nodes:
        return distance_matrix[current_node][destination_node]
    
    # Create a temporary path using nearest neighbor
    temp_path = construct_greedy_path(current_node, unvisited_nodes, destination_node, distance_matrix)
    
    # Calculate the base path length
    base_length = 0
    for i in range(len(temp_path) - 1):
        base_length += distance_matrix[temp_path[i]][temp_path[i + 1]]
    
    # Simulate potential 2-opt improvement by finding the best possible swap
    potential_improvement = find_best_potential_2opt_improvement(temp_path, distance_matrix)
    
    # Return the base length minus the potential improvement
    # This gives us a more optimistic estimate of the remaining cost
    return base_length - potential_improvement

def construct_greedy_path(start_node, unvisited_nodes, end_node, distance_matrix):
    """
    Construct a greedy path visiting all unvisited nodes and ending at the destination node.
    Returns the path as a list of nodes.
    """
    path = [start_node]
    remaining_nodes = unvisited_nodes.copy()
    current = start_node
    
    while remaining_nodes:
        # Find closest unvisited node
        next_node = min(remaining_nodes, key=lambda node: distance_matrix[current][node])
        path.append(next_node)
        current = next_node
        remaining_nodes.remove(next_node)
    
    # Add destination node
    path.append(end_node)
    
    return path

def find_best_potential_2opt_improvement(path, distance_matrix):
    """
    Find the best potential 2-opt improvement without actually applying it.
    This function looks for the best possible 2-opt swap that could reduce the path length.
    """
    n = len(path)
    best_improvement = 0
    
    # Check all possible 2-opt swaps
    for i in range(n - 3):  # Need at least 3 edges to swap
        for k in range(i + 2, n - 1):
            # Calculate the improvement from swapping edges (i, i+1) and (k, k+1)
            # Current edges: path[i]-path[i+1] and path[k]-path[k+1]
            # New edges: path[i]-path[k] and path[i+1]-path[k+1]
            current_edges_cost = (
                distance_matrix[path[i]][path[i+1]] + 
                distance_matrix[path[k]][path[k+1]]
            )
            new_edges_cost = (
                distance_matrix[path[i]][path[k]] + 
                distance_matrix[path[i+1]][path[k+1]]
            )
            
            improvement = current_edges_cost - new_edges_cost
            
            if improvement > best_improvement:
                best_improvement = improvement
    
    return best_improvement