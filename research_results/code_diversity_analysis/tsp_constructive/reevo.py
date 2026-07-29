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
    
    This heuristic implements an enhanced greedy approach with adaptive lookahead.
    It balances immediate greedy selection with future path estimation using a 
    simplified but effective lookahead mechanism. The algorithm focuses on building
    efficient local segments while maintaining computational efficiency.
    
    Args:
        current_node (int): The node currently being visited
        destination_node (int): The final destination node in the TSP tour
        unvisited_nodes (set): Set of nodes that haven't been visited yet
        distance_matrix (np.ndarray): NxN matrix where distance_matrix[i][j] is the distance from node i to j

    Returns:
        int: The selected next node to visit
    """
    # Handle base cases directly
    if len(unvisited_nodes) == 1:
        return next(iter(unvisited_nodes))
    
    if len(unvisited_nodes) == 2:
        unvisited_list = list(unvisited_nodes)
        node1, node2 = unvisited_list[0], unvisited_list[1]
        
        # Evaluate both possible complete paths
        path1_cost = (distance_matrix[current_node][node1] + 
                      distance_matrix[node1][node2] + 
                      distance_matrix[node2][destination_node])
        path2_cost = (distance_matrix[current_node][node2] + 
                      distance_matrix[node2][node1] + 
                      distance_matrix[node1][destination_node])
        
        return node1 if path1_cost < path2_cost else node2
    
    # Dynamic lookahead depth based on remaining nodes
    n_remaining = len(unvisited_nodes)
    lookahead_depth = min(max(2, n_remaining // 4), 3)
    
    # Balance between immediate and future costs
    alpha = max(0.4, min(0.8, 0.6 + 0.1 * np.log(n_remaining / 10)))
    beta = 1.0 - alpha
    
    best_node = None
    best_score = float('inf')
    
    for node in unvisited_nodes:
        # Immediate cost to reach this node
        immediate_cost = distance_matrix[current_node][node]
        
        # Estimate the continuation cost using greedy lookahead
        future_cost = estimate_future_cost_with_lookahead(
            node, unvisited_nodes - {node}, destination_node,
            distance_matrix, lookahead_depth
        )
        
        # Combined score with dynamic weights
        total_score = alpha * immediate_cost + beta * future_cost
        
        if total_score < best_score:
            best_score = total_score
            best_node = node
    
    return best_node


def estimate_future_cost_with_lookahead(current_node: int, remaining_nodes: set, destination_node: int,
                                      distance_matrix: np.ndarray, lookahead_depth: int) -> float:
    """
    Estimate the cost of visiting all remaining nodes and returning to destination
    using a greedy approach with limited lookahead.
    """
    if not remaining_nodes:
        return distance_matrix[current_node][destination_node]
    
    # For small remaining sets, use exact calculation
    if len(remaining_nodes) <= 3:
        return calculate_exact_cost_small_set(current_node, remaining_nodes, destination_node, distance_matrix)
    
    # Use greedy lookahead approach
    unvisited = remaining_nodes.copy()
    current = current_node
    total_cost = 0
    
    # Perform greedy lookahead steps
    for _ in range(min(lookahead_depth, len(unvisited))):
        next_node = min(unvisited, key=lambda x: distance_matrix[current][x])
        total_cost += distance_matrix[current][next_node]
        current = next_node
        unvisited.remove(next_node)
    
    # For remaining nodes, use greedy approach
    while unvisited:
        next_node = min(unvisited, key=lambda x: distance_matrix[current][x])
        total_cost += distance_matrix[current][next_node]
        current = next_node
        unvisited.remove(next_node)
    
    # Add cost back to destination
    total_cost += distance_matrix[current][destination_node]
    
    return total_cost


def calculate_exact_cost_small_set(current_node: int, remaining_nodes: set, destination_node: int,
                                 distance_matrix: np.ndarray) -> float:
    """
    Calculate the exact cost for small remaining sets by evaluating all permutations.
    """
    if not remaining_nodes:
        return distance_matrix[current_node][destination_node]
    
    nodes_list = list(remaining_nodes)
    n = len(nodes_list)
    
    if n == 1:
        return distance_matrix[current_node][nodes_list[0]] + distance_matrix[nodes_list[0]][destination_node]
    
    if n == 2:
        node1, node2 = nodes_list[0], nodes_list[1]
        path1_cost = (distance_matrix[current_node][node1] + 
                      distance_matrix[node1][node2] + 
                      distance_matrix[node2][destination_node])
        path2_cost = (distance_matrix[current_node][node2] + 
                      distance_matrix[node2][node1] + 
                      distance_matrix[node1][destination_node])
        return min(path1_cost, path2_cost)
    
    if n == 3:
        import itertools
        min_cost = float('inf')
        for perm in itertools.permutations(nodes_list):
            cost = distance_matrix[current_node][perm[0]]
            cost += distance_matrix[perm[0]][perm[1]]
            cost += distance_matrix[perm[1]][perm[2]]
            cost += distance_matrix[perm[2]][destination_node]
            min_cost = min(min_cost, cost)
        return min_cost
    
    # Fallback for any other case
    return distance_matrix[current_node][destination_node]