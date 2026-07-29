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
    # If there's only one unvisited node left, return it
    if len(unvisited_nodes) == 1:
        return unvisited_nodes.pop()
    
    best_node = None
    best_score = float('inf')
    
    # For each candidate node, evaluate using a scoring approach
    for node in unvisited_nodes:
        # Immediate cost to reach this node
        immediate_cost = distance_matrix[current_node][node]
        
        # Calculate remaining cost with a greedy approach
        remaining_nodes = list(unvisited_nodes - {node})
        if remaining_nodes:
            # Estimate remaining cost by greedy selection
            total_remaining_cost = 0
            current_pos = node
            
            # Greedy path from the selected node through remaining nodes
            temp_remaining = remaining_nodes.copy()
            while temp_remaining:
                closest = min(temp_remaining, key=lambda x: distance_matrix[current_pos][x])
                total_remaining_cost += distance_matrix[current_pos][closest]
                current_pos = closest
                temp_remaining.remove(closest)
            
            # Add return cost to destination
            total_remaining_cost += distance_matrix[current_pos][destination_node]
        else:
            total_remaining_cost = distance_matrix[node][destination_node]
        
        # Enhanced scoring with better balance and look-ahead
        score = immediate_cost + total_remaining_cost * 0.7
        
        # Consider also the distance from the node to the destination to avoid dead ends
        destination_proximity = distance_matrix[node][destination_node]
        score += destination_proximity * 0.15  # Adjusted penalty based on how far the node is from destination
        
        # Add a small penalty for nodes that are too far from current node relative to average distances
        avg_distance = sum(distance_matrix[current_node]) / len(distance_matrix[current_node])
        if avg_distance > 0:
            relative_distance = immediate_cost / avg_distance
            score += relative_distance * 0.05 * total_remaining_cost
        
        if score < best_score:
            best_score = score
            best_node = node
    
    return best_node