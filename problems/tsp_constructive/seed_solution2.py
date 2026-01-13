""" 
The algorithm selects the next node to visit in a Traveling Salesman Problem (TSP) tour by balancing four competing criteria through a weighted scoring system. The node with the lowest composite score is chosen as the next destination.

Scoring Criteria (with weights):
1. Distance from current node (weight c1 = 0.4): Prefer nodes closer to current position
2. Average distance to other unvisited nodes (weight c2 = 0.3): Prefer nodes that are centrally located among remaining nodes
3. Standard deviation of distances to unvisited nodes (weight c3 = 0.2): Prefer nodes with varied distances to other unvisited nodes
4. Distance to final destination (weight c4 = 0.1): Consider proximity to the ultimate destination

Score Calculation:
score = c1 * dist(current, node)
        - c2 * avg_dist(node, unvisited)
        + c3 * std_dev_dist(node, unvisited)
        - c4 * dist(destination, node)

Key Observations:
- Lower scores are better (minimum score is selected)
- The heuristic balances immediate proximity (c1) with strategic positioning (c2, c3, c4)
- The negative sign on c2 and c4 means lower average distances and lower destination distances improve the score
- The positive sign on c3 means higher variance in distances improves the score

Algorithm Characteristics:
- Deterministic greedy selection (always picks minimum score)
- Considers both local (current node) and global (destination) information
- Uses statistical properties (mean, standard deviation) of distances to unvisited nodes
- Weight coefficients can be tuned for different problem characteristics

Potential Improvements:
- The 'threshold = 0.7' parameter is currently unused but could enable probabilistic selection
- Weight coefficients could be optimized for specific problem instances
- Could incorporate additional criteria like node degree or clustering information
"""
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
    threshold = 0.7  # Unused threshold parameter - could be used for probabilistic selection
    c1, c2, c3, c4 = 0.4, 0.3, 0.2, 0.1  # Weight coefficients for different criteria

    scores = {}
    for node in unvisited_nodes:
        # Calculate distances from this node to all other unvisited nodes (excluding itself)
        all_distances = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
        average_distance_to_unvisited = np.mean(all_distances)
        std_dev_distance_to_unvisited = np.std(all_distances)

        # Compute composite score using weighted sum of criteria
        # Lower scores are better (minimum is selected)
        score = (c1 * distance_matrix[current_node][node]  # Distance from current node
                 - c2 * average_distance_to_unvisited       # Negative weight: prefer central nodes
                 + c3 * std_dev_distance_to_unvisited       # Positive weight: prefer varied distances
                 - c4 * distance_matrix[destination_node][node])  # Negative weight: prefer nodes near destination

        scores[node] = score

    # Select the node with the minimum score (best according to our criteria)
    next_node = min(scores, key=scores.get)
    return next_node
