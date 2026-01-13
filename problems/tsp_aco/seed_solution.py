import numpy as np

def heuristics(distance_matrix: np.ndarray) -> np.ndarray:
    """TSP-ACO Heuristic"""
    return 1 / distance_matrix