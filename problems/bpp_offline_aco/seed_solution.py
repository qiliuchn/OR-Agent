import numpy as np

def heuristics(demand: np.ndarray, capacity: int) -> np.ndarray:
    return np.tile(demand/demand.max(), (demand.shape[0], 1))