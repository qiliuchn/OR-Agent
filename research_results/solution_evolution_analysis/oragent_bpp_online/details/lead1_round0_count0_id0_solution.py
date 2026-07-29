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

def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """
    Best Fit heuristic: prioritize bins with smallest remaining capacity that can still fit the item.
    """
    scores = np.zeros_like(bins_remain_cap)
    
    # Can the bin fit the item?
    feasible = bins_remain_cap >= item
    
    # For feasible bins: higher priority to bins with LESS remaining space
    # Invert the capacity so smaller remaining = higher score
    scores = np.where(feasible, -bins_remain_cap, -np.inf)
    
    return scores