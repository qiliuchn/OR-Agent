# Evaluation script for MKP-ACO problem
""" 
Select a subset of n items to maximize total prize, subject to m capacity constraints. 
Inputs:
- prize: shape (n,) - prize/value of each item
- weight: shape (n, m) - weight[i, j] = weight of item i in constraint dimension j
Constraints:
- For each constraint dimension j (1 ≤ j ≤ m), sum of weights of selected items ≤ 1 (normalized capacity). 
Objective: 
- Maximize ∑(prize[i] for selected items i) Key characteristics:
Binary selection: Each item either selected (1) or not (0)
Capacity normalized: All constraints have capacity 1 after normalization
ACO approach: Artificial ants construct solutions by probabilistically selecting items based on pheromone trails and heuristic desirability, with infeasible items (exceeding any constraint) removed from consideration.
"""
import os
import sys
import traceback
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any
import torch
from torch.distributions import Categorical
import seed_solution as solution_module  # Note: solution module script is generated and saved on the fly


# =====Load function to evolve=====
problem = "mkp_aco"
heuristics = getattr(solution_module, "heuristics")  # Get function to evolve


# =====ACO class=====
class ACO():
    def __init__(self,  # constraints are set to 1 after normalize weight 
                 prize,  # shape [n,]
                 weight, # shape [n, m]
                 heuristic,
                 n_ants=30, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 device='cpu'
                 ):
        self.n, self.m = weight.shape

        self.prize = prize
        self.weight = weight
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

        self.pheromone = torch.ones(size=(self.n+1,), device=device)

        # Fidanova S. Hybrid ant colony optimization algorithm for multiple knapsack problem
        # self.heuristic = prize / self.weight.sum(dim=1) if heuristic is None else heuristic
        self.heuristic = heuristic
        # Leguizamon G, Michalewicz Z. A New Version of Ant System for Subset Problems
        self.Q = 1 / self.prize.sum()

        self.alltime_best_sol = None
        self.alltime_best_obj = 0
        self.device = device
        self.add_dummy_node()
        
    def add_dummy_node(self):
        self.prize = torch.cat((self.prize, torch.tensor([0.], device=self.device))) # (n+1,)
        self.weight = torch.cat((self.weight, torch.zeros((1, self.m), device=self.device)), dim=0) # (n+1, m)
        self.heuristic = torch.cat((self.heuristic, torch.tensor([1e-8], device=self.device))) # (n+1)

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            sols = self.gen_sol() # (n_ants, max_horizon)
            objs = self.gen_sol_obj(sols)             # (n_ants,)
            sols = sols.T
            best_obj, best_idx = objs.max(dim=0)
            if best_obj > self.alltime_best_obj:
                self.alltime_best_obj = best_obj
                self.alltime_best_sol = sols[best_idx]
            self.update_pheronome(sols, objs, best_obj.item(), best_idx.item())
        return self.alltime_best_obj, self.alltime_best_sol

    @torch.no_grad()
    def update_pheronome(self, sols, objs, best_obj, best_idx):
        self.pheromone = self.pheromone * self.decay 
        for i in range(self.n_ants):
            sol = sols[i]
            obj = objs[i]
            self.pheromone[sol] += self.Q * obj

    @torch.no_grad()
    def gen_sol_obj(self, solutions):
        '''
        Args:
            solutions: (n_ants, max_horizon)
        Return:
            obj: (n_ants,)
        '''
        return self.prize[solutions.T].sum(dim=1) # (n_ants,)

    def gen_sol(self):
        '''
        Solution contruction for all ants
        '''
        solutions = [] # solutions[i] is the i-th picked item for all ants
        knapsack = torch.zeros(size=(self.n_ants, self.m), device=self.device)  # used capacity
        mask = torch.ones(size=(self.n_ants, self.n+1), device=self.device)
        dummy_mask = torch.ones(size=(self.n_ants, self.n+1), device=self.device)
        dummy_mask[:, -1] = 0
        
        mask, knapsack = self.update_knapsack(mask, knapsack, new_item=None)
        dummy_mask = self.update_dummy_state(mask, dummy_mask)
        done = self.check_done(mask)
        while not done:
            items = self.pick_item(mask, dummy_mask)
            solutions.append(items)
            mask, knapsack = self.update_knapsack(mask, knapsack, items)
            dummy_mask = self.update_dummy_state(mask, dummy_mask)
            done = self.check_done(mask)
        return torch.stack(solutions)
    
    def pick_item(self, mask, dummy_mask):
        phe = self.pheromone.unsqueeze(0).repeat(self.n_ants, 1)
        heu = self.heuristic.unsqueeze(0).repeat(self.n_ants, 1)
        dist = ((phe ** self.alpha) * (heu ** self.beta) * mask * dummy_mask) # (n_ants, n+1)
        dist = Categorical(dist)
        item = dist.sample()
        return item # (n_ants,)
    
    def check_done(self, mask):
        # is mask all zero except for the dummy node?
        return (mask[:, :-1] == 0).all()
    
    def update_dummy_state(self, mask, dummy_mask):
        finished = (mask[: ,:-1] == 0).all(dim=1)
        dummy_mask[finished] = 1
        return dummy_mask
    
    def update_knapsack(self, mask, knapsack, new_item):
        '''
        Args:
            mask: (n_ants, n+1)
            knapsack: (n_ants, m)
            new_item: (n_ants)
        '''
        if new_item is not None:
            mask[torch.arange(self.n_ants), new_item] = 0
            knapsack += self.weight[new_item] # (n_ants, m)
        for ant_idx in range(self.n_ants):
            candidates = torch.nonzero(mask[ant_idx]) # (x, 1)
            if len(candidates) > 1:
                candidates.squeeze_()
                test_knapsack = knapsack[ant_idx].unsqueeze(0).repeat(len(candidates), 1) # (x, m)
                new_knapsack = test_knapsack + self.weight[candidates] # (x, m)
                infeasible_idx = candidates[(new_knapsack > 1).any(dim=1)]
                mask[ant_idx, infeasible_idx] = 0
        mask[:, -1] = 1
        return mask, knapsack


# =====Evaluation function=====
N_ITERATIONS = 50 # reevo paper: 50
N_ANTS = 30 # reevo papr: 10

def evaluate_heuristic(prize: np.ndarray, weight: np.ndarray):
    n, m = weight.shape
    heu = heuristics(prize.copy(), weight.copy()) + 1e-9
    assert heu.shape == (n,)
    heu[heu < 1e-9] = 1e-9
    aco = ACO(torch.from_numpy(prize), torch.from_numpy(weight), torch.from_numpy(heu), N_ANTS)
    obj, _ = aco.run(N_ITERATIONS)
    return obj


# =====Helper functions=====
def get_feature(metrics: Dict[int, float]) -> Tuple[int, ...]:
    """
    Convert the metrics dict to a feature vector

    Args:
        metrics (dict): A mapping of test problem size (int) to a score (float).

    Returns:
        (tuple): a tuple of discretized scores sorted by problem size
    """
    scores = metrics.values()
    features = tuple([int(x) for x in scores])
    return features
    
def get_score(metrics: Dict[int, float]) -> float:
    """
    Convert the metrics dict to a score

    Args:
        metrics (dict): A mapping of test problem size (int) to a score (float).

    Returns:
        (float): a score
    """
    return sum(metrics.values()) / len(metrics)


# =====Main function=====
if __name__ == "__main__":
    # -----Parse command line arguments (same for all problems)-----
    parser = argparse.ArgumentParser(description='Evaluation script.')
    parser.add_argument(
        '--root_dir',
        type=str,
        default=os.getcwd(),
        help='Project root directory for loading data (default: current working directory)'
    )
    parser.add_argument(
        '--file_output_prefix',
        type=str,
        default='',
        help='Output file prefix for saving evaluation results. '
             'Absolute path recommended. Files saved as {prefix}filename '
             '(default: empty string, saves to current directory)')
    parser.add_argument(
        '--mode',
        type=str,
        default='val',
        choices=['train', 'val'],
        help='Execution mode: train or val (default: val)'
    )
    parser.add_argument(
        '--problem_size',
        type=int,
        default=50,  # Customize this to your needs
        help='Problem size parameter'
    )
    # Parse arguments
    args = parser.parse_args()
    root_dir = args.root_dir
    file_output_prefix = args.file_output_prefix
    mode = args.mode
    problem_size = args.problem_size
    # Print parsed arguments for verification
    print(f"root_dir: {root_dir}")
    print(f"file_output_prefix: {file_output_prefix}")
    print(f"mode: {mode}")
    #print(f"problem_size: {problem_size}")
    
    # -----Run the evaluation-----
    # Run instances: 100, 300, 500; execution time: 44s
    try:
        basepath = os.path.join(root_dir, "problems", problem)
        
        if mode == 'train':
            dataset_path = os.path.join(basepath, f"dataset/{mode}{problem_size}_dataset.npz")
            dataset = np.load(dataset_path)
            prizes, weights = dataset['prizes'], dataset['weights']
            n_instances = prizes.shape[0]

            print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
            
            objs = []
            for i, (prize, weight) in enumerate(zip(prizes, weights)):
                obj = evaluate_heuristic(prize, weight)
                print(f"[*] Instance {i}: {obj}")
                objs.append(obj.item())
            
            print("[*] Average:")
            print(np.mean(objs))

        else: # mood == 'val'
            metrics = {}
            for problem_size in [100, 300]:  # options: 100, 300, 500
                dataset_path = os.path.join(basepath, f"dataset/{mode}{problem_size}_dataset.npz")
                dataset = np.load(dataset_path)
                prizes, weights = dataset['prizes'], dataset['weights']
                n_instances = prizes.shape[0]
                print(f"[*] Evaluating {dataset_path}")

                objs = []
                for i, (prize, weight) in enumerate(zip(prizes, weights)):
                    obj = evaluate_heuristic(prize, weight)
                    objs.append(obj.item())
                
                print(f"[*] Average for {problem_size}: {np.mean(objs)}")
                metrics[problem_size] = np.mean(objs)
                
        if metrics:
            features = get_feature(metrics)
            score = get_score(metrics)
        else:
            features = None
            score = None

        # -----Print results to stdout (same for all problems)-----
        print('__SANDBOX_RESULT__')        
        print('__METRICS_START__')
        print(repr(metrics))
        print('__METRICS_END__')
        
        print('__FEATURES_START__')
        print(repr(features))
        print('__FEATURES_END__')
        
        print('__SCORE_START__')
        print(repr(score))
        print('__SCORE_END__')
        
        print('__SANDBOX_SUCCESS__')
        
    except Exception as e:
        print('__SANDBOX_ERROR__:')
        print(f'Error type: {type(e).__name__}')
        print(f'Error message: {str(e)}')
        print('Full traceback:')
        traceback.print_exc()