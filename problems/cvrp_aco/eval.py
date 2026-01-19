# Evaluation script for CVRP-ACO problem
import os
import sys
import traceback
import numpy as np
import argparse
from typing import Dict, List, Tuple, Any
import torch
from torch.distributions import Categorical
from scipy.spatial import distance_matrix
import inspect
import seed_solution as solution_module  # Note: solution module script is generated and saved on the fly


# =====Load function to evolve=====
problem = "cvrp_aco"
heuristics = getattr(solution_module, "heuristics")  # Get function to evolve


# =====ACO class=====
class ACO():
    def __init__(self,  # 0: depot
                 distances, # (n, n) distance matrix between all nodes
                 demand,   # (n, ) demand at each node (0 for depot)
                 heuristic, # (n, n) heuristic matrix guiding ant movement
                 capacity,  # vehicle capacity constraint
                 n_ants=30,  # number of ants in colony
                 decay=0.9,  # pheromone evaporation rate
                 alpha=1,    # pheromone importance factor
                 beta=1,     # heuristic importance factor
                 device='cpu',  # computation device
                 ):
        self.problem_size = len(distances)  # number of nodes including depot
        self.distances = torch.tensor(distances, device=device) if not isinstance(distances, torch.Tensor) else distances
        self.demand = torch.tensor(demand, device=device) if not isinstance(demand, torch.Tensor) else demand
        self.capacity = capacity

        self.n_ants = n_ants
        self.decay = decay  # pheromone evaporation: τ = τ * decay
        self.alpha = alpha  # controls pheromone influence: τ^α
        self.beta = beta    # controls heuristic influence: η^β

        self.pheromone = torch.ones_like(self.distances)  # initial pheromone matrix
        self.heuristic = torch.tensor(heuristic, device=device) if not isinstance(heuristic, torch.Tensor) else heuristic

        self.shortest_path = None  # best solution found
        self.lowest_cost = float('inf')  # cost of best solution

        self.device = device
        
    @torch.no_grad()
    def run(self, n_iterations):
        """Main ACO loop: run for n_iterations"""
        for _ in range(n_iterations):
            paths = self.gen_path()  # generate paths for all ants
            costs = self.gen_path_costs(paths)  # compute total distance for each ant

            best_cost, best_idx = costs.min(dim=0)  # find best ant in this iteration
            if best_cost < self.lowest_cost:  # update global best if improved
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost

            self.update_pheronome(paths, costs)  # update pheromone trails

        return self.lowest_cost  # return best cost found
       
    @torch.no_grad()
    def update_pheronome(self, paths, costs):
        '''
        Update pheromone trails using ant solutions.
        Pheromone update rule: τ_ij = τ_ij * decay + Σ(Δτ_ij^k) where Δτ_ij^k = Q/L_k

        Args:
            paths: torch tensor with shape (problem_size, n_ants) - complete paths for all ants
            costs: torch tensor with shape (n_ants,) - total distance for each ant
        '''
        self.pheromone = self.pheromone * self.decay  # evaporation: τ = τ * ρ
        for i in range(self.n_ants):
            path = paths[:, i]  # path for ant i
            cost = costs[i]     # total distance for ant i
            # Add pheromone to edges used by this ant: Δτ = Q/L (Q=1 here)
            # path[:-1] gives current nodes, torch.roll(path, shifts=-1)[:-1] gives next nodes
            self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += 1.0/cost
        self.pheromone[self.pheromone < 1e-10] = 1e-10  # prevent pheromone from going to zero
    
    @torch.no_grad()
    def gen_path_costs(self, paths):
        """Compute total distance for each ant's path"""
        u = paths.permute(1, 0) # shape: (n_ants, max_seq_len) - transpose for easier indexing
        v = torch.roll(u, shifts=-1, dims=1)  # shift to get next node in sequence
        # Sum distances between consecutive nodes (excluding last to first wrap-around)
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def gen_path(self):
        """Generate complete paths for all ants using constructive heuristic"""
        actions = torch.zeros((self.n_ants,), dtype=torch.long, device=self.device)  # all ants start at depot (node 0)
        visit_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)  # 1=unvisited, 0=visited
        visit_mask = self.update_visit_mask(visit_mask, actions)  # mark depot as visited
        used_capacity = torch.zeros(size=(self.n_ants,), device=self.device)  # current load for each ant

        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)  # update capacity constraints

        paths_list = [actions]  # paths_list[i] contains the ith move for all ants

        done = self.check_done(visit_mask, actions)
        while not done:
            actions = self.pick_move(actions, visit_mask, capacity_mask)  # probabilistic node selection
            paths_list.append(actions)  # record move
            visit_mask = self.update_visit_mask(visit_mask, actions)  # update visited nodes
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)  # update capacity
            done = self.check_done(visit_mask, actions)  # check termination

        return torch.stack(paths_list)  # shape: (seq_len, n_ants)
        
    def pick_move(self, prev, visit_mask, capacity_mask):
        """Probabilistic node selection using transition probability: p_ij ∝ τ_ij^α * η_ij^β"""
        pheromone = self.pheromone[prev]  # shape: (n_ants, p_size) - pheromone on edges from current nodes
        heuristic = self.heuristic[prev]  # shape: (n_ants, p_size) - heuristic values from current nodes
        # Transition probability: p_ij = (τ_ij^α * η_ij^β) / Σ(τ_ik^α * η_ik^β)
        # Masked by visit_mask (unvisited nodes) and capacity_mask (feasible nodes)
        dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * visit_mask * capacity_mask)  # shape: (n_ants, p_size)
        dist = Categorical(dist)  # create categorical distribution
        actions = dist.sample()  # shape: (n_ants,) - sample next node for each ant
        return actions
    
    def update_visit_mask(self, visit_mask, actions):
        """Update mask of unvisited nodes after moving to new nodes"""
        visit_mask[torch.arange(self.n_ants, device=self.device), actions] = 0  # mark new nodes as visited
        visit_mask[:, 0] = 1  # depot can always be revisited (for returning/starting new route)
        # Exception: if ant returns to depot AND still has unvisited customers, don't allow immediate return
        # This prevents depot-depot cycles when work remains
        visit_mask[(actions==0) * (visit_mask[:, 1:]!=0).any(dim=1), 0] = 0
        return visit_mask
    
    def update_capacity_mask(self, cur_nodes, used_capacity):
        '''
        Update vehicle capacity constraints and create mask of feasible next nodes.

        Args:
            cur_nodes: shape (n_ants, ) - current node for each ant
            used_capacity: shape (n_ants, ) - current load for each ant

        Returns:
            used_capacity: updated capacity after visiting cur_nodes
            capacity_mask: mask where 1=feasible (demand ≤ remaining capacity), 0=infeasible
        '''
        capacity_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        # update capacity: reset to 0 when returning to depot, add demand of current node
        used_capacity[cur_nodes==0] = 0  # reset load when returning to depot
        used_capacity = used_capacity + self.demand[cur_nodes]  # add demand of current node

        # update capacity_mask: mask out nodes whose demand exceeds remaining capacity
        remaining_capacity = self.capacity - used_capacity  # (n_ants,) - remaining capacity for each ant
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.problem_size)  # (n_ants, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(self.n_ants, 1)  # (n_ants, p_size) - demand of all nodes
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0  # mask infeasible nodes

        return used_capacity, capacity_mask
    
    def check_done(self, visit_mask, actions):
        """Check termination condition: all customers visited and all ants at depot"""
        # All customers (nodes 1..n) visited AND all ants currently at depot (node 0)
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()  


# =====Evaluation function=====
N_ITERATIONS = 50  # number of ACO iterations
N_ANTS = 30         # number of ants in colony
CAPACITY = 50       # vehicle capacity

def evaluate_heuristic(node_pos, demand):
    """Evaluate a heuristic function using ACO on a CVRP instance"""
    # Compute distance matrix between all nodes
    dist_mat = distance_matrix(node_pos, node_pos)
    dist_mat[np.diag_indices_from(dist_mat)] = 1  # set diagonal to 1 (avoid division by zero in heuristics)

    # Call the heuristic function (evolved code) with appropriate arguments
    # The heuristic function can have different signatures (2 or 4 args)
    if len(inspect.getfullargspec(heuristics).args) == 4:
        # Signature: heuristics(dist_mat, node_pos, demand, capacity)
        heu = heuristics(dist_mat.copy(), node_pos.copy(), demand.copy(), CAPACITY) + 1e-9
    elif len(inspect.getfullargspec(heuristics).args) == 2:
        # Signature: heuristics(dist_mat, normalized_demand)
        heu = heuristics(dist_mat.copy(), demand / CAPACITY) + 1e-9

    heu[heu < 1e-9] = 1e-9  # ensure heuristic values are positive

    # Run ACO with the computed heuristic matrix
    aco = ACO(dist_mat, demand, heu, CAPACITY, n_ants=N_ANTS)
    obj = aco.run(N_ITERATIONS)  # get best solution cost
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
    # Run instances: 20, 50, 100; execution time: 170s
    try:
        basepath = os.path.join(root_dir, "problems", problem)
        if not os.path.isfile(os.path.join(basepath, "dataset/train50_dataset.npy")):
            raise FileNotFoundError("[!] Dataset not found.")
        
        if mode == 'train':
            dataset_path = os.path.join(basepath, f"dataset/{mode}{problem_size}_dataset.npy")
            dataset = np.load(dataset_path)
            demands, node_positions = dataset[:, :, 0], dataset[:, :, 1:]
            
            n_instances = node_positions.shape[0]
            print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
            
            objs = []
            for i, (node_pos, demand) in enumerate(zip(node_positions, demands)):
                obj = evaluate_heuristic(node_pos, demand)
                print(f"[*] Instance {i}: {obj}")
                objs.append(obj.item())
            
            print("[*] Average:")
            print(np.mean(objs))
        else:  # mode: "val"
            metrics = {}
            for problem_size in [20, 50]:  # options: 20, 50, 100
                dataset_path = os.path.join(basepath, f"dataset/{mode}{problem_size}_dataset.npy")
                dataset = np.load(dataset_path)
                demands, node_positions = dataset[:, :, 0], dataset[:, :, 1:]
                
                n_instances = node_positions.shape[0]
                print(f"[*] Evaluating {dataset_path}")
                
                objs = []
                for i, (node_pos, demand) in enumerate(zip(node_positions, demands)):
                    obj = evaluate_heuristic(node_pos, demand)
                    objs.append(obj.item())
                
                print(f"[*] Average for {problem_size}: {np.mean(objs)}")
                metrics[problem_size] = float(np.mean(objs))
                
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