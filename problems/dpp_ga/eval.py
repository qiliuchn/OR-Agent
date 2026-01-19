# Evaluation script for DPP-GA problem
""" 
The genetic algorithm solves the Decap Placement Problem by evolving populations of candidate solutions (decap placements) over generations.
Each individual is a vector of n_decap positions (e.g., 20 decaps)
Positions are indices on an nxm grid (10x10 PDN)
Constraints: positions ≠ probe location and not in prohibited zones
The `crossover` function implements crossover between two parents.
"""
import os
import sys
import traceback
import time
import argparse
from typing import Dict, List, Tuple, Any
import random
import numpy as np
from tqdm import tqdm
from numpy.linalg import inv
import seed_solution as solution_module  # Note: solution module script is generated and saved on the fly

seed = 5678
random.seed("%d" % (seed))


# =====Load function to evolve=====
problem = "dpp_ga"
crossover = getattr(solution_module, "crossover")  # Get function to evolve


# =====Utility functions=====
def decap_placement(n, m, raw_pdn, pi, probing_port, freq_pts, decap):
    num_decap = np.size(pi)
    probe = probing_port
    z1 = raw_pdn
    z2 = np.zeros((freq_pts, num_decap, num_decap))

    qIndx = []
    for i in range(num_decap):
        z2[:, i, i] = np.abs(decap)
        qIndx.append(i)
    pIndx = pi.astype(int)
    # pIndx : index of ports in z1 for connecting
    # qIndx : index of ports in z2 for connecting

    aIndx = np.arange(len(z1[0]))
    aIndx = np.delete(aIndx, pIndx)

    z1aa = z1[:, aIndx, :][:, :, aIndx]
    z1ap = z1[:, aIndx, :][:, :, pIndx]
    z1pa = z1[:, pIndx, :][:, :, aIndx]
    z1pp = z1[:, pIndx, :][:, :, pIndx]
    z2qq = z2[:, qIndx, :][:, :, qIndx]

    zout = z1aa - np.matmul(np.matmul(z1ap, inv(z1pp + z2qq)), z1pa)

    for i in range(n * m):
        if i in pi:

            if i < probing_port:
                probe = probe - 1

    probe = int(probe)
    zout = zout[:, probe, probe]
    return zout

def initial_impedance(n, m, raw_pdn, probe):
    probe = int(probe)
    zout = raw_pdn[:, probe, probe]
    return zout


# =====Reward function=====
def model_1(freq_pts, z_initial, z_final, freq):
    impedance_gap = np.zeros(freq_pts)

    freq_point = 2e9
    min = 0.32
    grad = 0.16
    target_impedance = np.zeros(np.shape(freq))
    idx0 = np.argwhere(freq < freq_point)
    idx1 = np.argwhere(freq >= freq_point)
    target_impedance[idx0] = min
    target_impedance[idx1] = grad * 1e-9 * freq[idx1]

    penalty = 1
    reward = 0

    for i in range(freq_pts):
        if z_final[i] > target_impedance[i]:
            impedance_gap[i] = (z_final[i] - target_impedance[i]) * penalty
        else:
            impedance_gap[i] = 0
        # impedance_gap[i]=target_impedance[i]-z_final[i]

        reward = reward - (impedance_gap[i] / (434 * penalty))
    return reward

def model_2(freq_pts, z_initial, z_final, freq):
    impedance_gap = np.zeros(freq_pts)

    reward = 0

    for i in range(freq_pts):
        impedance_gap[i] = z_initial[i] - z_final[i]
        reward = reward + impedance_gap[i]
    reward = reward / 10
    return reward

def model_3(freq_pts, z_initial, z_final, freq):
    impedance_gap = np.zeros(freq_pts)

    freq_point = 2e9
    reward = 0

    for i in range(freq_pts):
        impedance_gap[i] = z_initial[i] - z_final[i]

        if freq[i] < freq_point:
            reward = reward + (impedance_gap[i] * 1.5)

        else:
            reward = reward + impedance_gap[i]
    reward = reward / 10
    return reward

def model_4(freq_pts, z_initial, z_final, freq):
    impedance_gap = np.zeros(freq_pts)

    freq_point = 2e9
    reward = 0

    for i in range(freq_pts):
        impedance_gap[i] = z_initial[i] - z_final[i]

        if freq[i] < freq_point:
            if impedance_gap[i] > 0:
                reward = reward + (impedance_gap[i] * 1.5)
            else:
                reward = reward + (impedance_gap[i] * 3)
        else:
            if impedance_gap[i] > 0:
                reward = reward + impedance_gap[i]
            else:
                reward = reward + (impedance_gap[i] * 3)
    reward = reward / 10
    return reward

def model_5(freq_pts, z_initial, z_final, freq):
    impedance_gap = np.zeros(freq_pts)

    # vectorized version
    impedance_gap = z_initial - z_final
    reward = np.sum(impedance_gap * 1000000000 / freq) / 10
    return reward

class RewardModel:
    def __init__(self, 
                    basepath,
                    model_number=5,
                    freq_pts = 201,
                    n=10,
                    m=10,
                    freq_data_path="DPP_data/freq_201.npy",
                    raw_pdn_path="DPP_data/10x10_pkg_chip.npy"):
        self.model_number = model_number
        self.freq_pts = freq_pts
        self.n = n
        self.m = m
        self.basepath = basepath
        
        freq_data_path = os.path.join(basepath, freq_data_path)
        raw_pdn_path = os.path.join(basepath, raw_pdn_path)
        self.freq = self.load_data(freq_data_path)
        self.raw_pdn = self.load_data(raw_pdn_path)
        
        decap_path = os.path.join(basepath, "DPP_data/01nF_decap.npy")
        with open(decap_path, "rb") as f:
            self.decap = np.load(f).reshape(-1)

        # get reward model based on model number       
        class_name = "model_" + str(model_number) # e.g. get model_5 as function
        self.model = globals()[class_name]
        
    def load_data(self, path):
        with open(path, "rb") as f:
            return np.load(f)
        
    def __call__(self, probe, pi):
        z_initial = initial_impedance(self.n, self.m, self.raw_pdn, probe)
        z_initial = np.abs(z_initial)

        pi = pi.astype(int)

        z_final = decap_placement(self.n, self.m, self.raw_pdn, pi, probe, self.freq_pts, self.decap)
        z_final = np.abs(z_final)
        
        return self.model(self.freq_pts, z_initial, z_final, self.freq)


# =====GA algorithm=====
def mutation(population: np.ndarray, probe: int, prohibit: np.ndarray, size: int=100) -> np.ndarray:
    """Seed mutation, if not considering the validation step.
    We separate the mutation step from the validation step in the DevFormer implementation.
    """
    return population

def validate(population: np.ndarray, probe: int, prohibit: np.ndarray, size: int=100) -> np.ndarray:
    """Seed mutation;
    Mutation while validating the population.
    
    Args:
        population (np.ndarray): Population of individuals; shape: (P, n_decap).
        probe (int): Probe value; each element in the population should not be equal to this value. 
        prohibit (np.ndarray): Prohibit values; each element in the population should not be in this set.
        size (int): Size of the PDN; each element in the population should be in the range [0, size).
    """
    n_pop, n_decap = population.shape
    for i in range(n_pop):
        ind = population[i]
        unique_actions = np.unique(population[i])
        if len(unique_actions) < n_decap:
            # Find the indices wherein the action is taken the second time
            dup_idx = []
            action_set = set()
            for j, action in enumerate(ind):
                if action in action_set:
                    dup_idx.append(j)
                action_set.add(action)

            # Mutate the duplicated actions
            infeasible_actions = np.concatenate([prohibit, [probe], unique_actions])
            feasible_actions = np.setdiff1d(np.arange(size), infeasible_actions)
            assert n_decap - len(unique_actions) == len(dup_idx)
            new_actions = np.random.choice(feasible_actions, len(dup_idx), replace=False)
            ind[dup_idx] = new_actions
    return population

def reevo_crossover(parents: np.ndarray, n_pop: int) -> np.ndarray:
    """Crossover generated by ReEvo."""
    n_parents, n_decap = parents.shape

    parents_idx = np.random.choice(n_parents, (n_pop, 2))
    crossover_points = np.random.randint(1, n_decap, n_pop)

    mask = np.tile(np.arange(n_decap), (n_pop, 1))
    offspring = np.where(mask < crossover_points.reshape(-1, 1), 
                         parents[parents_idx[:, 0], :], 
                         parents[parents_idx[:, 1], :])

    return offspring

def reevo_mutation(population: np.ndarray, probe: int, prohibit: np.ndarray, size: int = 100) -> np.ndarray:
    """Mutation generated by ReEvo."""
    p, n_decap = population.shape

    is_not_probe = np.all(population != probe, axis=1)
    is_not_prohibited = np.all(np.isin(population, prohibit, invert=True), axis=1)
    is_feasible = is_not_probe & is_not_prohibited

    mutation_mask = np.random.rand(p, n_decap) < 0.1
    mutation_values = np.random.randint(0, size, size=(p, n_decap))

    mutated_population = np.where(mutation_mask & is_feasible[:, None], mutation_values, population) # If mutate and feasible, then mutate

    return mutated_population

def generate_population(population_size: int, n_decap: int, probe: int, prohibit: np.ndarray, n: int, m: int) -> np.ndarray:
    # Create the full range of actions, excluding 'probe' and any 'prohibit' values
    possible_actions = np.setdiff1d(np.arange(n * m), np.append(prohibit, probe))
   # Ensure that the possible actions can fill the required number of decaps
    if len(possible_actions) < n_decap:
        raise ValueError("Not enough valid actions to fill the individuals without replacement.")
    # Randomly select 'n_decap' unique actions from the possible actions
    pop = np.stack([np.random.choice(possible_actions, n_decap, replace=False) for _ in range(population_size)])
    return pop

def check_feasibility(population: np.ndarray, probe: int, prohibit: np.ndarray) -> None:
    """Check if the population is feasible."""
    n_pop, n_decap = population.shape
    for i in range(n_pop):
        unique_actions = np.unique(population[i])
        if len(unique_actions) < n_decap:
            raise ValueError("Population is infeasible.")
        for action in population[i]:
            if action in prohibit or action == probe:
                raise ValueError("Population is infeasible.")

def eval_population(population, probe, reward_model) -> np.ndarray:
    rewards = [
        reward_model(probe, pi)
        for pi in population
        ]
    return np.array(rewards)

def selection(population: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    """Return selected part of the population.
    Args:
        population (np.ndarray): Population of individuals; shape: (P, n_decap); already sorted according to the rewards in ascending order.
        rewards (np.ndarray): Reward values of the individuals; shape: (P,); already sorted in ascending order.
    """
    better_half = population[int(len(population) / 2):]
    return better_half


# =====Evaluation function=====
def evaluate_heuristic(n_pop: int, n_iter: int, n_inst: int, elite_rate: float, n_decap: int, reward_model: RewardModel) -> float:
    """
    Runs the Genetic Algorithm (GA) for optimization.

    Args:
        n_pop (int): Population size.
        n_iter (int): Number of generations.
        n_inst (int): Number of test instances.
        elite_rate (float): Percentage of elite individuals.
        n_decap (int): Number of decap.
        reward_model (RewardModel): Reward model for scoring the individuals.
    """
    sum_reward = 0

    # Outer loop: test instances
    metrics = {}
    for j in tqdm(range(n_inst), desc="Testing {} instances".format(n_inst), disable=True):
        start_time = time.time()

        probe = int(test_probe[j])
        prohibit = test_prohibit[j]
        keep_num = int(keepout_num[j])
        prohibit = prohibit[0: keep_num]

        population = generate_population(n_pop, n_decap, probe, prohibit, n, m)  # shape: (P, n x m)
        rewards = eval_population(population, probe, reward_model)  # shape: (P,)
        print(f"[Instance {j}] Initial population avg. reward:", rewards.mean())
        # Inner loop: generations
        for i in range(n_iter):
            # Sort the population and rewards according to the reward
            sorted_idx = rewards.argsort() # ascending order
            population = population[sorted_idx]
            rewards = rewards[sorted_idx]
            
            # Select the population for crossover
            selected_population = selection(population, rewards)

            # Preserve the elites
            n_elite = int(n_pop * elite_rate)
            elites = population[-n_elite:]

            # Crossover with the better half
            population_nxt = crossover(selected_population, n_pop=n_pop - n_elite)
            # Mutate the population
            population_nxt = mutation(population_nxt, probe, prohibit, n * m)
            # Validate the population
            population_nxt = validate(population_nxt, probe, prohibit, n * m)

            # Check the feasibility of the next generation
            # check_feasibility(population_nxt, probe, prohibit)
            
            # Evaluate the population
            rewards_nxt = eval_population(population_nxt, probe, reward_model)

            # Elitism
            # 1. Concate the elites
            population = np.concatenate([elites, population_nxt], axis=0)
            # 2. Concate the rewards
            rewards = np.concatenate([rewards[-n_elite:], rewards_nxt], axis=0)

            print("[Instance {:d}] Generation {:d} - Elite reward: {:.4f}".format(j, i, rewards[:n_elite].mean()) + " - Best reward: {:.4f}".format(rewards.max()))

        # Evaluate the final population
        best_idx = np.argmax(rewards)
        best_solution, best_reward = population[best_idx], rewards[best_idx]
        sum_reward += best_reward
        print(f"[Instance {j}] Best solution:", best_solution)
        print(f"[Instance {j}] Best reward:", best_reward)
        print(f"[Instance {j}] %s seconds" % (time.time() - start_time))
        metrics[j] = float(best_reward)
        
    # result = plot_result.plot(raw_pdn, probe, guide_action, n, m, j)
    print("Average reward:", sum_reward / n_inst)
    
    return metrics, sum_reward / n_inst


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
    # Run 5 instances; execution time: 50s
    try:
        basepath = os.path.join(root_dir, "problems", problem)

        # Parameters
        n = 10 # PDN shape
        m = 10 # PDN shape
        model = 5 # Reward model type
        freq_pts = 201 # Number of Frequencies

        # Paths
        test_probe_path = os.path.join(basepath, "test_problems", "test_100_probe.npy")
        test_prohibit_path = os.path.join(basepath, "test_problems", "test_100_keepout.npy")
        keepout_num_path = os.path.join(basepath, "test_problems", "test_100_keepout_num.npy")

        # Model initialization
        reward_model = RewardModel(basepath, n=n, m=m, model_number=model, freq_pts=freq_pts)

        # File reading
        with open(test_probe_path, "rb") as f:
            test_probe = np.load(f)  # shape (test,)

        with open(test_prohibit_path, "rb") as f1:
            test_prohibit = np.load(f1)  # shape (test, n_keepout)

        with open(keepout_num_path, "rb") as f2:
            keepout_num = np.load(f2)  # shape (test,)

        elite_rate = 0.2
        n_decap = 20
        n_pop = 20
        
        if mode == 'train':
            n_inst = 3
            n_iter = 5
            test_probe = test_probe[0: 3]
            test_prohibit = test_prohibit[0: 3]
            keepout_num = keepout_num[0: 3]
            metrics, avg_reward = evaluate_heuristic(n_pop, n_iter, n_inst, elite_rate, n_decap, reward_model)
            print("[*] Average:")
            print(avg_reward)
        elif mode == 'val':
            n_inst = 5
            n_iter = 10
            test_probe = test_probe[5: 10]
            test_prohibit = test_prohibit[5: 10]
            keepout_num = keepout_num[5: 10]
            metrics, avg_reward = evaluate_heuristic(n_pop, n_iter, n_inst, elite_rate, n_decap, reward_model)
            print("[*] Average:")
            print(avg_reward)
        else:
            assert mode == 'test'
            n_inst = 64
            n_iter = 10
            test_probe = test_probe[-64: ]
            test_prohibit = test_prohibit[-64: ]
            keepout_num = keepout_num[-64: ]
            avg_reward = evaluate_heuristic(n_pop, n_iter, n_inst, elite_rate, n_decap, reward_model)
            print("[*] Average:")
            print(avg_reward)
    
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