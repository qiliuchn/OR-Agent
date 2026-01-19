# Evaluation script for BPP-Offline-ACO problem
import os
import sys
import traceback
from math import floor
import argparse
from typing import NamedTuple, Tuple, List, Annotated, Dict, Any
import numpy as np
import numpy.typing as npt
import seed_solution as solution_module  # Note: solution module script is generated and saved on the fly


# =====Load function to evolve=====
problem = "bpp_offline_aco"
heuristics = getattr(solution_module, "heuristics")  # Get function to evolve


# =====Configuration and Parameters=====
IntArray = npt.NDArray[np.int_]
FloatArray = npt.NDArray[np.float64]

class BPPInstance(NamedTuple):
    n: int
    capacity: int
    demands: npt.NDArray[np.int_]

DEMAND_LOW = 20
DEMAND_HIGH = 100
CAPACITY = 150

dataset_conf = {
    'train': (500,),
    'val':   (120, 500),  # 120, 500, 1000
    'test':  (120, 500),  # 120, 500, 1000
}


# =====Utility Functions=====
def load_dataset(fp) -> list[BPPInstance]:
    data = np.load(fp)
    demands = data['demands']
    instances = []
    n = demands.shape[1]
    for demand in demands:
        instance = BPPInstance(n, CAPACITY, demand)
        instances.append(instance)
    return instances

def organize_path(path: IntArray) -> Tuple[int, IntArray]:
    order = {}
    result = np.zeros_like(path)
    for i, v in enumerate(path):
        if v in order:
            result[i] = order[v]
        else:
            result[i] = order[v] = len(order)
    return len(order), result

def calculate_path_cost_fitness(vacancies: IntArray, capacity: int) -> Tuple[int, float]:
    occupied = (capacity - vacancies[vacancies!=capacity]).astype(float)
    cost = len(occupied)
    result = ((occupied/capacity)**2).sum().item()/cost
    return cost, result

def calculate_path_fitness(vacancies: List[int], capacity: int) -> float:
    occupied = capacity - np.array(vacancies, dtype=float)
    result = ((occupied/capacity)**2).sum().item()/len(vacancies)
    return result

def greedy_sample(prob: FloatArray) -> int:
    return prob.argmax().item()

def random_sample(prob: FloatArray) -> int:
    # not used, `random_sample_discrete_distribution` is a faster implementation
    sampled = np.random.choice(prob.size, p=prob/prob.sum())
    return sampled

def random_sample_discrete_distribution(prob: FloatArray) -> int:
    # prob_exp = np.exp(prob-prob.max())
    # prob_exp[prob==0] = 0
    # np.random.choice is somehow slow
    cumprob = np.cumsum(prob)
    sampled = np.searchsorted(cumprob, next(uniform_generator)*cumprob[-1]).item()
    return sampled if sampled<len(cumprob) else len(cumprob)-1

def uniform_number_generator(batch_size = 500):
    # it's also slow to generate random numbers one by one
    while 1:
        numbers = np.random.random(batch_size)
        for n in numbers:
            yield n.item()

uniform_generator = uniform_number_generator()


# =====ACO class=====
class ACO(object):
    def __init__(self,
                 demand: IntArray,   # (n, )
                 heuristic: FloatArray,   # (n, n)
                 capacity: int,
                 n_ants=20, 
                 decay=0.95,
                 alpha=1,
                 beta=1,
                 greedy = False
                 ):
        
        self.problem_size = len(demand)
        self.capacity = capacity
        self.demand = demand
        assert self.demand.max() <= self.capacity
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        self.pheromone: FloatArray = np.ones((self.problem_size, self.problem_size)) # problem_size x self.problem_size
        heuristic[heuristic > 1e6] = 1e6
        heuristic[heuristic < 1e-6] = 1e-6
        heuristic = heuristic/heuristic.max() # normalize
        heuristic[heuristic < 1e-6] = 1e-6
        self.heuristic: FloatArray = heuristic # problem_size x self.problem_size

        self.shortest_path: IntArray = np.arange(self.problem_size)
        self.best_cost = self.problem_size

        self._ordinal: IntArray = np.arange(self.problem_size, dtype=int) # for indexing
        self.greedy_mode = greedy
    
    def run(self, iterations: int) -> Tuple[int, IntArray]:
        for _ in range(iterations):
            prob = self.pheromone**self.alpha * self.heuristic**self.beta
            paths, costs, fitnesses = self.gen_paths(self.n_ants, prob)
            best_index = costs.argmin()
            best_cost = costs[best_index].item()
            if best_cost < self.best_cost:
                self.shortest_path = paths[best_index]
                self.best_cost = best_cost
            self.update_pheronome(paths, fitnesses)
        assert self.is_valid_path(self.shortest_path)
        # cost, path = organize_path(self.shortest_path)
        # assert cost >= np.ceil(np.sum(self.demand).astype(float)/self.capacity).item()
        return organize_path(self.shortest_path)

    def sample_only(self, count: int) -> Tuple[int, IntArray]:
        self.greedy_mode = True
        paths, costs, _ = self.gen_paths(count, self.heuristic)
        best_index = costs.argmin()
        best_path = paths[best_index]
        assert self.is_valid_path(best_path)
        return organize_path(best_path)

    def update_pheronome(self, paths: List[IntArray], fitnesses: FloatArray):
        delta_phe = np.zeros_like(self.pheromone) # problem_size x problem_size
        for path, f in zip(paths, fitnesses):
            delta_phe[path[:, None]==path[None, :]] += f / self.n_ants
        self.pheromone *= self.decay
        self.pheromone += delta_phe

    def gen_paths(self, count: int, prob: FloatArray) -> Tuple[List[IntArray], IntArray, FloatArray]:
        paths, costs, fitnesses = [], [], []
        for _ in range(count):
            path, cost, fitness = self.sample_path(prob)
            paths.append(path)
            costs.append(cost)
            fitnesses.append(fitness)
        return paths, np.array(costs, dtype=int), np.array(fitnesses, dtype=float)
    
    def sample_path(self, prob: FloatArray
                    ) -> Tuple[
                        Annotated[IntArray, "sampled path"], 
                        Annotated[int, "used bins"], 
                        Annotated[float, "fitness"]]:
        
        if self.greedy_mode:
            sample_func = greedy_sample
        else:
            sample_func = random_sample_discrete_distribution
    
        path = np.ones(self.problem_size, dtype=int)*-1 # x=path[i] => put item i in bin x
        valid_items = np.ones(self.problem_size, dtype=bool)
        current_bin = item_count = 0
        vacancies = []
        bin_vacancy = self.capacity
        bin_items = np.zeros_like(valid_items)

        for _ in range(self.problem_size):
            mask = np.bitwise_and(self.demand <= bin_vacancy, valid_items)
            if not np.any(mask): # no valid item
                # move to the next bin
                vacancies.append(bin_vacancy)
                bin_vacancy, item_count = self.capacity, 0
                current_bin += 1
                bin_items[:] = False
                # uniformly select one
                selected = self.random_select(valid_items)
            else:
                if item_count == 0:
                    selected = self.random_select(mask)
                else:
                    item_prob = (prob[bin_items].sum(0)/item_count+1e-5) * mask
                    selected = sample_func(item_prob)
            
            # put item in this bin
            bin_items[selected] = True
            bin_vacancy -= self.demand[selected]
            valid_items[selected] = False
            path[selected] = current_bin
            item_count += 1
        
        vacancies.append(bin_vacancy)
        fitness = calculate_path_fitness(vacancies, self.capacity)
        return path, len(vacancies), fitness
    
    def random_select(self, mask: npt.NDArray[np.bool_]) -> int:
        valid = self._ordinal[mask]
        return valid[floor(next(uniform_generator)*len(valid))].item()
        # return valid[np.random.randint(0, len(valid))].item()
    
    def is_valid_path(self, path: IntArray) -> bool:
        # not used
        if path.shape[0] != self.problem_size:
            return False
        bins, path = organize_path(path)
        occupied = np.zeros(bins, dtype=int)
        for i, v in enumerate(path):
            if v<0:
                return False
            occupied[v] += self.demand[i]
            if occupied[v] > self.capacity:
                return False
        return True


# =====Evaluation function=====
N_ITERATIONS = 15
N_ANTS = 20
SAMPLE_COUNT = 200

def evaluate_heuristic(inst: BPPInstance, mode = 'sample'):
    heu = heuristics(inst.demands.copy(), inst.capacity) # normalized in ACO
    assert tuple(heu.shape) == (inst.n, inst.n)
    assert 0 < heu.max() < np.inf
    aco = ACO(inst.demands, heu.astype(float), capacity = inst.capacity, n_ants=N_ANTS, greedy=False)
    if mode == 'sample':
        obj, _ = aco.sample_only(SAMPLE_COUNT)
    else:
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
    method = 'aco'
    # Print parsed arguments for verification
    print(f"root_dir: {root_dir}")
    print(f"file_output_prefix: {file_output_prefix}")
    print(f"mode: {mode}")
    #print(f"problem_size: {problem_size}")
    
    # -----Run the evaluation-----
    # Run two instances: 120, 500; execution time: 125s
    try:
        basepath = os.path.join(root_dir, "problems", problem)
        
        if not os.path.isfile(os.path.join(basepath, f"dataset/train{dataset_conf['train'][0]}_dataset.npz")):
            raise ValueError("Dataset does not exist. Please generate it first.")
        
        if mode == 'train':
            dataset_path = os.path.join(basepath, f"dataset/{mode}{problem_size}_dataset.npz")
            dataset = load_dataset(dataset_path)
            n_instances = len(dataset)

            print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
            
            objs = []
            for i, instance in enumerate(dataset):
                obj = evaluate_heuristic(instance, mode=method)
                print(f"[*] Instance {i}: {obj}")
                objs.append(obj)
            
            print("[*] Average:")
            print(np.mean(objs))

        else: # mood == 'val'
            metrics = {}
            for problem_size in dataset_conf['val']:
                dataset_path = os.path.join(basepath, f"dataset/{mode}{problem_size}_dataset.npz")
                dataset = load_dataset(dataset_path)
                n_instances = dataset[0].n
                print(f"[*] Evaluating {dataset_path}")

                objs = []
                for i, instance in enumerate(dataset):
                    obj = evaluate_heuristic(instance, mode=method)
                    objs.append(obj)
                
                print(f"[*] Average for problem size {problem_size}: {np.mean(objs)}")
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