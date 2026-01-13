# Evaluation script for TSP-POMO problem.
""" 
POMO (Policy Optimization with Multiple Optima) is a reinforcement learning approach for solving combinatorial optimization problems like the Traveling Salesman Problem (TSP). 
The key ideas: 
- Multiple Starting Points: Instead of training a single policy from one starting city, POMO trains from multiple starting cities simultaneously (called POMO size). 
    This helps the model explore different solution paths and find better optima. 
- Shared Policy Network: All starting points share the same neural network parameters, making training more efficient. 
- Symmetry Exploitation: For TSP, the optimal tour length is independent of the starting city. By training from multiple starting points, the model learns to find good solutions regardless of starting position. 
- Inference: During testing, the model generates multiple solutions from different starting points and selects the best one.

The `heuristics` function is used to generate attention bias in the model's attention mechanism.
- Encoder Attention: Self-attention between all city embeddings
- Decoder Attention: Self-attention between each city embedding and the previous city embeddings
Key insight: Attention in POMO allows the model to dynamically focus on different cities at each step based on the current tour state, while maintaining awareness of all cities through the encoder's global context. 
The multi-head mechanism enables learning different types of relationships (spatial, sequential, etc.) simultaneously.
"""
import os
import sys
import traceback
import time
from datetime import datetime
import logging
from typing import Dict, Tuple, List, Any
import pytz
import numpy as np
import json
import argparse
import shutil
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import seed_solution as solution_module  # Note: solution module script is generated and saved on the fly


# =====Load function to evolve=====
problem = "tsp_pomo"
heuristics = getattr(solution_module, "heuristics")  # Get function to evolve


# =====Configuration and Parameters=====
# Machine Environment Config
DEBUG_MODE = False  # Debug mode flag
USE_CUDA = False    # Whether to use GPU acceleration
CUDA_DEVICE_NUM = 0 # GPU device number

# Dataset configuration: problem sizes for training and validation
dataset_conf = {
    'train': (200, 500, 1000),  # Training problem sizes
    'val':   (200, 500, 1000),  # Validation problem sizes
}

# Environment parameters for TSP problem
env_params = {
    'problem_size': 100,  # Number of cities in TSP instance
    'pomo_size': 1,       # POMO size: number of parallel starting points (Policy Optimization with Multiple Optima)
}

# Neural network model parameters
model_params = {
    'embedding_dim': 128,           # Dimension of node embeddings
    'sqrt_embedding_dim': 128**(1/2), # Square root of embedding dim for scaling
    'encoder_layer_num': 6,         # Number of encoder layers
    'qkv_dim': 16,                  # Dimension of query/key/value vectors in attention
    'head_num': 8,                  # Number of attention heads
    'logit_clipping': 10,           # Clipping value for logits to prevent overflow
    'ff_hidden_dim': 512,           # Hidden dimension in feed-forward network
    'eval_type': 'argmax',          # Evaluation type: 'argmax' for greedy, 'softmax' for sampling
}

# Tester parameters for model evaluation
tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './checkpoints',  # Directory path of pre-trained model and log files
        'epoch': 3100,            # Epoch version of pre-trained model to load
    },
    'test_episodes': 10,          # Number of test episodes to run
    'test_batch_size': 10,        # Batch size for testing
    'augmentation_enable': False, # Whether to enable data augmentation
    'aug_factor': 8,              # Augmentation factor (8-fold symmetry for TSP)
    'aug_batch_size': 100,        # Batch size when augmentation is enabled
}
# Adjust batch size if augmentation is enabled
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

# Logger parameters for saving results
logger_params = {
    'log_file': {
        'desc': 'test__tsp100_longTrain',  # Description for log file
        'filename': 'log.txt'              # Log file name
    }
}


# =====Utility function=====
# Create timestamp for result folder naming
process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))
result_folder = './result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'

def get_result_folder():
    """Get the current result folder path."""
    return result_folder

def set_result_folder(folder):
    """Set the result folder path globally."""
    global result_folder
    result_folder = folder

def create_logger(log_file=None):
    """
    Create and configure a logger for the experiment.

    Args:
        log_file: Dictionary containing logger configuration with keys:
            - filepath: Path to log file directory
            - desc: Description to append to folder name
            - filename: Log file name (default: 'log.txt')
    """
    if 'filepath' not in log_file:
        log_file['filepath'] = get_result_folder()

    # Format the filepath with description if provided
    if 'desc' in log_file:
        log_file['filepath'] = log_file['filepath'].format(desc='_' + log_file['desc'])
    else:
        log_file['filepath'] = log_file['filepath'].format(desc='')

    set_result_folder(log_file['filepath'])

    # Determine full filename
    if 'filename' in log_file:
        filename = log_file['filepath'] + '/' + log_file['filename']
    else:
        filename = log_file['filepath'] + '/' + 'log.txt'

    # Create directory if it doesn't exist
    if not os.path.exists(log_file['filepath']):
        os.makedirs(log_file['filepath'])

    # Determine file mode (append if exists, write if new)
    file_mode = 'a' if os.path.isfile(filename)  else 'w'

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    # Remove existing handlers to avoid duplicate logging
    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # Write to file handler
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # Write to console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

class AverageMeter:
    """Utility class to compute and store running average of values."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the meter to initial state."""
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter with new value(s).

        Args:
            val: Value to add
            n: Number of samples represented by val (default: 1)
        """
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        """Compute and return the current average."""
        return self.sum / self.count if self.count else 0

class LogData:
    """
    Data structure for storing and managing logged metrics during training/evaluation.
    Stores data as key-value pairs where values are lists of (x, y) coordinates.
    """

    def __init__(self):
        self.keys = set()  # Set of all logged metric names
        self.data = {}     # Dictionary mapping metric names to data lists

    def get_raw_data(self):
        """Return raw internal data structure."""
        return self.keys, self.data

    def set_raw_data(self, r_data):
        """Set raw internal data structure."""
        self.keys, self.data = r_data

    def append_all(self, key, *args):
        """
        Append multiple data points for a key at once.

        Args:
            key: Metric name
            *args: Either:
                - Single list of y-values (x-values auto-generated as indices)
                - Two lists: x-values and y-values
        """
        if len(args) == 1:
            value = [list(range(len(args[0]))), args[0]]
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].extend(value)
        else:
            self.data[key] = np.stack(value, axis=1).tolist()
            self.keys.add(key)

    def append(self, key, *args):
        """
        Append a single data point for a key.

        Args:
            key: Metric name
            *args: Either:
                - Single value (int/float): y-value, x-value auto-incremented
                - Single tuple/list: (x, y) pair
                - Two values: x and y separately
        """
        if len(args) == 1:
            args = args[0]

            if isinstance(args, int) or isinstance(args, float):
                if self.has_key(key):
                    value = [len(self.data[key]), args]
                else:
                    value = [0, args]
            elif type(args) == tuple:
                value = list(args)
            elif type(args) == list:
                value = args
            else:
                raise ValueError('Unsupported value type')
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].append(value)
        else:
            self.data[key] = [value]
            self.keys.add(key)

    def get_last(self, key):
        """Get the most recent data point for a key."""
        if not self.has_key(key):
            return None
        return self.data[key][-1]

    def has_key(self, key):
        """Check if key exists in the log data."""
        return key in self.keys

    def get(self, key):
        """Get all y-values for a key as a list."""
        split = np.hsplit(np.array(self.data[key]), 2)
        return split[1].squeeze().tolist()

    def getXY(self, key, start_idx=0):
        """
        Get x and y values separately for a key.

        Args:
            key: Metric name
            start_idx: Starting index for x-values (default: 0)

        Returns:
            Tuple of (x_values, y_values)
        """
        split = np.hsplit(np.array(self.data[key]), 2)

        xs = split[0].squeeze().tolist()
        ys = split[1].squeeze().tolist()

        if type(xs) is not list:
            return xs, ys

        if start_idx == 0:
            return xs, ys
        elif start_idx in xs:
            idx = xs.index(start_idx)
            return xs[idx:], ys[idx:]
        else:
            raise KeyError('no start_idx value in X axis data.')

    def get_keys(self):
        """Get all metric names stored in the log."""
        return self.keys

class TimeEstimator:
    """
    Utility class for estimating remaining time based on progress.
    Useful for tracking training/evaluation progress.
    """

    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0  # Offset for count (useful when starting from non-zero)

    def reset(self, count=1):
        """
        Reset the timer.

        Args:
            count: Starting count value (default: 1)
        """
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        """
        Get elapsed and remaining time estimates.

        Args:
            count: Current progress count
            total: Total count to complete

        Returns:
            Tuple of (elapsed_time_hours, remaining_time_hours)
        """
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        # Estimate remaining time based on average time per item
        remain_time = elapsed_time * remain / (count - self.count_zero)

        # Convert to hours
        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        """
        Get formatted string representations of time estimates.

        Args:
            count: Current progress count
            total: Total count to complete

        Returns:
            Tuple of (elapsed_time_str, remaining_time_str)
        """
        elapsed_time, remain_time = self.get_est(count, total)

        # Format as hours if >1 hour, otherwise as minutes
        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        """
        Print time estimate to logger.

        Args:
            count: Current progress count
            total: Total count to complete
        """
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))

def util_print_log_array(logger, result_log: LogData):
    """
    Print all logged metrics to logger.

    Args:
        logger: Logger instance
        result_log: LogData object containing metrics
    """
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    for key in result_log.get_keys():
        logger.info('{} = {}'.format(key+'_list', result_log.get(key)))

def copy_all_src(dst_root):
    """
    Copy all source files used in the current execution to a destination directory.
    Useful for creating reproducible experiment snapshots.

    Args:
        dst_root: Root destination directory where 'src' folder will be created
    """
    # Determine execution directory (handles Jupyter notebook case)
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])

    # Determine home directory (project root)
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))

    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else:
        home_dir = tmp_dir1

    # Create target directory
    dst_path = os.path.join(dst_root, 'src')
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    # Copy all source files from loaded modules
    for item in sys.modules.items():
        key, value = item

        if hasattr(value, '__file__') and value.__file__:
            src_abspath = os.path.abspath(value.__file__)

            # Only copy files within the project directory
            if os.path.commonprefix([home_dir, src_abspath]) == home_dir:
                dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))

                # Handle duplicate filenames by adding index
                if os.path.exists(dst_filepath):
                    split = list(os.path.splitext(dst_filepath))
                    split.insert(1, '({})')
                    filepath = ''.join(split)
                    post_index = 0

                    while os.path.exists(filepath.format(post_index)):
                        post_index += 1

                    dst_filepath = filepath.format(post_index)

                shutil.copy(src_abspath, dst_filepath)

# =====TSPEnv class=====
def get_random_problems(batch_size, problem_size):
    """
    Generate random TSP instances with cities uniformly distributed in [0,1]².

    Args:
        batch_size: Number of TSP instances to generate
        problem_size: Number of cities in each instance

    Returns:
        Tensor of shape (batch_size, problem_size, 2) with city coordinates
    """
    problems = torch.rand(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)
    return problems

def augment_xy_data_by_8_fold(problems):
    """
    Apply 8-fold symmetry augmentation to TSP instances.
    Exploits symmetry in Euclidean TSP: reflections and coordinate swaps.

    Args:
        problems: Tensor of shape (batch, problem, 2) with city coordinates

    Returns:
        Augmented tensor of shape (8*batch, problem, 2)
    """
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    # 8 symmetry transformations:
    dat1 = torch.cat((x, y), dim=2)          # Original
    dat2 = torch.cat((1 - x, y), dim=2)      # Reflect x
    dat3 = torch.cat((x, 1 - y), dim=2)      # Reflect y
    dat4 = torch.cat((1 - x, 1 - y), dim=2)  # Reflect both
    dat5 = torch.cat((y, x), dim=2)          # Swap coordinates
    dat6 = torch.cat((1 - y, x), dim=2)      # Swap and reflect y
    dat7 = torch.cat((y, 1 - x), dim=2)      # Swap and reflect x
    dat8 = torch.cat((1 - y, 1 - x), dim=2)  # Swap and reflect both

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems

@dataclass
class Reset_State:
    """State container for environment reset."""
    problems: torch.Tensor
    # shape: (batch, problem, 2)

@dataclass
class Step_State:
    """State container for environment step."""
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)

class TSPEnv:
    """
    Environment for the Traveling Salesman Problem (TSP).
    Implements a sequential decision process where agent selects cities one by one.
    """

    def __init__(self, **env_params):
        # Constants initialized at environment creation
        self.env_params = env_params
        self.problem_size = env_params['problem_size']  # Number of cities
        self.pomo_size = env_params['pomo_size']        # POMO: number of parallel starting points
        self.test_file_path = env_params['test_file_path']  # Path to test dataset

        # Constants set when problems are loaded
        self.batch_size = None      # Number of TSP instances in batch
        self.BATCH_IDX = None       # Batch indices tensor, shape: (batch, pomo)
        self.POMO_IDX = None        # POMO indices tensor, shape: (batch, pomo)
        self.problems = None        # City coordinates, shape: (batch, problem, 2)

        # Dynamic state variables
        self.selected_count = None      # Number of cities selected so far
        self.current_node = None        # Current city for each instance, shape: (batch, pomo)
        self.selected_node_list = None  # Sequence of selected cities, shape: (batch, pomo, 0~problem)

    def load_problems(self, batch_size, aug_factor=1):
        """
        Load TSP problems for a batch.

        Args:
            batch_size: Number of TSP instances
            aug_factor: Augmentation factor (1 for no augmentation, 8 for 8-fold symmetry)
        """
        self.batch_size = batch_size
        if self.test_file_path is not None:
            self.problems = torch.load(self.test_file_path)
        else:
            self.problems = get_random_problems(batch_size, self.problem_size)
        # problems.shape: (batch, problem, 2)

        # Apply data augmentation if requested
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError

        # Create index tensors for batch and POMO dimensions
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        """
        Reset environment to initial state.

        Returns:
            Reset_State: Initial state with problem data
            reward: None (no reward at reset)
            done: False (episode not done)
        """
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        # Initialize mask with zeros (all cities available)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        """
        Prepare for next step without selecting a city.
        Used at the beginning of rollout.

        Returns:
            Step_State: Current state
            reward: None
            done: False
        """
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        """
        Execute one step: select a city and update environment state.

        Args:
            selected: Tensor of selected city indices, shape: (batch, pomo)

        Returns:
            Step_State: Updated state
            reward: Negative tour distance if episode done, else None
            done: Whether episode is complete
        """
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        # Append selected city to sequence
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        # Mask selected city to prevent re-selection (set to -inf for softmax)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # Check if episode is complete
        done = (self.selected_count == self.problem_size)
        if done:
            # Reward is negative tour distance (we want to minimize distance)
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        """
        Compute total travel distance for completed tours.

        Returns:
            Tensor of travel distances, shape: (batch, pomo)
        """
        # Gather city coordinates in the order they were selected
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        # Compute distances between consecutive cities
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)  # Shift by 1 for next city
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        # Sum distances to get total tour length
        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances


# =====TSPModel class=====
IMPL_REEVO = True  # Flag to enable/disable ReEvo heuristic integration
    
class TSPModel(nn.Module):
    """
    Neural network model for solving TSP using attention mechanism.
    Implements encoder-decoder architecture with POMO (multiple starting points).
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)      # Encodes city coordinates
        self.decoder = TSP_Decoder(**model_params)      # Decodes sequential decisions
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        """
        Pre-compute encodings and attention biases before rollout.

        Args:
            reset_state: Reset_State containing problem data
        """
        # reset_state.problems.shape: (batch, problem, 2)
        # Compute pairwise Euclidean distances between cities
        distance_matrices = torch.cdist(reset_state.problems, reset_state.problems, p=2)

        ######################## ReEvo Integration #############################
        if IMPL_REEVO:
            # Compute heuristic bias for each instance in batch
            self.attention_bias = torch.stack([
                heuristics(distance_matrices[i]) for i in range(distance_matrices.size(0))
            ], dim=0)
            # Sanity checks
            assert not torch.isnan(self.attention_bias).any()
            assert not torch.isinf(self.attention_bias).any()
        else:
            self.attention_bias = None
        #######################################################################

        # Encode city coordinates
        self.encoded_nodes = self.encoder(reset_state.problems)
        # shape: (batch, problem, EMBEDDING_DIM)

        # Set decoder's key-value cache from encoded nodes
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        """
        Forward pass: select next city based on current state.

        Args:
            state: Step_State containing current environment state

        Returns:
            selected: Tensor of selected city indices, shape: (batch, pomo)
            prob: Selection probabilities (only during training/softmax evaluation)
        """
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            # First step: select starting cities (POMO starting points)
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

            # Encode first selected nodes and set decoder's initial query
            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:
            # Subsequent steps: select next city based on current state
            # self.encoded_nodes.shape == (batch, problem, d_model)
            # state.current_node.shape == (batch, pomo)
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)

            # ReEvo: get attention bias for the current node
            attention_bias_current_node = self.attention_bias[torch.arange(batch_size)[:, None], state.current_node, :] if IMPL_REEVO else None
            # shape: (batch, pomo, problem)

            # Get probability distribution over next cities
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask, attention_bias_current_node=attention_bias_current_node)
            # shape: (batch, pomo, problem)

            if self.training or self.model_params['eval_type'] == 'softmax':
                # Training or sampling mode: sample from distribution
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    # Ensure we didn't sample a zero-probability action
                    if (prob != 0).all():
                        break

            else:
                # Greedy evaluation mode: select highest probability
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None

        return selected, prob

def _get_encoding(encoded_nodes, node_index_to_pick):
    """
    Extract embeddings for specific nodes from encoded node tensor.

    Args:
        encoded_nodes: Tensor of shape (batch, problem, embedding) with all node embeddings
        node_index_to_pick: Tensor of shape (batch, pomo) with indices of nodes to extract

    Returns:
        Tensor of shape (batch, pomo, embedding) with embeddings of selected nodes
    """
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    # Create gathering index with same shape as output
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    # Gather embeddings for specified indices
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes

########################################
# ENCODER
########################################
class TSP_Encoder(nn.Module):
    """Encoder module that transforms city coordinates into embeddings."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        # Linear projection from 2D coordinates to embedding space
        self.embedding = nn.Linear(2, embedding_dim)
        # Stack of encoder layers (Transformer-like)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data):
        """
        Forward pass through encoder.

        Args:
            data: Tensor of shape (batch, problem, 2) with city coordinates

        Returns:
            Tensor of shape (batch, problem, embedding_dim) with encoded node representations
        """
        # data.shape: (batch, problem, 2)

        # Project coordinates to embedding space
        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        # Pass through encoder layers
        out = embedded_input
        for layer in self.layers:
            out = layer(out)

        return out

class EncoderLayer(nn.Module):
    """Single encoder layer with multi-head attention and feed-forward network."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # Linear projections for query, key, value
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # Residual connections with normalization
        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        """
        Forward pass through encoder layer.

        Args:
            input1: Tensor of shape (batch, problem, EMBEDDING_DIM)

        Returns:
            Tensor of same shape with transformed representations
        """
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        # Compute query, key, value projections
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        # Multi-head self-attention
        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        # Combine multi-head outputs
        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        # Residual connection + layer norm
        out1 = self.addAndNormalization1(input1, multi_head_out)
        # Feed-forward network
        out2 = self.feedForward(out1)
        # Another residual connection + layer norm
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)

########################################
# DECODER
########################################
class TSP_Decoder(nn.Module):
    """Decoder module that selects next city based on current tour state."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # Query projections: first node and last node queries
        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # Key and value projections (cached from encoder outputs)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # Cached tensors for efficiency
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head attention
        self.single_head_key = None  # saved, for single-head attention (dot product)
        self.q_first = None  # saved q1, for multi-head attention (first node query)

    def set_kv(self, encoded_nodes):
        """
        Cache key and value projections from encoded nodes.
        Called once per problem instance.

        Args:
            encoded_nodes: Tensor of shape (batch, problem, embedding) from encoder
        """
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        # Compute and cache key/value projections
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        # Cache for single-head attention (dot product with embeddings)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        """
        Cache query projection for first selected node.
        Called at the beginning of each rollout.

        Args:
            encoded_q1: Tensor of shape (batch, n, embedding) where n is 1 or pomo_size
        """
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']

        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, ninf_mask, attention_bias_current_node):
        """
        Compute probability distribution over next cities.

        Args:
            encoded_last_node: Tensor of shape (batch, pomo, embedding) - last selected city
            ninf_mask: Tensor of shape (batch, pomo, problem) - mask for unavailable cities
            attention_bias_current_node: Optional heuristic bias tensor from ReEvo

        Returns:
            Probability tensor of shape (batch, pomo, problem) over next cities
        """
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        # Compute query for last selected node
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        # Combine first and last node queries
        q = self.q_first + q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        # Multi-head attention over all cities
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        # Combine multi-head outputs
        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        # Dot product between decoder output and all city embeddings
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        # ReEvo: add heuristic bias to attention scores
        score = score + attention_bias_current_node if IMPL_REEVO else score

        # Scale and clip logits
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        # Apply mask for unavailable cities
        score_masked = score_clipped + ninf_mask

        # Convert to probability distribution
        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs

########################################
# NN SUB CLASS / FUNCTIONS
########################################
def reshape_by_heads(qkv, head_num):
    """
    Reshape tensor for multi-head attention.

    Args:
        qkv: Tensor of shape (batch, n, head_num*key_dim)
        head_num: Number of attention heads

    Returns:
        Tensor of shape (batch, head_num, n, key_dim)
    """
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed

def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    """
    Compute multi-head attention.

    Args:
        q: Query tensor of shape (batch, head_num, n, key_dim)
        k: Key tensor of shape (batch, head_num, problem, key_dim)
        v: Value tensor of shape (batch, head_num, problem, key_dim)
        rank2_ninf_mask: Mask of shape (batch, problem) for unavailable cities
        rank3_ninf_mask: Mask of shape (batch, group, problem) for unavailable cities

    Returns:
        Tensor of shape (batch, n, head_num*key_dim) with attention output
    """
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    # Compute attention scores
    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    # Scale scores
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    # Apply masks if provided
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    # Compute attention weights
    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    # Apply attention to values
    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    # Reshape back to original format
    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat

class Add_And_Normalization_Module(nn.Module):
    """Residual connection with instance normalization."""

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        """
        Add input1 and input2, then apply instance normalization.

        Args:
            input1: Tensor of shape (batch, problem, embedding)
            input2: Tensor of shape (batch, problem, embedding)

        Returns:
            Normalized tensor of same shape
        """
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        # InstanceNorm1d expects (batch, embedding, problem)
        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans

class Feed_Forward_Module(nn.Module):
    """Simple feed-forward network with ReLU activation."""

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        """
        Forward pass through feed-forward network.

        Args:
            input1: Tensor of shape (batch, problem, embedding)

        Returns:
            Tensor of same shape after transformation
        """
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


# =====Tester class=====
class TSPTester:
    """Class for evaluating TSP model performance."""

    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):
        """
        Initialize tester with parameters.

        Args:
            env_params: Environment parameters
            model_params: Model architecture parameters
            tester_params: Testing configuration parameters
        """
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        # self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # Setup device (CPU/GPU)
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # Initialize environment and model
        self.env = TSPEnv(**self.env_params)
        self.model = TSPModel(**self.model_params)

        # Load pre-trained model checkpoint
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        """
        Run evaluation on test episodes.

        Returns:
            Average augmented score across all test episodes
        """
        self.time_estimator.reset()

        score_AM = AverageMeter()      # For scores without augmentation
        aug_score_AM = AverageMeter()  # For scores with augmentation

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        # Process test episodes in batches
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            # Test one batch
            score, aug_score = self._test_one_batch(batch_size)

            # Update statistics
            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            # Log progress (commented out in this version)
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            # self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
            #     episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)
            # if all_done:
            #     self.logger.info(" *** Test Done *** ")
            #     self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
            #     self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

        return aug_score_AM.avg

    def _test_one_batch(self, batch_size):
        """
        Test model on a single batch of problems.

        Args:
            batch_size: Number of TSP instances in batch

        Returns:
            Tuple of (score_without_augmentation, score_with_augmentation)
        """
        # Determine augmentation factor
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Setup model for evaluation
        self.model.eval()
        with torch.no_grad():
            # Load problems and reset environment
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            # Pre-compute encodings
            self.model.pre_forward(reset_state)

        # POMO Rollout: sequentially select cities
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Process results
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        # Get best result from each POMO starting point
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        # Score without augmentation (first augmentation slice)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        # Get best result across all augmentations
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item()


# =====Evaluation function=====
def eval_heuristic():

    tester = TSPTester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    avg_aug_obj = tester.run()
    return avg_aug_obj


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


# =====Main Function=====
if __name__ == '__main__':
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
    # Run instances: 200, 500, 1000; execution time:
    try:
        basepath = os.path.join(root_dir, "problems", problem)
        metrics = {}
        if not os.path.isfile(os.path.join(basepath, "checkpoints/checkpoint-3100.pt")):
            raise FileNotFoundError("No checkpoints found. Please see the readme.md and download the checkpoints.")

        if mode == 'train':
            dataset_path = os.path.join(basepath, f"dataset/{mode}{problem_size}_dataset.pt")
            env_params['test_file_path'] = dataset_path
            env_params['problem_size'] = problem_size
            tester_params['test_episodes'] = 10
            tester_params['test_batch_size'] = 10
            # Changes the current working directory to the problem directory so that all files are relative to the problem directory when executing `eval_heuristic`
            os.chdir(basepath)
            avg_obj = eval_heuristic()
            print("[*] Average:")
            print(avg_obj)
        else:
            for problem_size in dataset_conf['val']:  # options: 200, 500, 1000
                dataset_path = os.path.join(basepath, f"dataset/{mode}{problem_size}_dataset.pt")
                env_params['test_file_path'] = dataset_path
                env_params['problem_size'] = problem_size
                tester_params['test_episodes'] = 64
                tester_params['test_batch_size'] = 64
                # Changes the current working directory to the problem directory so that all files are relative to the problem directory when executing `eval_heuristic`
                os.chdir(basepath)
                avg_obj = eval_heuristic()
                print(f"[*] Average for {problem_size}: {avg_obj}")
                metrics[problem_size] = np.mean(avg_obj)
                
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