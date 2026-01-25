# Evaluation script for TSP-LEHD problem.
""" 
The LEHD (Learning with Heavy Decoder) model is a neural combinatorial optimization approach for solving Traveling Salesman Problems (TSP). 
The key ideas are: 
 - Architecture: Uses an encoder-decoder transformer architecture where:
    Encoder: Processes node coordinates into embeddings (1 layer)
    Decoder: Heavier structure (6 layers) that sequentially selects nodes; Heavy decoder allows better learning of complex routing patterns
"""
import os
import sys
import traceback
import argparse
import logging
from typing import Dict, Tuple, List, Any
import numpy as np
import json
import shutil
import time
from datetime import datetime
import pytz
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import seed_solution as solution_module  # Note: solution module script is generated and saved on the fly


# =====Load function to evolve=====
problem = "tsp_lehd"
heuristics = getattr(solution_module, "heuristics")  # Get function to evolve


# =====Configuration and Parameters=====
# Global configuration parameters for the LEHD TSP solver
DEBUG_MODE = False  # Enable debug output if True
USE_CUDA = False    # Use GPU acceleration if True
CUDA_DEVICE_NUM = 0  # GPU device number

# decode method: use RRC or not (greedy)
# RRC (Route Reconstruction and Correction) is a local search improvement method
Use_RRC = False

# RRC budget - number of RRC iterations to perform
RRC_budget = 0
# Path to load pre-trained model checkpoints
model_load_path = 'checkpoints/'
# Which epoch checkpoint to load
model_load_epoch = 150

# If RRC is disabled, set budget to 0
if not Use_RRC:
    RRC_budget = 0


# =====Utility functions=====
process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))
b = os.path.abspath('.')
result_folder = b+'/result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'

def get_result_folder():
    return result_folder

def set_result_folder(folder):
    global result_folder
    result_folder = folder

def create_logger(log_file=None):
    if 'filepath' not in log_file:
        log_file['filepath'] = get_result_folder()

    if 'desc' in log_file:
        log_file['filepath'] = log_file['filepath'].format(desc='_' + log_file['desc'])
    else:
        log_file['filepath'] = log_file['filepath'].format(desc='')

    set_result_folder(log_file['filepath'])

    if 'filename' in log_file:
        filename = log_file['filepath'] + '/' + log_file['filename']
    else:
        filename = log_file['filepath'] + '/' + 'log.txt'

    if not os.path.exists(log_file['filepath']):
        os.makedirs(log_file['filepath'])

    file_mode = 'a' if os.path.isfile(filename)  else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0

class LogData:
    def __init__(self):
        self.keys = set()
        self.data = {}

    def get_raw_data(self):
        return self.keys, self.data

    def set_raw_data(self, r_data):
        self.keys, self.data = r_data

    def append_all(self, key, *args):
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
        if not self.has_key(key):
            return None
        return self.data[key][-1]

    def has_key(self, key):
        return key in self.keys

    def get(self, key):
        split = np.hsplit(np.array(self.data[key]), 2)

        return split[1].squeeze().tolist()

    def getXY(self, key, start_idx=0):
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
        return self.keys

class TimeEstimator:
    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))

def util_print_log_array(logger, result_log: LogData):
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    for key in result_log.get_keys():
        logger.info('{} = {}'.format(key+'_list', result_log.get(key)))

def copy_all_src(dst_root):
    # execution dir
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])

    # home dir setting
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))

    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else:
        home_dir = tmp_dir1

    # make target directory
    dst_path = os.path.join(dst_root, 'src')

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for item in sys.modules.items():
        key, value = item

        if hasattr(value, '__file__') and value.__file__:
            src_abspath = os.path.abspath(value.__file__)

            if os.path.commonprefix([home_dir, src_abspath]) == home_dir:
                dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))

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
# Environment for TSP problem simulation
@dataclass
class Reset_State:
    """State when environment is reset"""
    problems: torch.Tensor  # Node coordinates
    # shape: (batch, problem, 2)

@dataclass
class Step_State:
    """State during step execution"""
    data: torch.Tensor  # Current problem data

class TSPEnv:
    """Environment for Traveling Salesman Problem simulation"""
    def __init__(self, **env_params):
        self.env_params = env_params  # Environment parameters
        self.problem_size = None  # Number of nodes in problem
        self.data_path = env_params['data_path']  # Path to TSP data files
        self.sub_path = env_params['sub_path']  # Whether to use sub-path sampling
        self.batch_size = None  # Batch size for processing
        self.problems = None  # Current batch of problems (node coordinates)
        self.raw_data_nodes = []  # Raw node data loaded from file
        self.raw_data_tours = []  # Raw optimal tours loaded from file
        self.selected_count = None  # Counter for selected nodes
        self.selected_node_list = None  # List of nodes selected by model
        self.selected_student_list = None  # List of nodes selected by student model
        self.episode = None  # Current episode index

    def load_problems(self, episode, batch_size):
        """Load a batch of problems for processing"""
        self.episode = episode
        self.batch_size = batch_size
        # Load problems and their optimal solutions
        self.problems, self.solution = self.raw_data_nodes[episode:episode + batch_size], self.raw_data_tours[episode:episode + batch_size]
        # shape: [B,V,2]  ;  shape: [B,V] where B=batch, V=num_nodes

        # If sub-path sampling is enabled, sample a sub-path from the full problem
        if self.sub_path:
            self.problems, self.solution = self.sampling_subpaths(self.problems, self.solution,mode='train')

        # Randomly decide whether to reverse the solution (data augmentation)
        if_inverse = True
        if_inverse_index = torch.randint(low=0, high=100, size=[1])[0]  # Random number 0-99
        if if_inverse_index < 50:  # 50% chance to not inverse
            if_inverse = False

        if if_inverse:
            self.solution = torch.flip( self.solution , dims=[1])  # Reverse the tour

        self.problem_size = self.problems.shape[1]  # Update problem size

    def sampling_subpaths(self, problems, solution, length_fix=False, mode='test', repair=False):
        """Sample a contiguous sub-path from the full TSP tour"""
        problems_size = problems.shape[1]  # Number of nodes
        batch_size = problems.shape[0]     # Batch size
        embedding_size = problems.shape[2] # Coordinate dimension (2 for x,y)

        # Randomly select starting node for sub-path
        first_node_index = torch.randint(low=0, high=problems_size, size=[1])[0]  # in [0,N)

        # Sample sub-path length: uniform sampling from 4 to N (inclusive)
        if mode == 'test':
            length_of_subpath = torch.randint(low=4, high=problems_size + 1, size=[1])[0]  # in [4,N]
        else:
            if length_fix:
                length_of_subpath = problems_size  # Use full path
            else:
                length_of_subpath = torch.randint(low=4, high=problems_size + 1, size=[1])[0]  # in [4,N]

        # -----------------------------
        # Create new solution (tour indices) for sub-path
        # -----------------------------
        # Double the solution to handle wrap-around when sampling contiguous segments
        double_solution = torch.cat([solution, solution], dim=-1)
        # Extract contiguous sub-path starting from first_node_index
        new_sulution = double_solution[:, first_node_index: first_node_index + length_of_subpath]
        # Sort to get node indices in ascending order, then get rank mapping
        new_sulution_ascending, rank = torch.sort(new_sulution, dim=-1, descending=False)  # Ascending sort
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)  # Get rank positions

        # -----------------------------
        # Create new problem (node coordinates) for sub-path
        # -----------------------------
        # Create index arrays to extract corresponding node coordinates
        index_2, _ = torch.cat((new_sulution_ascending, new_sulution_ascending), dim=1).type(torch.long).sort(dim=-1,
                                                                                                              descending=False)  # shape: [B, 2*current_step]
        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[1])  # shape: [B, 2*current_step]
        temp = torch.arange((embedding_size), dtype=torch.long)[None, :].expand(batch_size, embedding_size)  # shape: [B, current_step]
        index_3 = temp.repeat([1, length_of_subpath])

        # Extract coordinates for nodes in the sub-path
        new_data = problems[index_1, index_2, index_3].view(batch_size, length_of_subpath, 2)

        if repair == True:
            # Return additional info needed for RRC repair
            return new_data, new_sulution_rank, first_node_index, length_of_subpath, double_solution
        else:
            return new_data, new_sulution_rank

    def shuffle_data(self):
        index = torch.randperm(len(self.raw_data_nodes)).long()
        self.raw_data_nodes = self.raw_data_nodes[index]
        self.raw_data_tours = self.raw_data_tours[index]

    def load_raw_data(self, episode,begin_index=0):
        print('load raw dataset begin!')
        self.raw_data_nodes = []
        self.raw_data_tours = []
        for line in tqdm(open(self.data_path, "r").readlines()[0+begin_index:episode+begin_index], ascii=True):
            line = line.split(" ")
            num_nodes = int(line.index('output') // 2)
            nodes = [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]

            self.raw_data_nodes.append(nodes)
            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]

            self.raw_data_tours.append(tour_nodes)

        self.raw_data_nodes = torch.tensor(self.raw_data_nodes,requires_grad=False)
        self.raw_data_tours = torch.tensor(self.raw_data_tours,requires_grad=False)
        print(f'load raw dataset done!', )

    def destroy_solution(self, problem, complete_solution):
        """Destroy a complete solution by sampling a sub-path (for RRC local search)"""
        # Sample a sub-path from the complete solution
        self.problems, self.solution,first_node_index,length_of_subpath,double_solution = self.sampling_subpaths(
            problem, complete_solution, mode=self.env_params['mode'],repair=True)

        # Calculate length of the partial solution (sub-path)
        partial_solution_length = self._get_travel_distance_2(self.problems, self.solution)
        return partial_solution_length,first_node_index,length_of_subpath,double_solution

    def reset(self, mode,):
        self.selected_count = 0
        self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
        self.selected_student_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
        self.step_state = Step_State(data=self.problems)
        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        reward_student = None
        done = False
        return self.step_state, reward, reward_student, done

    def step(self, selected, selected_student):
        self.selected_count += 1
        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, None]), dim=1)  # shape: [B, current_step]
        self.selected_student_list = torch.cat((self.selected_student_list, selected_student[:, None]), dim=1)
        done = (self.selected_count == self.problems.shape[1])
        if done:
            reward, reward_student = self._get_travel_distance()
        else:
            reward, reward_student = None, None

        return self.step_state, reward, reward_student, done

    def make_dir(self,path_destination):
        isExists = os.path.exists(path_destination)
        if not isExists:
            os.makedirs(path_destination)
        return
    
    def _get_travel_distance(self):
        """Calculate travel distance for both optimal solution and student's solution"""
        # Calculate optimal solution distance (teacher)
        gathering_index = self.solution.unsqueeze(2).expand(self.batch_size, self.problems.shape[1], 2)
        seq_expanded = self.problems
        ordered_seq = seq_expanded.gather(dim=1, index=gathering_index)  # Reorder nodes according to solution
        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)  # Shift for pairwise distance calculation
        segment_lengths = ((ordered_seq - rolled_seq) ** 2)  # Squared Euclidean distances
        segment_lengths = segment_lengths.sum(2).sqrt()  # Actual Euclidean distances
        travel_distances = segment_lengths.sum(1)  # Sum distances for complete tour

        # Calculate trained model's distance (student)
        gathering_index_student = self.selected_student_list.unsqueeze(2).expand(-1, self.problems.shape[1], 2)
        ordered_seq_student = self.problems.gather(dim=1, index=gathering_index_student)
        rolled_seq_student = ordered_seq_student.roll(dims=1, shifts=-1)
        segment_lengths_student = ((ordered_seq_student - rolled_seq_student) ** 2)
        segment_lengths_student = segment_lengths_student.sum(2).sqrt()
        # shape: (batch,problem) - distances between consecutive nodes
        travel_distances_student = segment_lengths_student.sum(1)
        # shape: (batch) - total tour length
        return travel_distances, travel_distances_student

    def _get_travel_distance_2(self, problems, solution):
        """Calculate travel distance for given problems and solution (generic version)"""
        gathering_index = solution.unsqueeze(2).expand(problems.shape[0], problems.shape[1], 2)
        seq_expanded = problems
        ordered_seq = seq_expanded.gather(dim=1, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2)
        segment_lengths = segment_lengths.sum(2).sqrt()
        travel_distances = segment_lengths.sum(1)
        return travel_distances


# =====TSPModel class=====
# Main neural network model for TSP solving
IMPL_REEVO = True  # Whether to implement ReEvo enhancement (heuristic attention bias)

class TSPModel(nn.Module):
    """Main LEHD model for TSP solving with encoder-decoder architecture"""
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']  # 'train' or 'val'
        self.encoder = TSP_Encoder(**model_params)  # Light encoder (1 layer)
        self.decoder = TSP_Decoder(**model_params)  # Heavy decoder (6 layers)
        self.encoded_nodes = None  # Cached node embeddings

    def forward(self, state, selected_node_list, solution, current_step, repair=False):
        """Forward pass to select next node in tour"""
        # solution's shape : [B, V] where B=batch, V=num_nodes
        batch_size_V = state.data.size(0)

        if self.mode == 'train':
            raise NotImplementedError  # Training not implemented in this eval script

        if self.mode == 'val':
            if repair == False:  # Normal greedy decoding
                if current_step <= 1:
                    # Encode nodes only once at the beginning (cached for efficiency)
                    self.encoded_nodes = self.encoder(state.data)  # state.data.shape: [B, V, 2]

                    ######################## ReEvo Enhancement #############################
                    # Compute pairwise Euclidean distances between all nodes
                    distance_matrices = torch.cdist(state.data, state.data, p=2)
                    if IMPL_REEVO:
                        # Create attention bias using heuristic function for each problem in batch
                        self.attention_bias = torch.stack([
                            heuristics(distance_matrices[i]) for i in range(distance_matrices.size(0))
                        ], dim=0)
                        # Safety checks
                        assert not torch.isnan(self.attention_bias).any()
                        assert not torch.isinf(self.attention_bias).any()
                    else:
                        self.attention_bias = None
                    #######################################################################

                # selected_node_list.shape: (batch size, current_step)
                # Get probability distribution over next nodes
                probs = self.decoder(self.encoded_nodes, selected_node_list, attention_bias=self.attention_bias)

                # Greedy selection: choose node with highest probability
                selected_student = probs.argmax(dim=1)
                selected_teacher = selected_student  # In val mode, teacher = student
                prob = 1  # Greedy selection probability

            if repair == True:  # RRC repair mode (not implemented)
                raise NotImplementedError
                if current_step <= 2:
                    self.encoded_nodes = self.encoder(state.data)

                probs = self.decoder(self.encoded_nodes, selected_node_list)

                selected_student = probs.argmax(dim=1)
                selected_teacher = selected_student
                prob = 1

        return selected_teacher, prob, 1, selected_student

########################################
# ENCODER
########################################
class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num =  1
        self.embedding = nn.Linear(2, embedding_dim, bias=True)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data):
        embedded_input = self.embedding(data)
        out = embedded_input
        for layer in self.layers:
            out = layer(out)
        return out

class TSP_Decoder(nn.Module):
    """Heavy decoder for sequential node selection (6 layers)"""
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['decoder_layer_num']  # Typically 6 (heavy decoder)
        # Special embeddings for first and last selected nodes (context)
        self.embedding_first_node = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node = nn.Linear(embedding_dim, embedding_dim, bias=True)
        # Multiple decoder layers (heavy part of the model)
        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(encoder_layer_num)])
        self.k_1 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.Linear_final = nn.Linear(embedding_dim, 1, bias=True)  # Final scoring layer

    def _get_new_data(self, data, selected_node_list, prob_size, B_V):
        list = selected_node_list
        new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1)
        new_list_len = prob_size - list.shape[1]  # shape: [B, V-current_step]
        index_2 = list.type(torch.long)
        index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2.shape[1])
        new_list[index_1, index_2] = -2
        unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)
        # ----------------------------------------------------------------------------
        new_data = data
        emb_dim = data.shape[-1]
        new_data_len = new_list_len
        index_2_ = unselect_list.repeat_interleave(repeats=emb_dim, dim=1)
        index_1_ = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2_.shape[1])
        index_3_ = torch.arange(emb_dim)[None, :].repeat(repeats=(B_V, new_data_len))
        new_data_ = new_data[index_1_, index_2_, index_3_].view(B_V, new_data_len, emb_dim)
        return new_data_, unselect_list

    def _get_encoding(self, encoded_nodes, node_index_to_pick):
        batch_size = node_index_to_pick.size(0)
        pomo_size = node_index_to_pick.size(1)
        embedding_dim = encoded_nodes.size(2)
        gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
        return picked_nodes

    def forward(self, data, selected_node_list, attention_bias=None):
        """Forward pass to compute probabilities for next node selection"""
        # data.shape = (B, problem, embedding_dim)
        batch_size_V = data.shape[0]  # B = batch size
        problem_size = data.shape[1]  # V = number of nodes
        new_data = data

        # selected_node_list.shape: [B, current_step]
        # Get unselected nodes and their embeddings
        left_encoded_node, unselect_list = self._get_new_data(new_data, selected_node_list, problem_size, batch_size_V)

        # Get embeddings of first and last selected nodes (context)
        first_and_last_node = self._get_encoding(new_data,selected_node_list[:,[0,-1]])
        embedded_first_node_ = first_and_last_node[:,0]  # First selected node
        embedded_last_node_ = first_and_last_node[:,1]   # Last selected node

        # Transform context embeddings
        embedded_first_node_ = self.embedding_first_node(embedded_first_node_)
        embedded_last_node_ = self.embedding_last_node(embedded_last_node_)

        # Concatenate: [first_node, unselected_nodes, last_node]
        out = torch.cat((embedded_first_node_.unsqueeze(1), left_encoded_node,embedded_last_node_.unsqueeze(1)), dim=1)
        layer_count=0

        # Process through heavy decoder layers
        for layer in self.layers:
            out = layer(out)
            layer_count += 1

        # Final linear layer to get scores
        out = self.Linear_final(out).squeeze(-1)
        # Linear_final: (B, V, 1) -> (B, V)

        # ReEvo: add attention bias to guide selection
        if IMPL_REEVO:
            # Fetch the last selected node's attention bias for each batch
            current_node_idx = selected_node_list[:, -1]  # shape: (B,)
            attention_bias_current_node = attention_bias[torch.arange(batch_size_V), current_node_idx]  # shape: (B, V)
            # Extract bias only for unselected nodes
            attention_bias_current_node_unselect = attention_bias_current_node[torch.arange(batch_size_V)[:, None], unselect_list]  # shape: (B, V-current_step)
            out[:, 1:-1] += attention_bias_current_node_unselect  # Add bias to unselected nodes

        # Mask first and last positions (already selected)
        out[:, [0,-1]] = out[:, [0,-1]] + float('-inf')

        # Convert scores to probabilities
        props = F.softmax(out, dim=-1)
        props = props[:, 1:-1]  # Remove first and last positions (context nodes)

        # Prevent probabilities from being too small (numerical stability)
        index_small = torch.le(props, 1e-5)
        props_clone = props.clone()
        props_clone[index_small] = props_clone[index_small] + torch.tensor(1e-7, dtype=props_clone[index_small].dtype)
        props = props_clone

        # Create full probability distribution over all nodes
        new_props = torch.zeros(batch_size_V, problem_size)
        index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:, None].expand(batch_size_V, selected_node_list.shape[1])  # shape: [B*(V-1), n]
        index_2_ = selected_node_list.type(torch.long)
        new_props[index_1_, index_2_] = -2  # Mark selected nodes
        index = torch.gt(new_props, -1).view(batch_size_V, -1)  # Find unselected positions

        new_props[index] = props.ravel()  # Fill unselected positions with probabilities
        return new_props

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.feedForward = Feed_Forward_Module(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 +  out2
        return out3

class DecoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.feedForward = Feed_Forward_Module(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 +  out2
        return out3

def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed

def multi_head_attention(q, k, v):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3))  # shape: (B, head_num, n, n)
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    weights = nn.Softmax(dim=3)(score_scaled)  # shape: (B, head_num, n, n)
    out = torch.matmul(weights, v)  # shape: (B, head_num, n, key_dim)
    out_transposed = out.transpose(1, 2)  # shape: (B, n, head_num, key_dim)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)  # shape: (B, n, head_num*key_dim)
    return out_concat

class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)
        return self.W2(F.relu(self.W1(input1)))


# ======TSPTester=====
class TSPTester():
    """Tester class for evaluating the model on TSP problems"""
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        # result folder, logger
        # self.logger = getLogger(name='trainer')
        # self.result_folder = get_result_folder()

        # cuda setup
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
        torch.set_printoptions(precision=20)  # Set printing precision

        # Utility objects for time estimation
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 =  TimeEstimator()

    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()
        self.env.load_raw_data(self.tester_params['test_episodes'], begin_index=self.tester_params['test_start_idx'])
        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        problems_100 = []
        problems_100_200 = []
        problems_200_500 = []
        problems_500_1000 = []
        problems_1000 = []
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score_teacher, score_student,problems_size = self._test_one_batch(episode,batch_size,clock=self.time_estimator_2)
            current_gap = (score_student-score_teacher)/score_teacher
            if problems_size<100:
                problems_100.append(current_gap)
                print('problems_100 mean gap:',np.mean(problems_100),len(problems_100))
            elif 100<=problems_size<200:
                problems_100_200.append(current_gap)
                print('problems_100_200 mean gap:', np.mean(problems_100_200),len(problems_100_200))
            elif 200<=problems_size<500:
                problems_200_500.append(current_gap)
                print('problems_200_500 mean gap:', np.mean(problems_200_500),len(problems_200_500))
            elif 500<=problems_size<1000:
                problems_500_1000.append(current_gap)
                print('problems_500_1000 mean gap:', np.mean(problems_500_1000),len(problems_500_1000))
            elif 1000<=problems_size:
                problems_1000.append(current_gap)
                print('problems_1000 mean gap:', np.mean(problems_1000),len(problems_1000))

            score_AM.update(score_teacher, batch_size)
            score_student_AM.update(score_student, batch_size)
            episode += batch_size
            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            # self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Score_teacher:{:.4f},Score_studetnt: {:.4f},".format(
                # episode, test_num_episode, elapsed_time_str, remain_time_str, score_teacher,score_student,))
            all_done = (episode == test_num_episode)

            if all_done:
            #     self.logger.info(" *** Test Done *** ")
            #     self.logger.info(" Teacher SCORE: {:.4f} ".format(score_AM.avg))
            #     self.logger.info(" Student SCORE: {:.4f} ".format(score_student_AM.avg))
            #     self.logger.info(" Gap: {:.4f}%".format((score_student_AM.avg-score_AM.avg) / score_AM.avg * 100))
                gap_ = (score_student_AM.avg-score_AM.avg) / score_AM.avg * 100
                
        return score_AM.avg, score_student_AM.avg, gap_

    def decide_whether_to_repair_solution(self,after_repair_sub_solution,before_reward, after_reward,
                                          first_node_index, length_of_subpath, double_solution):

        the_whole_problem_size  = int(double_solution.shape[1]/2)
        other_part_1 = double_solution[:,:first_node_index]
        other_part_2 = double_solution[:,first_node_index+length_of_subpath:]
        origin_sub_solution = double_solution[:, first_node_index : first_node_index+length_of_subpath]

        jjj, _ = torch.sort(origin_sub_solution, dim=1, descending=False)
        index = torch.arange(jjj.shape[0])[:,None].repeat(1,jjj.shape[1])
        kkk_2 = jjj[index,after_repair_sub_solution]
        if_repair = before_reward>after_reward
        double_solution[if_repair] = torch.cat((other_part_1[if_repair],
                                                        kkk_2[if_repair],
                                                        other_part_2[if_repair]),dim=1)
        after_repair_complete_solution = double_solution[:,first_node_index:first_node_index+the_whole_problem_size]

        return after_repair_complete_solution

    def _test_one_batch(self, episode, batch_size,clock=None):
        """Test one batch of problems with optional RRC improvement"""
        self.model.eval()
        with torch.no_grad():
            # Load problems for current batch
            self.env.load_problems(episode, batch_size)
            self.origin_problem = self.env.problems
            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            # Calculate optimal tour length (ground truth)
            self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)
            name = 'TSP'+str(self.origin_problem.shape[1])
            B_V = batch_size * 1

            # Step 1: Generate initial solution greedily
            current_step = 0
            state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

            while not done:
                if current_step == 0:
                    # Start from node 0 (convention)
                    selected_teacher= torch.zeros(B_V,dtype=torch.int64)
                    selected_student = selected_teacher
                else:
                    # Model selects next node
                    selected_teacher, _,_,selected_student = self.model(
                        state,self.env.selected_node_list,self.env.solution,current_step,)
                current_step += 1
                state, reward,reward_student, done = self.env.step(selected_teacher, selected_student)
            #print('Get first complete solution!')

            # 1. The complete solution is obtained.
            best_select_node_list = self.env.selected_node_list
            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            escape_time, _ = clock.get_est_string(1, 1)

            # Calculate optimality gap
            gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
            # self.logger.info("greedy, name:{}, gap:{:4f} %,  Elapsed[{}], stu_l:{:4f} , opt_l:{:4f}".format(
            #     name, gap, escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))
            budget = self.env_params['RRC_budget']

            # Step 2: Apply RRC (Route Reconstruction and Correction) if budget > 0
            for bbbb in range(budget):
                self.env.load_problems(episode, batch_size)
                # 2. Randomly sample the partial solution
                # random inverse (data augmentation)
                if_inverse = True
                if_inverse_index = torch.randint(low=0, high=100, size=[1])[0]  # in [4,N]
                if if_inverse_index<50:
                    if_inverse=False

                if if_inverse:
                    best_select_node_list = torch.flip(best_select_node_list,dims=[1])

                # Sample partial solution (destroy operation)
                partial_solution_length, first_node_index,length_of_subpath,double_solution = self.env.destroy_solution(self.env.problems,best_select_node_list )
                before_reward = partial_solution_length  # Length before repair
                current_step = 0
                reset_state, _, _ = self.env.reset(self.env_params['mode'])
                state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

                # 3. Reconstruct the sub-problem using model
                while not done:
                    if current_step == 0:
                        selected_teacher = self.env.solution[:, -1]  # destination node
                        selected_student = self.env.solution[:, -1]

                    elif current_step == 1:
                        selected_teacher = self.env.solution[:, 0]  # starting node
                        selected_student = self.env.solution[:, 0]

                    else:
                        selected_teacher, _,_,selected_student = self.model(
                            state,self.env.selected_node_list,self.env.solution,current_step, repair = True)

                    current_step += 1
                    state, reward, reward_student, done = self.env.step(selected_teacher, selected_student)

                ahter_repair_sub_solution = torch.roll(self.env.selected_node_list,shifts=-1,dims=1)
                after_reward = reward_student  # Length after repair

                # 4. Decide whether to accept the reconstructed partial solution (accept if better)
                after_repair_complete_solution = self.decide_whether_to_repair_solution(ahter_repair_sub_solution,
                                                  before_reward, after_reward, first_node_index, length_of_subpath,
                                                                                        double_solution )
                best_select_node_list = after_repair_complete_solution
                current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

                escape_time,_ = clock.get_est_string(1, 1)
                gap =  ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
                # self.logger.info("RRC step{}, name:{}, gap:{:4f} %, Elapsed[{}], stu_l:{:4f} , opt_l:{:4f}".format(
                #    bbbb,name,gap, escape_time,current_best_length.mean().item(), self.optimal_length.mean().item()))

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            gap = (current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean() * 100
            print(name, f'current_best_length',gap , '%')

            # 5. Cycle until the budget is consumed.
            return self.optimal_length.mean().item(),current_best_length.mean().item(), self.env.problem_size


# ======Evaluation function=====
def eval_heuristic(use_RRC=None, cuda_device_num=None):
    env_params = {
        'mode': mode,
        'data_path': f"./data/{test_paras[problem_size][0]}",
        'sub_path': False,
        'RRC_budget': RRC_budget
    }

    model_params = {
        'mode': mode,
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128 ** (1 / 2),
        'decoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'ff_hidden_dim': 512,
    }

    tester_params = {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        'test_episodes': test_paras[problem_size][1],
        'test_batch_size': test_paras[problem_size][2],
        'test_start_idx': test_paras[problem_size][3],
    }

    if use_RRC is not None:
        env_params['RRC_budget'] = 0
    if cuda_device_num is not None:
        tester_params['cuda_device_num'] = cuda_device_num

    tester_params['model_load'] = {
        'path': model_load_path,
        'epoch': model_load_epoch,
    }

    tester = TSPTester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student, gap


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

# =======Main function=====
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
    # Run instance: 200, 500, 1000; execution time: 210s
    try:
        basepath = os.path.join(root_dir, "problems", problem)
        if not os.path.isfile(os.path.join(basepath, "checkpoints/checkpoint-150.pt")):
            raise FileNotFoundError("No checkpoints found. Please see the readme.md and download the checkpoints.")
        if not os.path.isfile(os.path.join(basepath, "data/test_TSP200_n128.txt")):
            raise FileNotFoundError("No test data found. Please see the readme.md and download the data.")

        if mode == 'train':
            test_paras = {
                # problem_size: [filename, episode, batch, start_idx]
                200: ['test_TSP200_n128.txt', 10, 10, 0],
                500: ['test_TSP500_n128.txt', 10, 10, 0],
                1000: ['test_TSP1000_n128.txt', 10, 10, 0],
            }
            # Changes the current working directory to the problem directory so that all files are relative to the problem directory when executing `eval_heuristic`
            os.chdir(basepath)
            score_optimal, score_student, gap = eval_heuristic()
            print("[*] Average:")
            print(score_student)
        else:
            if mode == 'val':
                test_paras = {
                    # problem_size: [filename, episode, batch, start_idx]
                    200: ['test_TSP200_n128.txt', 32, 32, 10],
                    500: ['test_TSP500_n128.txt', 32, 32, 10],
                    1000: ['test_TSP1000_n128.txt', 32, 32, 10],
                }
            else:
                test_paras = {
                    # problem_size: [filename, episode, batch, start_idx]
                    200: ['test_TSP200_n128.txt', 64, 64, 64],
                    500: ['test_TSP500_n128.txt', 64, 64, 64],
                    1000: ['test_TSP1000_n128.txt', 64, 64, 64],
                }
            metrics = {}
            for problem_size in [200, 500]:  # options: 200, 500, 1000
                # Changes the current working directory to the problem directory so that all files are relative to the problem directory when executing `eval_heuristic`
                os.chdir(basepath)
                score_optimal, score_student, gap = eval_heuristic()
                print(f"Problem size: {problem_size}, Optimal: {score_optimal}, Student: {score_student}, Gap (%): {gap}")
                metrics[problem_size] = float(np.mean(score_student))
                
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