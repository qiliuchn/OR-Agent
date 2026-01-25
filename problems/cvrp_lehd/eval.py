# Evaluation script for CVRP-LEHD problem.
""" 
The LEHD (Learning with Heavy Decoder) model is a neural combinatorial optimization approach for solving Capacitated Vehicle Routing Problem (CVRP). 
The key ideas are: 
 - Architecture: Uses an encoder-decoder transformer architecture where:
    Encoder: Processes node coordinates into embeddings (1 layer)
    Decoder: Heavier structure (6 layers) that sequentially selects nodes; Heavy decoder allows better learning of complex routing patterns
"""
import os
import sys
import traceback
import numpy as np
from typing import Dict, Tuple, List, Any
import argparse
import json
import logging
from dataclasses import dataclass
import time
from datetime import datetime
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pytz
import seed_solution as solution_module  # Note: solution module script is generated and saved on the fly


# =====Load function to evolve=====
problem = "cvrp_lehd"
heuristics = getattr(solution_module, "heuristics")  # Get function to evolve


# =====Configuration and Parameters=====
USE_CUDA = False
CUDA_DEVICE_NUM = 0

# testing problem size
problem_size = 1000

# decode method: use RRC or not (greedy)
Use_RRC = False

# RRC budget
RRC_budget = 0

model_load_path = './checkpoints'
model_load_epoch = 40

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


# =====VRPEnv class=====
@dataclass
class Reset_State:
    problems: torch.Tensor

@dataclass
class Step_State:
    problems: torch.Tensor

class VRPEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size = None
        self.data_path = env_params['data_path']
        self.sub_path = env_params['sub_path']
        self.batch_size = None
        self.problems = None
        self.start_capacity=None
        self.selected_count = None
        self.selected_node_list = None
        self.selected_student_list = None
        self.episode = None

    def load_problems(self, episode, batch_size, ):
        self.episode = episode
        self.batch_size = batch_size

        self.problems_nodes = self.raw_data_nodes[episode:episode + batch_size]
        # shape (B,V+1,2)
        self.Batch_demand = self.raw_data_demand[episode:episode + batch_size]
        # shape (B,V+1)

        self.Batch_capacity = self.raw_data_capacity[episode:episode + batch_size]

        self.solution = self.raw_data_node_flag[episode:episode + batch_size]
        # shape (B,V,2)
        self.Batch_capacity = self.Batch_capacity[:,None].repeat(1,self.solution.shape[1]+1)
        # shape (B,V+1)

        self.problems = torch.cat((self.problems_nodes,self.Batch_demand[:,:,None],
                                   self.Batch_capacity[:,:,None]),dim=2)
        # shape (B,V+1,4)

        if self.sub_path:
            self.problems, self.solution = self.sampling_subpaths(self.problems, self.solution)

        self.problem_size = self.problems.shape[1]-1

    def vrp_whole_and_solution_subrandom_inverse(self, solution):
        clockwise_or_not = torch.rand(1)[0]

        if clockwise_or_not >= 0.5:
            solution = torch.flip(solution, dims=[1])
            index = torch.arange(solution.shape[1]).roll(shifts=1)
            solution[:, :, 1] = solution[:, index, 1]

        # 1.
        # find the number of subtours in each instance.
        # the total number of subpaths in all instances:     all_subtour_num，
        # The longest length in a subpath among all instances:  max_subtour_length
        batch_size = solution.shape[0]
        problem_size = solution.shape[1]
        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)
        all_subtour_num = torch.sum(visit_depot_num)
        fake_solution = torch.cat((solution[:, :, 1], torch.ones(batch_size)[:, None]), dim=1)
        start_from_depot = fake_solution.nonzero()
        start_from_depot_1 = start_from_depot[:, 1]
        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)
        sub_tours_length = start_from_depot_2 - start_from_depot_1
        max_subtour_length = torch.max(sub_tours_length)

        # 2。
        # For each subpath, take it out separately, pandding 0 to length max_subtour_length
        #For each instance, padding 0 to max_subtour_num number of subpaths
        # 3.
        # Put all subpaths of all instances into the same array
        start_from_depot2 = solution[:, :, 1].nonzero()
        start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()
        repeat_solutions_node = solution[:, :, 0].repeat_interleave(visit_depot_num, dim=0)
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)
        x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             >= start_from_depot2[:, 1][:, None]
        x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             <= start_from_depot3[:, 1][:, None]
        x3 = (x1 * x2).long()
        sub_tourss = double_repeat_solution_node * x3
        x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             < (start_from_depot2[:, 1][:, None] + max_subtour_length)
        x5 = x1 * x4
        sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)

        # 4.
        # For each row, a random number of [0,100] is generated, greater than 50 is positive and less than 50 is inverse
        clockwise_or_not = torch.rand(len(sub_tours_padding))
        clockwise_or_not_bool = clockwise_or_not.le(0.5)

        # 5.
        # For each row, randomly flip
        sub_tours_padding[clockwise_or_not_bool] = torch.flip(sub_tours_padding[clockwise_or_not_bool], dims=[1])

        # 6。
        # Map the subtours to the original solution matrix dimension
        sub_tourss_back = sub_tourss
        sub_tourss_back[x5] = sub_tours_padding.ravel()
        solution_node_flip = sub_tourss_back[sub_tourss_back.gt(0.1)].reshape(batch_size, problem_size)
        solution_flip = torch.cat((solution_node_flip.unsqueeze(2), solution[:, :, 1].unsqueeze(2)), dim=2)

        return solution_flip

    def vrp_whole_and_solution_subrandom_shift_V2inverse(self, solution):
        '''
        For each instance, shift randomly so that different end_with depot nodes can reach the last digit.
        '''
        problem_size = solution.shape[1]
        batch_size = solution.shape[0]

        start_from_depot = solution[:, :, 1].nonzero()
        end_with_depot = start_from_depot.clone()
        end_with_depot[:, 1] = end_with_depot[:, 1] - 1
        end_with_depot[end_with_depot.le(-0.5)] = solution.shape[1] - 1
        end_with_depot[:,1] = torch.roll(end_with_depot[:,1],dims=0,shifts=-1)
        visit_depot_num = solution[:,:,1].sum(1)
        min_length = torch.min(visit_depot_num)

        first_node_index = torch.randint(low=0, high=min_length, size=[1])[0]  # in [0,N)

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()

        temp_index = np.dot(visit_depot_num_numpy, temp_tri)
        temp_index_torch = torch.from_numpy(temp_index).long().cuda()

        pick_end_with_depot_index = temp_index_torch + first_node_index
        pick_end_with_depot_ = end_with_depot[pick_end_with_depot_index][:,1]
        first_index= pick_end_with_depot_
        end_indeex = pick_end_with_depot_+problem_size

        index = torch.arange(2*problem_size)[None,:].repeat(batch_size,1)
        x1 = index > first_index[:,None]
        x2 = index<= end_indeex[:,None]
        x3 = x1.int()*x2.int()
        double_solution = solution.repeat(1,2,1)
        solution = double_solution[x3.gt(0.5)[:,:,None].repeat(1,1,2)].reshape(batch_size,problem_size,2)

        return solution


    def sampling_subpaths(self, problems, solution, length_fix=False):
        # problems shape (B,V+1,4)
        # solution shape (B,V,2)

        # step：
        # 1.Extract subtour
        problems_size = problems.shape[1] - 1
        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]

        # the first node of subpath: uniform sampling, from 0 to N
        # 1.1
        length_of_subpath = torch.randint(low=4, high=problems_size + 1, size=[1])[0]  # in [4,V]
        solution = self.vrp_whole_and_solution_subrandom_inverse(solution)
        solution = self.vrp_whole_and_solution_subrandom_shift_V2inverse(solution)
        # 1.3
        #  Find the points that start from deopt, and then subtract 1 to get the point that ends with depot
        start_from_depot = solution[:, :, 1].nonzero()
        end_with_depot = start_from_depot
        end_with_depot[:, 1] = end_with_depot[:, 1] - 1
        end_with_depot[end_with_depot.le(-0.5)] = solution.shape[1] - 1

        # 1.4
        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)
        p = torch.rand(len(visit_depot_num))
        select_end_with_depot_node_index = p * visit_depot_num
        select_end_with_depot_node_index = torch.floor(select_end_with_depot_node_index).long()

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()

        temp_index = np.dot(visit_depot_num_numpy, temp_tri)
        temp_index_torch = torch.from_numpy(temp_index).long().cuda()
        select_end_with_depot_node_index_ = select_end_with_depot_node_index + temp_index_torch

        # This is the point at which each instance is randomly selected with an end with depot
        select_end_with_depot_node = end_with_depot[select_end_with_depot_node_index_, 1]

        # 1.5
        double_solution = torch.cat((solution, solution), dim=1)
        select_end_with_depot_node = select_end_with_depot_node + problems_size
        indexx = torch.arange(length_of_subpath).repeat(batch_size, 1)
        offset = select_end_with_depot_node - length_of_subpath + 1
        indexxxx = indexx + offset[:, None]
        sub_tour = double_solution[:, indexxxx, :]
        sub_tour = sub_tour.view(-1, length_of_subpath, 2)
        index_1 = torch.arange(0, batch_size * batch_size, batch_size)
        index_2 = torch.arange(batch_size)
        index_3 = index_1 + index_2
        sub_solution = sub_tour[index_3, :, :]

        # Calculate the capacity of the first point
        offset_index = problems.shape[0]
        start_index = indexxxx[:,0]

        x1 = torch.arange(double_solution[:offset_index,:,1].shape[1])<=start_index[:offset_index][:,None]

        start_capacity = 0
        before_is_via_depot_all = double_solution[:offset_index,:,1]*x1
        before_is_via_depot = before_is_via_depot_all.nonzero()

        visit_depot_num_2 = torch.sum(before_is_via_depot_all, dim=1)

        select_end_with_depot_node_index_2 = visit_depot_num_2-1

        temp_tri_2 = np.triu(np.ones((len(visit_depot_num_2), len(visit_depot_num_2))), k=1)
        visit_depot_num_numpy_2 = visit_depot_num_2.clone().cpu().numpy()

        temp_index_2 = np.dot(visit_depot_num_numpy_2, temp_tri_2)
        temp_index_torch_2 = torch.from_numpy(temp_index_2).long().cuda()

        select_end_with_depot_node_index_2 = select_end_with_depot_node_index_2 + temp_index_torch_2
        before_is_via_depot_index = before_is_via_depot[select_end_with_depot_node_index_2]

        before_start_index = before_is_via_depot_index[:,1]
        x2 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) <start_index[:offset_index][:, None]
        x3 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) >=before_start_index[:, None]
        x4 = x2 * x3
        double_solution_demand = problems[:offset_index,:,2][torch.arange(offset_index)[:,None].repeat(1,double_solution.shape[1]),double_solution[:offset_index,:,0] ]
        before_demand = double_solution_demand*x4
        self.satisfy_demand = before_demand.sum(1)

        problems[:offset_index,:,3] = problems[:offset_index,:,3] - self.satisfy_demand[:,None]
        # -----------------------------
        # 2. Update the subtour's index
        # -----------------------------
        # 2.1
        sub_solution_node = sub_solution[:, :, 0]
        new_sulution_ascending, rank = torch.sort(sub_solution_node, dim=-1, descending=False)  # 升序
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)  # 升序
        sub_solution[:, :, 0] = new_sulution_rank+1

        # 2.2
        index_2, _ = torch.cat((new_sulution_ascending, new_sulution_ascending, new_sulution_ascending, new_sulution_ascending), dim=1). \
            type(torch.long).sort(dim=-1, descending=False)

        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[1])  # shape: [B, 2current_step]
        temp = torch.arange((embedding_size), dtype=torch.long)[None, :].expand(batch_size, embedding_size)  # shape: [B, current_step]
        index_3 = temp.repeat([1, length_of_subpath])

        new_data = problems[index_1, index_2, index_3].view(batch_size, length_of_subpath, embedding_size)
        new_data = torch.cat((problems[:, 0, :].unsqueeze(dim=1), new_data), dim=1)

        return new_data, sub_solution

    def shuffle_data(self):
        # shuffle the training set data
        index = torch.randperm(len(self.raw_data_nodes)).long()
        self.raw_data_nodes = self.raw_data_nodes[index]
        self.raw_data_capacity = self.raw_data_capacity[index]
        self.raw_data_demand = self.raw_data_demand[index]
        self.raw_data_cost = self.raw_data_cost[index]
        self.raw_data_node_flag = self.raw_data_node_flag[index]

    def load_raw_data(self,episode=1000000, start_idx=0):
        def tow_col_nodeflag(node_flag):
            tow_col_node_flag = []
            V = int(len(node_flag) / 2)
            for i in range(V):
                tow_col_node_flag.append([node_flag[i], node_flag[V + i]])
            return tow_col_node_flag

        # Because the dataset is too large, I split it into two reads
        if self.env_params['mode']=='train':
            raise NotImplementedError

        if self.env_params['mode'] == 'val':
            self.raw_data_nodes = []
            self.raw_data_capacity = []
            self.raw_data_demand = []
            self.raw_data_cost = []
            self.raw_data_node_flag = []
            for line in tqdm(open(self.data_path, "r").readlines()[start_idx: start_idx + episode], ascii=True, disable=True):
                line = line.split(",")

                depot_index = int(line.index('depot'))
                customer_index = int(line.index('customer'))
                capacity_index = int(line.index('capacity'))
                demand_index = int(line.index('demand'))
                cost_index = int(line.index('cost'))
                node_flag_index = int(line.index('node_flag'))

                depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
                customer = [[float(line[idx]), float(line[idx + 1])] for idx in range(customer_index + 1, capacity_index, 2)]

                loc = depot + customer
                capacity = int(float(line[capacity_index + 1]))
                if int(line[demand_index + 1]) ==0:
                    demand = [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
                else:
                    demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]

                cost = float(line[cost_index + 1])
                node_flag = [int(line[idx]) for idx in range(node_flag_index + 1, len(line))]
                node_flag = tow_col_nodeflag(node_flag)
                self.raw_data_nodes.append(loc)
                self.raw_data_capacity.append(capacity)
                self.raw_data_demand.append(demand)
                self.raw_data_cost.append(cost)
                self.raw_data_node_flag.append(node_flag)

            self.raw_data_nodes = torch.tensor(self.raw_data_nodes, requires_grad=False)
            # shape (B,V+1,2)  customer num + depot
            self.raw_data_capacity = torch.tensor(self.raw_data_capacity, requires_grad=False)
            # shape (B )
            self.raw_data_demand = torch.tensor(self.raw_data_demand, requires_grad=False)
            # shape (B,V+1) customer num + depot
            self.raw_data_cost = torch.tensor(self.raw_data_cost, requires_grad=False)
            # shape (B )
            self.raw_data_node_flag = torch.tensor(self.raw_data_node_flag, requires_grad=False)
            # shape (B,V,2)

    def reset(self, mode, sample_size = 1):
        # start capacity per instance (shape [B])
        # capacity is stored in problems[:,:,3], repeated across nodes  [oai_citation:2‡evaluation_description.txt](sediment://file_00000000d43472068829c045718ecbab)
        self.start_capacity_vec = self.problems[:, 0, 3].clone().view(-1)

        self.rem_cap_teacher = self.start_capacity_vec.clone()
        self.rem_cap_student = self.start_capacity_vec.clone()
    
        self.selected_count = 0
        self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
        self.selected_teacher_flag = torch.zeros((self.batch_size, 0), dtype=torch.long)
        self.selected_student_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
        self.selected_student_flag= torch.zeros((self.batch_size, 0), dtype=torch.long)
        self.step_state = Step_State(problems=self.problems)
        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        reward_student = None
        done = False
        return self.step_state, reward, reward_student, done

    def step(self, selected, selected_student, selected_flag_teacher, selected_flag_student):
        self.selected_count += 1

        # ---- Teacher capacity (optional, kept separate) ----
        # Refill if teacher explicitly returns to depot
        is_depot_teacher = (selected_flag_teacher == 1)
        self.rem_cap_teacher = torch.where(is_depot_teacher, self.start_capacity_vec, self.rem_cap_teacher)

        demand_teacher = self.Batch_demand.gather(1, selected[:, None]).squeeze(1)  # [B]
        need_return_teacher = self.rem_cap_teacher < demand_teacher
        if need_return_teacher.any():
            selected_flag_teacher = selected_flag_teacher.clone()
            selected_flag_teacher[need_return_teacher] = 1
            self.rem_cap_teacher[need_return_teacher] = self.start_capacity_vec[need_return_teacher]
        self.rem_cap_teacher = self.rem_cap_teacher - demand_teacher

        # ---- Student capacity (the one that matters for scoring) ----
        is_depot_student = (selected_flag_student == 1)
        self.rem_cap_student = torch.where(is_depot_student, self.start_capacity_vec, self.rem_cap_student)

        demand_student = self.Batch_demand.gather(1, selected_student[:, None]).squeeze(1)  # [B]
        need_return_student = self.rem_cap_student < demand_student
        if need_return_student.any():
            selected_flag_student = selected_flag_student.clone()
            selected_flag_student[need_return_student] = 1
            self.rem_cap_student[need_return_student] = self.start_capacity_vec[need_return_student]
        self.rem_cap_student = self.rem_cap_student - demand_student

        # Optional but recommended: expose student remaining capacity to the model via state.problems
        self.problems[:, :, 3] = self.rem_cap_student[:, None].expand(-1, self.problems.size(1))

        # ---- Record selections/flags ----
        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, None]), dim=1)
        self.selected_teacher_flag = torch.cat((self.selected_teacher_flag, selected_flag_teacher[:, None]), dim=1)

        self.selected_student_list = torch.cat((self.selected_student_list, selected_student[:, None]), dim=1)
        self.selected_student_flag = torch.cat((self.selected_student_flag, selected_flag_student[:, None]), dim=1)

        done = (self.selected_count == self.problems.shape[1] - 1)
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

    def cal_length(self, problems, order_node, order_flag):
        # problems:   [B,V+1,2]
        # order_node: [B,V]
        # order_flag: [B,V]
        order_node_ = order_node.clone()
        order_flag_ = order_flag.clone()
        
        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)

        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0

        roll_node = order_node_.roll(dims=1, shifts=1)

        problem_size = problems.shape[1] - 1

        order_gathering_index = order_node_.unsqueeze(2).expand(-1, problem_size, 2)
        order_loc = problems.gather(dim=1, index=order_gathering_index)

        roll_gathering_index = roll_node.unsqueeze(2).expand(-1, problem_size, 2)
        roll_loc = problems.gather(dim=1, index=roll_gathering_index)

        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)

        order_lengths = ((order_loc - flag_loc) ** 2)

        order_flag_[:,0]=0
        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)

        roll_lengths = ((roll_loc - flag_loc) ** 2)
        length = (order_lengths.sum(2).sqrt() + roll_lengths.sum(2).sqrt()).sum(1)

        return length

    def _get_travel_distance(self):
        # teacher's length
        problems = self.problems[:,:,[0,1]]
        order_node = self.solution[:,:,0]
        order_flag = self.solution[:,:,1]
        travel_distances = self.cal_length( problems, order_node, order_flag)
        # self.drawPic_VRP(problems[0,:,:], order_node[0],order_flag[0],name='teather')

        # trained model's distance
        problems = self.problems[:, :, [0, 1]]
        order_node = self.selected_student_list.clone()
        order_flag = self.selected_student_flag.clone()

        travel_distances_student = self.cal_length(problems, order_node, order_flag)
        # draw figure， validate the result.
        # self.drawPic_VRP(problems[0,:,:], order_node[0],order_flag[0],name='student')

        return -travel_distances, -travel_distances_student

    def _get_travel_distance_2(self, problems_, solution_,):
        problems = problems_[:, :, [0, 1]].clone()
        order_node = solution_[:, :, 0].clone()
        order_flag = solution_[:, :, 1].clone()
        travel_distances = self.cal_length(problems, order_node, order_flag)
        return travel_distances

    def destroy_solution(self, problem, complete_solution):
        self.problems, self.solution, first_node_index,length_of_subpath,double_solution = self.sampling_subpaths_repair(
            problem, complete_solution, mode=self.env_params['mode'])
        partial_solution_length = self._get_travel_distance_2(self.problems, self.solution)
        return partial_solution_length,first_node_index,length_of_subpath,double_solution

    def sampling_subpaths_repair(self, problems, solution, length_fix=False, mode='test', repair=True):
        # problems shape (B,V+1,4)
        # solution shape (B,V,2) index从1开始
        problems_size = problems.shape[1] - 1
        # print('problems_size',problems_size)
        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]

        # the first node of subpath: uniform sampling, from 0 to N
        # 1.1
        length_of_subpath = torch.randint(low=4, high=problems_size+1 , size=[1])[0]  # in [4,N]

        start_from_depot = solution[:, :, 1].nonzero()

        end_with_depot = start_from_depot
        end_with_depot[:, 1] = end_with_depot[:, 1] - 1
        end_with_depot[end_with_depot.le(-0.5)] = solution.shape[1] - 1

        # 1.4
        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)

        p = torch.rand(len(visit_depot_num))
        select_end_with_depot_node_index = p * visit_depot_num
        select_end_with_depot_node_index = torch.floor(select_end_with_depot_node_index).long()

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()

        temp_index = np.dot(visit_depot_num_numpy, temp_tri)
        temp_index_torch = torch.from_numpy(temp_index).long()

        select_end_with_depot_node_index_ = select_end_with_depot_node_index + temp_index_torch

        select_end_with_depot_node = end_with_depot[select_end_with_depot_node_index_, 1]
        # 1.5
        double_solution = torch.cat((solution, solution), dim=1)

        select_end_with_depot_node = select_end_with_depot_node + problems_size

        indexx = torch.arange(length_of_subpath).repeat(batch_size, 1)
        offset = select_end_with_depot_node - length_of_subpath + 1

        indexxxx = indexx + offset[:, None]

        sub_solu_index1 = torch.arange(batch_size)[:,None].repeat(1,2*length_of_subpath)
        sub_solu_index2 =indexxxx.repeat_interleave(2,dim=1)
        sub_solu_index3 = torch.arange(double_solution.shape[2])[None,:].repeat(batch_size,length_of_subpath)
        sub_solution = double_solution[sub_solu_index1,sub_solu_index2,sub_solu_index3].reshape(batch_size,length_of_subpath,2)

        offset_index = problems.shape[0]
        start_index = indexxxx[:, 0]

        x1 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) <= start_index[:offset_index][:, None]

        start_capacity = 0
        before_is_via_depot_all = double_solution[:offset_index, :, 1] * x1
        before_is_via_depot = before_is_via_depot_all.nonzero()

        visit_depot_num_2 = torch.sum(before_is_via_depot_all, dim=1)

        select_end_with_depot_node_index_2 = visit_depot_num_2 - 1

        temp_tri_2 = np.triu(np.ones((len(visit_depot_num_2), len(visit_depot_num_2))), k=1)
        visit_depot_num_numpy_2 = visit_depot_num_2.clone().cpu().numpy()

        temp_index_2 = np.dot(visit_depot_num_numpy_2, temp_tri_2)
        temp_index_torch_2 = torch.from_numpy(temp_index_2).long()

        select_end_with_depot_node_index_2 = select_end_with_depot_node_index_2 + temp_index_torch_2
        before_is_via_depot_index = before_is_via_depot[select_end_with_depot_node_index_2]

        before_start_index = before_is_via_depot_index[:, 1]
        x2 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) < start_index[:offset_index][:, None]
        x3 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) >= before_start_index[:, None]
        x4 = x2 * x3
        double_solution_demand = problems[:offset_index, :, 2][
            torch.arange(offset_index)[:, None].repeat(1, double_solution.shape[1]), double_solution[:offset_index, :, 0]]

        before_demand = double_solution_demand * x4

        self.satisfy_demand = before_demand.sum(1)

        problems[:offset_index, :, 3] = problems[:offset_index, :, 3] - self.satisfy_demand[:, None]

        # -----------------------------
        # 2.
        # -----------------------------
        # 2.1
        sub_solution_node = sub_solution[:, :, 0]
        new_sulution_ascending, rank = torch.sort(sub_solution_node, dim=-1, descending=False)  # 升序
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)  # 升序
        sub_solution[:, :, 0] = new_sulution_rank + 1
        # 2.2
        index_2, _ = torch.cat((new_sulution_ascending, new_sulution_ascending, new_sulution_ascending, new_sulution_ascending), dim=1). \
            type(torch.long).sort(dim=-1, descending=False)

        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[1])  # shape: [B, 2current_step]
        temp = torch.arange((embedding_size), dtype=torch.long)[None, :].expand(batch_size, embedding_size)  # shape: [B, current_step]
        index_3 = temp.repeat([1, length_of_subpath])

        new_data = problems[index_1, index_2, index_3].view(batch_size, length_of_subpath, embedding_size)
        new_data = torch.cat((problems[:, 0, :].unsqueeze(dim=1), new_data), dim=1)
        if repair == True:
            return new_data, sub_solution,start_index,length_of_subpath,double_solution
        else:
            return new_data, sub_solution

    def valida_solution_legal(self, problem, solution,capacity_=50):
        capacitys = {100: 50,
                     200: 80,
                     500: 100,
                     1000: 250}

        problem_size = solution.shape[1]
        capacity = capacitys[problem_size]

        coor = problem[:, :, [0, 1]]
        demand = problem[:, :, 2]

        order_node = solution[:, :, 0].clone()
        order_flag = solution[:, :, 1].clone()

        if_begin_flag_legal = (order_flag[:,0]!=1).any()

        # 0.
        if if_begin_flag_legal:
            assert False, 'e1: wrong begin_flag_legal!'

        # 1. Determine whether each index of the solution node list is unique
        uniques = torch.unique(order_node[0])
        if len(uniques) != problem.shape[1] - 1:
            assert False, 'e2: wrong node list!'


        # 2. Find the demand for each sub tour and determine whether it exceeds capacity
        batch_size = solution.shape[0]
        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)
        all_subtour_num = torch.sum(visit_depot_num)
        fake_solution = torch.cat((solution[:, :, 1], torch.ones(batch_size)[:, None]), dim=1)
        start_from_depot = fake_solution.nonzero()
        start_from_depot_1 = start_from_depot[:, 1]
        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)
        sub_tours_length = start_from_depot_2 - start_from_depot_1
        max_subtour_length = torch.max(sub_tours_length)

        start_from_depot2 = solution[:, :, 1].nonzero()
        start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()

        repeat_solutions_node = solution[:, :, 0].repeat_interleave(visit_depot_num, dim=0)
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)

        x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             >= start_from_depot2[:, 1][:, None]
        x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             <= start_from_depot3[:, 1][:, None]
        x3 = (x1 * x2).long()
        sub_tourss = double_repeat_solution_node * x3
        x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             < (start_from_depot2[:, 1][:, None] + max_subtour_length)
        x5 = x1 * x4
        sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)
        demands = torch.repeat_interleave(demand, repeats=visit_depot_num, dim=0)
        index = torch.arange(sub_tours_padding.shape[0])[:, None].repeat(1, sub_tours_padding.shape[1])
        sub_tours_demands = demands[index, sub_tours_padding].sum(dim=1)
        if_legal = (sub_tours_demands > capacity)

        if if_legal.any():
            assert False, 'e3: wrong capacity!'

        return


# =====VRPModel class=====
IMPL_REEVO = True

class VRPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']
        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None

    def forward(self, state, selected_node_list, solution, current_step,raw_data_capacity=None,):
        # solution's shape : [B, V]
        self.capacity = raw_data_capacity.ravel()[0].item()
        batch_size = state.problems.shape[0]
        problem_size = state.problems.shape[1]
        split_line = problem_size - 1

        def probs_to_selected_nodes(probs_,split_line_,batch_size_):
            selected_node_student_ = probs_.argmax(dim=1)  # shape: B
            is_via_depot_student_ = selected_node_student_ >= split_line_ # Nodes with an index greater than customer_num are via depot
            not_via_depot_student_ = selected_node_student_ < split_line_

            selected_flag_student_ = torch.zeros(batch_size_,dtype=torch.int)
            selected_flag_student_[is_via_depot_student_] = 1
            selected_node_student_[is_via_depot_student_] = selected_node_student_[is_via_depot_student_]-split_line_ +1
            selected_flag_student_[not_via_depot_student_] = 0
            selected_node_student_[not_via_depot_student_] = selected_node_student_[not_via_depot_student_]+ 1
            return selected_node_student_, selected_flag_student_ # node 的 index 从 1 开始

        if self.mode == 'train':
            raise NotImplementedError

        if self.mode == 'val':
            remaining_capacity = state.problems[:, 1, 3]
            # print(state.problems.shape)
            if current_step <= 1:
                self.encoded_nodes = self.encoder(state.problems,self.capacity)
                # print(self.encoded_nodes.shape) (B, V+1, EMBEDDING_DIM)
                coor = state.problems[:, :, :2]
                demands = state.problems[:, :, 2]
                ######################## ReEvo #############################
                distance_matrices = torch.cdist(coor, coor, p=2)
                if IMPL_REEVO:
                    self.attention_bias = torch.stack([
                        heuristics(distance_matrices[i], demands[i]) for i in range(distance_matrices.size(0))
                    ], dim=0)
                    assert not torch.isnan(self.attention_bias).any()
                    assert not torch.isinf(self.attention_bias).any()
                else:
                    self.attention_bias = None
                ###########################################################

            probs = self.decoder(self.encoded_nodes, selected_node_list,self.capacity, remaining_capacity, attention_bias=self.attention_bias)

            selected_node_student = probs.argmax(dim=1)  # shape: B
            is_via_depot_student = selected_node_student >= split_line  # 节点index大于 customer_num的是通过depot的
            not_via_depot_student = selected_node_student < split_line
            # print(selected_node_student)
            selected_flag_student = torch.zeros(batch_size, dtype=torch.int)
            selected_flag_student[is_via_depot_student] = 1
            selected_node_student[is_via_depot_student] = selected_node_student[is_via_depot_student] - split_line + 1
            selected_flag_student[not_via_depot_student] = 0
            selected_node_student[not_via_depot_student] = selected_node_student[not_via_depot_student] + 1

            selected_node_teacher = selected_node_student
            selected_flag_teacher = selected_flag_student
            loss_node = torch.tensor(0)

        return loss_node,selected_node_teacher,  selected_node_student,selected_flag_teacher,selected_flag_student

class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num =  1
        self.embedding = nn.Linear(3, embedding_dim, bias=True)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data_,capacity):
        data = data_.clone().detach()
        data= data[:,:,:3]
        data[:,:,2] = data[:,:,2]/capacity
        embedded_input = self.embedding(data)
        out = embedded_input  # [B*(V-1), problem_size - current_step +2, embedding_dim]
        layer_count = 0
        for layer in self.layers:
            out = layer(out)
            layer_count += 1
        return out

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
        out_concat = multi_head_attention(q, k, v)  # shape: (B, n, head_num*key_dim)
        multi_head_out = self.multi_head_combine(out_concat)  # shape: (B, n, embedding_dim)
        out1 = input1 +   multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 + out2
        return out3
        # shape: (batch, problem, EMBEDDING_DIM)

########################################
# DECODER
########################################
class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        decoder_layer_num = self.model_params['decoder_layer_num']

        self.embedding_first_node = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node = nn.Linear(embedding_dim+1, embedding_dim, bias=True)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(decoder_layer_num)])
        self.Linear_final = nn.Linear(embedding_dim, 2, bias=True)

    def _get_new_data(self, data, selected_node_list, prob_size, B_V):
        list = selected_node_list
        new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1)
        new_list_len = prob_size - list.shape[1]  # shape: [B, V-current_step]
        index_2 = list.type(torch.long)
        index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2.shape[1])
        new_list[index_1, index_2] = -2
        unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)
        new_data = data
        emb_dim = data.shape[-1]
        new_data_len = new_list_len
        index_2_ = unselect_list.repeat_interleave(repeats=emb_dim, dim=1)
        index_1_ = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2_.shape[1])
        index_3_ = torch.arange(emb_dim)[None, :].repeat(repeats=(B_V, new_data_len))
        new_data_ = new_data[index_1_, index_2_, index_3_].view(B_V, new_data_len, emb_dim)
        return new_data_, unselect_list

    def _get_encoding(self,encoded_nodes, node_index_to_pick):
        batch_size = node_index_to_pick.size(0)
        pomo_size = node_index_to_pick.size(1)
        embedding_dim = encoded_nodes.size(2)
        gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
        return picked_nodes


    def forward(self, data,selected_node_list,capacity,remaining_capacity,attention_bias=None):
        data_ = data[:,1:,:].clone().detach()
        selected_node_list_ = selected_node_list.clone().detach() - 1
        batch_size_V = data_.shape[0]  # B
        problem_size = data_.shape[1]
        new_data = data_.clone().detach()
        left_encoded_node, unselect_list = self._get_new_data(new_data, selected_node_list_, problem_size, batch_size_V)
        embedded_first_node = data[:,[0],:]
        
        if selected_node_list_.shape[1]==0:
            embedded_last_node = data[:,[0],:]
        else:
            embedded_last_node = self._get_encoding(new_data, selected_node_list_[:, [-1]])

        remaining_capacity = remaining_capacity.reshape(batch_size_V,1,1)/capacity
        first_node_cat = torch.cat((embedded_first_node,remaining_capacity), dim=2)
        last_node_cat = torch.cat((embedded_last_node,remaining_capacity), dim=2)

        embedded_first_node_ = self.embedding_first_node(first_node_cat)
        embedded_last_node_ = self.embedding_last_node(last_node_cat)
        embeded_all = torch.cat((embedded_first_node_,left_encoded_node,embedded_last_node_), dim=1)
        out = embeded_all  # [B*(V-1), problem_size - current_step +2, embedding_dim]

        layer_count = 0
        for layer in self.layers:
            out = layer(out)
            layer_count += 1

        out = self.Linear_final(out)  # shape: [B*(V-1), reminding_nodes_number + 2, embedding_dim ]
        # print(out.shape) 202 -> 3 for CVRP 200

        # ReEvo: add attention bias
        if IMPL_REEVO:
            unselect_list = unselect_list + 1
            # Fetch the last selected node's attention bias for each batch
            current_node_idx = selected_node_list[:, -1] if selected_node_list.shape[1] > 0 else torch.zeros(batch_size_V, dtype=torch.long, device=selected_node_list.device) 
            # shape: (B,)
            attention_bias_current_node = attention_bias[torch.arange(batch_size_V), current_node_idx]  # shape: (B, V)
            attention_bias_current_node_unselect = attention_bias_current_node[torch.arange(batch_size_V)[:, None], unselect_list]  # shape: (B, V-current_step)
            out[:, 1:-1] += attention_bias_current_node_unselect[:, :, None]  # shape: (B, V-current_step, 2)
            
        out[:, [0, -1], :] = out[:, [0, -1], :] + float('-inf')  # first node、last node
        out = torch.cat((out[:, :, 0], out[:, :, 1]), dim=1)  # shape:(B, 2 * ( V - current_step ))
        props = F.softmax(out, dim=-1)
        customer_num = left_encoded_node.shape[1]
        props = torch.cat((props[:, 1:customer_num + 1], props[:, customer_num + 1 + 1 + 1:-1]), dim=1)

        index_small = torch.le(props, 1e-5)
        props_clone = props.clone()
        props_clone[index_small] = props_clone[index_small] + torch.tensor(1e-7, dtype=props_clone[index_small].dtype)
        props = props_clone
        new_props = torch.zeros(batch_size_V, 2 * (problem_size))

        # The function of the following part is to fill the probability of props into the new_props,
        index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:,None].repeat(1,selected_node_list_.shape[1]*2)
        index_2_ =torch.cat( ((selected_node_list_).type(torch.long), (problem_size)+ (selected_node_list_).type(torch.long) ),dim=-1) # shape: [B*V, n]
        new_props[index_1_, index_2_,] = -2
        index = torch.gt(new_props, -1).view(batch_size_V, -1)
        new_props[index] = props.ravel()

        return new_props


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
        out3 = out1 + out2
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
        return self.W2(F.relu(self.W1(input1)))


# =====VRPTester class=====
class VRPTester():
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

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            # torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda:0')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = VRPEnv(**self.env_params)
        self.model = VRPModel(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 = TimeEstimator()

    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()
        self.env.load_raw_data(self.tester_params['test_episodes'], start_idx=self.tester_params['test_start_idx'])
        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
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

            score_teacher, score_student, problems_size = self._test_one_batch(
                episode, batch_size, clock=self.time_estimator_2,logger = None)
            current_gap = (score_student - score_teacher) / score_teacher
            if problems_size < 100:
                problems_100.append(current_gap)
                # print('problems_100 mean gap:', np.mean(problems_100), len(problems_100))
            elif 100 <= problems_size < 200:
                problems_100_200.append(current_gap)
                # print('problems_100_200 mean gap:', np.mean(problems_100_200), len(problems_100_200))
            elif 200 <= problems_size < 500:
                problems_200_500.append(current_gap)
                # print('problems_200_500 mean gap:', np.mean(problems_200_500), len(problems_200_500))
            elif 500 <= problems_size < 1000:
                problems_500_1000.append(current_gap)
                # print('problems_500_1000 mean gap:', np.mean(problems_500_1000), len(problems_500_1000))
            elif 1000 <= problems_size:
                problems_1000.append(current_gap)
                # print('problems_1000 mean gap:', np.mean(problems_1000), len(problems_1000))
                
            score_AM.update(score_teacher, batch_size)
            score_student_AM.update(score_student, batch_size)

            episode += batch_size
            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            # self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Score_teacher:{:.4f}, Score_studetnt: {:.4f}".format(
            #     episode, test_num_episode, elapsed_time_str, remain_time_str, score_teacher, score_student))

            all_done = (episode == test_num_episode)
            if all_done:

                # self.logger.info(" *** Test Done *** ")
                # self.logger.info(" Teacher SCORE: {:.4f} ".format(score_AM.avg))
                # self.logger.info(" Student SCORE: {:.4f} ".format(score_student_AM.avg))
                gap_ = (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100
                # self.logger.info(" Gap: {:.4f}%".format(gap_))

        return score_AM.avg, score_student_AM.avg, gap_

    def decide_whether_to_repair_solution(self,
                                          after_repair_sub_solution, before_reward, after_reward,
                                          first_node_index, length_of_subpath, double_solution):
        the_whole_problem_size = int(double_solution.shape[1] / 2)
        batch_size = len(double_solution)

        temp = torch.arange(double_solution.shape[1])
        x3 = temp >= first_node_index[:, None].long()
        x4 = temp < (first_node_index[:, None] + length_of_subpath).long()
        x5 = x3 * x4

        origin_sub_solution = double_solution[x5.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, length_of_subpath, 2)
        jjj, _ = torch.sort(origin_sub_solution[:, :, 0], dim=1, descending=False)
        index = torch.arange(batch_size)[:, None].repeat(1, jjj.shape[1])
        kkk_2 = jjj[index, after_repair_sub_solution[:, :, 0] - 1]
        after_repair_sub_solution[:, :, 0] = kkk_2
        if_repair = before_reward > after_reward

        need_to_repari_double_solution = double_solution[if_repair]
        need_to_repari_double_solution[x5[if_repair].unsqueeze(2).repeat(1, 1, 2)] = after_repair_sub_solution[if_repair].ravel()
        double_solution[if_repair] = need_to_repari_double_solution

        x6 = temp >= (first_node_index[:, None] + length_of_subpath - the_whole_problem_size).long()
        x7 = temp < (first_node_index[:, None] + length_of_subpath).long()
        x8 = x6 * x7
        after_repair_complete_solution = double_solution[x8.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, the_whole_problem_size, -1)

        return after_repair_complete_solution

    def _test_one_batch(self, episode, batch_size, clock=None,logger = None):
        random_seed = 12
        torch.manual_seed(random_seed)
        self.model.eval()

        with torch.no_grad():
            self.env.load_problems(episode, batch_size)
            reset_state, _, _ = self.env.reset(self.env_params['mode'])
            current_step = 0

            state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node
            self.origin_problem = self.env.problems.clone().detach()

            self.optimal_length= self.env._get_travel_distance_2(self.origin_problem, self.env.solution)
            name = 'vrp'+str(self.env.solution.shape[1])
            B_V = batch_size * 1

            while not done:
                loss_node, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                    self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                               raw_data_capacity=self.env.raw_data_capacity)  # 更新被选择的点和概率
                if current_step == 0:
                    selected_flag_teacher = torch.ones(B_V, dtype=torch.int)
                    selected_flag_student = selected_flag_teacher
                current_step += 1

                state, reward, reward_student, done = \
                    self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student)

            # print('Get first complete solution!')
            
            # 1. The complete solution is obtained
            best_select_node_list = torch.cat((self.env.selected_student_list.reshape(batch_size, -1, 1),
                                               self.env.selected_student_flag.reshape(batch_size, -1, 1)), dim=2)

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

            escape_time, _ = clock.get_est_string(1, 1)

            # self.logger.info("Greedy, name:{}, gap:{:5f} %, Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(name,
                # ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
            # current_best_length.mean().item(), self.optimal_length.mean().item()))

            budget = self.env_params['RRC_budget']

            for bbbb in range(budget):
                torch.cuda.empty_cache()
                self.env.load_problems(episode, batch_size)

                # 2. Sample the partial solution
                best_select_node_list = self.env.vrp_whole_and_solution_subrandom_inverse(best_select_node_list)
                partial_solution_length, first_node_index, length_of_subpath, double_solution = \
                    self.env.destroy_solution(self.env.problems, best_select_node_list)
                before_repair_sub_solution = self.env.solution
                before_reward = partial_solution_length
                current_step = 0
                reset_state, _, _ = self.env.reset(self.env_params['mode'])

                state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

                # 3. Reconstruct the partial solution.
                while not done:
                    if current_step == 0:
                        selected_teacher = self.env.solution[:, 0, 0]
                        selected_flag_teacher = self.env.solution[:, 0, 1]
                        selected_student = selected_teacher
                        selected_flag_student = selected_flag_teacher
                    else:
                        _, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                            self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                                       raw_data_capacity=self.env.raw_data_capacity)

                    current_step += 1

                    state, reward, reward_student, done = \
                        self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student)

                ahter_repair_sub_solution = torch.cat((self.env.selected_student_list.unsqueeze(2),
                                                       self.env.selected_student_flag.unsqueeze(2)), dim=2)

                after_reward = - reward_student
                after_repair_complete_solution = self.decide_whether_to_repair_solution(
                    ahter_repair_sub_solution,
                    before_reward, after_reward, first_node_index, length_of_subpath, double_solution)

                best_select_node_list = after_repair_complete_solution
                current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
                escape_time, _ = clock.get_est_string(1, 1)
                # self.logger.info(
                #     "RRC step{}, name:{}, gap:{:6f} %, Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(
                #          bbbb, name, ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100,
                #         escape_time,current_best_length.mean().item(), self.optimal_length.mean().item()))

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            # print(f'current_best_length', (current_best_length.mean() - self.optimal_length.mean())
            #       / self.optimal_length.mean() * 100, '%', 'escape time:', escape_time,
            #       f'optimal:{self.optimal_length.mean()}, current_best:{current_best_length.mean()}')

            # 4. Cycle until the budget is consumed.
            # self.env.valida_solution_legal(self.origin_problem, best_select_node_list)

            return self.optimal_length.mean().item(), current_best_length.mean().item(), self.env.problem_size


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
        'sqrt_embedding_dim': 128**(1/2),
        'decoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'ff_hidden_dim': 512,
    }

    tester_params = {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        'test_episodes': test_paras[problem_size][1],   # 65
        'test_batch_size': test_paras[problem_size][2],
        'test_start_idx': test_paras[problem_size][3],
    }
    tester_params['model_load']={
        'path': model_load_path,
        'epoch': model_load_epoch,
    }
    if use_RRC is not None:
        env_params['RRC_budget']=0
    if cuda_device_num is not None:
        tester_params['cuda_device_num'] = cuda_device_num
    tester = VRPTester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    score_optimal, score_student, gap = tester.run()
    
    return score_optimal, score_student,gap


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
    # Run instances 200, 500; execution time: 206s
    try:
        basepath = os.path.join(root_dir, "problems", problem)
        if not os.path.isfile(os.path.join(basepath, "checkpoints/checkpoint-40.pt")):
            raise FileNotFoundError("No checkpoints found. Please see the readme.md and download the checkpoints.")
        if not os.path.isfile(os.path.join(basepath, "data/vrp200_test_lkh.txt")):
            raise FileNotFoundError("No test data found. Please see the readme.md and download the data.")

        if mode == 'train':
            test_paras = {
                # problem_size: [filename, episode, batch, start_idx]
                200: ['vrp200_test_lkh.txt', 10, 10, 0],
                500: ['vrp500_test_lkh.txt', 10, 10, 0],
                1000: ['vrp1000_test_lkh.txt', 10, 10, 0],
            }
            # Changes the current working directory to the problem directory so that all files are relative to the problem directory when executing `eval_heuristic`
            os.chdir(basepath)
            score_optimal, score_student, gap = eval_heuristic()
            print(f"Optimal: {score_optimal}, Student: {score_student}, Gap (%): {gap}")
            print("[*] Average:")
            print(score_student)
        else:
            if mode == 'val':
                test_paras = {
                    # problem_size: [filename, episode, batch, start_idx]
                    200: ['vrp200_test_lkh.txt', 32, 32, 10],
                    500: ['vrp500_test_lkh.txt', 32, 32, 10],
                    1000: ['vrp1000_test_lkh.txt', 32, 32, 10],
                }
            else:
                test_paras = {
                    # problem_size: [filename, episode, batch, start_idx]
                    200: ['vrp200_test_lkh.txt', 64, 64, 64],
                    500: ['vrp500_test_lkh.txt', 64, 64, 64],
                    1000: ['vrp1000_test_lkh.txt', 64, 64, 64],
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