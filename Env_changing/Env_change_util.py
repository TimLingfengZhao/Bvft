
from abc import ABC, abstractmethod
import numpy as np
import sys
from multiprocessing import shared_memory
import os
import re
import multiprocess.context as ctx
import pickle

from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import ShareableList
from pathos.multiprocessing import  Pool
from pathos.multiprocessing import  ProcessingPool
import heapq
import multiprocessing
from multiprocessing import Lock
import threading
from typing import Sequence
import gc

from d3rlpy.datasets import get_d4rl
import gym
import pathos

from pathos.multiprocessing import  ProcessPool
from fastapi import FastAPI
from celery import Celery
from celery.result import AsyncResult
import psutil
import random
import copy
import concurrent.futures
from d3rlpy.models.encoders import VectorEncoderFactory
import sys
import torch.optim as optim
import matplotlib.pyplot as plt
from d3rlpy.dataset import FIFOBuffer, ReplayBuffer
import torch
import pandas as pd
from datetime import datetime
import time
from d3rlpy.dataset import MDPDataset, Episode
from scope_rl.dataset import SyntheticDataset
from scope_rl.policy import GaussianHead
from scope_rl.ope import OffPolicyEvaluation as OPE
# from top_k_cal import *
from scope_rl.ope.continuous import DirectMethod as DM
from scope_rl.policy import ContinuousEvalHead
from d3rlpy.algos import DDPGConfig
from d3rlpy.algos import SACConfig
from d3rlpy.dataset import create_fifo_replay_buffer
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQLConfig
from d3rlpy.algos import BCQConfig
from d3rlpy.ope import FQE, FQEConfig
from d3rlpy.models.q_functions import QFunctionFactory, MeanQFunctionFactory
from d3rlpy.models.q_functions import IQNQFunctionFactory
from d3rlpy.models.encoders import DefaultEncoderFactory
import time
from d3rlpy.preprocessing import MinMaxActionScaler
from d3rlpy.preprocessing import StandardObservationScaler
from sklearn.metrics import mean_squared_error
import argparse
from scope_rl.ope import CreateOPEInput
import d3rlpy
from scope_rl.utils import check_array
import torch
import torch.nn as nn
from scope_rl.ope.estimators_base import BaseOffPolicyEstimator
from d3rlpy.dataset import Episode

from abc import ABC, abstractmethod
import numpy as np
import sys
import os

import heapq
from typing import Sequence
from d3rlpy.datasets import get_d4rl
import gym
import random
from d3rlpy.models.encoders import VectorEncoderFactory
import sys
import torch.optim as optim
import matplotlib.pyplot as plt

import dill
from d3rlpy.dataset import FIFOBuffer, ReplayBuffer
import torch
import pandas as pd
from datetime import datetime
import time
from d3rlpy.dataset import MDPDataset, Episode
from scope_rl.dataset import SyntheticDataset
from scope_rl.policy import GaussianHead
from scope_rl.ope import OffPolicyEvaluation as OPE
# from top_k_cal import *
from scope_rl.ope.continuous import DirectMethod as DM
from scope_rl.policy import ContinuousEvalHead
from d3rlpy.algos import DDPGConfig
from d3rlpy.dataset import create_fifo_replay_buffer
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQLConfig
from d3rlpy.algos import BCQConfig
from d3rlpy.ope import FQE, FQEConfig
from d3rlpy.models.q_functions import QFunctionFactory, MeanQFunctionFactory
from d3rlpy.models.q_functions import IQNQFunctionFactory
from d3rlpy.models.encoders import DefaultEncoderFactory
import time
from d3rlpy.preprocessing import MinMaxActionScaler
from d3rlpy.preprocessing import StandardObservationScaler
from sklearn.metrics import mean_squared_error
import argparse
from scope_rl.ope import CreateOPEInput
import d3rlpy
from scope_rl.utils import check_array
import torch
import torch.nn as nn
from scope_rl.ope.estimators_base import BaseOffPolicyEstimator
# random state
# dataset_d, env = get_d4rl('hopper-medium-v0')
from d3rlpy.dataset import Episode
import gymnasium
import logging
from multiprocessing import Process, resource_tracker
from memory_profiler import memory_usage
from memory_profiler import profile
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
        self.array = np.array([0.1,0.2,0.3])
    def predict(self, states):
        return np.array([self.array for _ in range(len(states))])


class CustomDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.current = 0
        self.size = 0
        self.length = len(dataset)
        for i in range(len(dataset)):
            self.size += len(dataset[i]["actions"])

    def get_iter_length(self, iteration_number):
        return len(self.dataset[iteration_number]["observations"])

    def get_state_shape(self):
        first_state = self.dataset[0]["observations"]
        return np.array(first_state).shape

    def sample(self, iteration_number):
        dones = np.array(self.dataset[iteration_number]["dones"])
        states = np.array(self.dataset[iteration_number]["observations"])
        actions = np.array(self.dataset[iteration_number]["actions"])
        padded_next_states = np.array(self.dataset[iteration_number]["next_steps"])
        rewards = np.array(self.dataset[iteration_number]["rewards"])

        return states, actions, padded_next_states, rewards, dones


def delete_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print("The folder does not exist.")
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted {file_path}")
            elif os.path.isdir(file_path):
                print(f"Skipping directory {file_path}")
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

class BvftRecord:
    def __init__(self):
        self.resolutions = []
        self.losses = []
        self.loss_matrices = []
        self.group_counts = []
        self.avg_q = []
        self.optimal_grouping_skyline = []
        self.e_q_star_diff = []
        self.bellman_residual = []
        self.ranking = []

    def record_resolution(self, resolution):
        self.resolutions.append(resolution)

    def record_ranking(self,ranking):
        self.ranking = ranking

    def record_losses(self, max_loss):
        self.losses.append(max_loss)

    def record_loss_matrix(self, matrix):
        self.loss_matrices.append(matrix)

    def record_group_count(self, count):
        self.group_counts.append(count)

    def record_avg_q(self, avg_q):
        self.avg_q.append(avg_q)

    def record_optimal_grouping_skyline(self, skyline):
        self.optimal_grouping_skyline.append(skyline)

    def record_e_q_star_diff(self, diff):
        self.e_q_star_diff = diff

    def record_bellman_residual(self, br):
        self.bellman_residual = br

    def save(self, directory="Bvft_Records", file_prefix="BvftRecord_"):
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(directory, f"{file_prefix}.pkl")
        with open(filename, "wb") as file:
            pickle.dump(self, file)
        print(f"Record saved to {filename}")
        return filename

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as file:
            return pickle.load(file)
    def summary(self):
        pass
class BVFT_(object):
    def __init__(self, q_sa, r_plus_vfsp, data, gamma, rmax, rmin,file_name_pre, record: BvftRecord = BvftRecord(), q_type='torch_actor_critic_cont',
                 verbose=False, bins=None, data_size=5000,trajectory_num=276):
        self.data = data                                                        #Data D
        self.gamma = gamma                                                      #gamma
        self.res = 0                                                            #\epsilon k (discretization parameter set)
        self.q_sa_discrete = []                                                 #discreate q function
        self.q_to_data_map = []                                                 # to do
        self.q_size = len(q_sa)                                          #how many (s,a) pairs (q function length)
        self.verbose = verbose                                                  #if true, print log
        if bins is None:
            bins = [2,  4, 5,  7, 8,  10, 11, 12, 16, 19, 22,23]
        self.bins = bins                                                        #used for discretizing Q-values
        self.q_sa = q_sa                                                    #all trajectory q s a
        self.r_plus_vfsp = r_plus_vfsp                                                 #reward
                                      #all q functions
        self.record = record
        self.file_name = file_name_pre
        self.n = data_size


        if self.verbose:
            print(F"Data size = {self.n}")
        self.record.avg_q = [np.sum(qsa) for qsa in self.q_sa]
        self.vmax = np.max(self.q_sa)
        self.vmin = np.min(self.q_sa)


    def discretize(self):                                       #discritization step
        self.q_sa_discrete = []
        self.q_to_data_map = []
        bins = int((self.vmax - self.vmin) / self.res) + 1

        for q in self.q_sa:
            discretized_q = np.digitize(q, np.linspace(self.vmin, self.vmax, bins), right=True) #q belong to which interval
            self.q_sa_discrete.append(discretized_q)
            q_to_data_map = {}
            for i, q_val in enumerate(discretized_q):
                if q_val not in q_to_data_map:
                    q_to_data_map[q_val] = i
                else:
                    if isinstance(q_to_data_map[q_val], int):
                        q_to_data_map[q_val] = [q_to_data_map[q_val]]
                    q_to_data_map[q_val].append(i)
            self.q_to_data_map.append(q_to_data_map)                      #from q value to the position it in discretized_q

    def get_groups(self, q1, q2):
        q1_dic = self.q_to_data_map[q1]
        q2_inds, q2_dic = self.q_sa_discrete[q2], self.q_to_data_map[q2] #dic: indices from q value in the map
        groups = []
        for key in q1_dic:
            if isinstance(q1_dic[key], list):
                q1_list = q1_dic[key]
                set1 = set(q1_list)
                for p1 in q1_list:
                    if p1 in set1 and isinstance(q2_dic[q2_inds[p1]], list):
                        set2 = set(q2_dic[q2_inds[p1]])
                        intersect = set1.intersection(set2)              #intersection
                        set1 = set1.difference(intersect)                #in set1 but not in intersection
                        if len(intersect) > 1:
                            groups.append(list(intersect))               #piecewise constant function
        return groups


    def compute_loss(self, q1, groups):                                 #
        Tf = self.r_plus_vfsp[q1].copy()
        for group in groups:
            Tf[group] = np.mean(Tf[group])
        diff = self.q_sa[q1] - Tf
        return np.sqrt(np.mean(diff ** 2))  #square loss function

    def get_bins(self, groups):
        group_sizes = [len(g) for g in groups]                                  #group size
        bin_ind = np.digitize(group_sizes, self.bins, right=True)               #categorize each group size to bins
        percent_bins = np.zeros(len(self.bins) + 1)    #total group size
        count_bins = np.zeros(len(self.bins) + 1)      #count of groups in each bin
        for i in range(len(group_sizes)):
            count_bins[bin_ind[i] + 1] += 1
            percent_bins[bin_ind[i] + 1] += group_sizes[i]
        percent_bins[0] = self.n - np.sum(percent_bins)
        count_bins[0] = percent_bins[0]    #groups that do not fit into any of predefined bins
        return percent_bins, count_bins

    def run(self, resolution=1e-2):
        self.res = resolution
        if self.verbose:
            print(F"Being  discretizing outputs of Q functions on batch data with resolution = {resolution}")
        self.discretize()
        if self.verbose:
            print("Starting pairwise comparison")
        percent_histos = []
        count_histos = []
        group_count = []
        loss_matrix = np.zeros((self.q_size, self.q_size))
        for q1 in range(self.q_size):
            for q2 in range(q1, self.q_size):
                groups = self.get_groups(q1, q2)
                # percent_bins, count_bins = self.get_bins(groups)
                # percent_histos.append(percent_bins)
                # count_histos.append(count_bins)
                group_count.append(len(groups))

                loss_matrix[q1, q2] = self.compute_loss(q1, groups)
                # if self.verbose:
                #     print("loss |Q{}; Q{}| = {}".format(q1, q2, loss_matrix[q1, q2]))

                if q1 != q2:
                    loss_matrix[q2, q1] = self.compute_loss(q2, groups)
                    # if self.verbose:
                    #     print("loss |Q{}; Q{}| = {}".format(q2, q1, loss_matrix[q2, q1]))

        # average_percent_bins = np.mean(np.array(percent_histos), axis=0) / self.n
        # average_count_bins = np.mean(np.array(count_histos), axis=0)
        average_group_count = np.mean(group_count)
        if self.verbose:
            print(np.max(loss_matrix, axis=1))
        self.record.resolutions.append(resolution)
        self.record.losses.append(np.max(loss_matrix, axis=1))
        self.record.loss_matrices.append(loss_matrix)
        # self.record.percent_bin_histogram.append(average_percent_bins)
        # self.record.count_bin_histogram.append(average_count_bins)
        self.get_br_ranking()
        self.record.group_counts.append(average_group_count)
        if not os.path.exists("Bvft_Records"):
            os.makedirs("Bvft_Records")
        self.record.save(directory="Bvft_Records",file_prefix=self.file_name)


    def compute_optimal_group_skyline(self):
        groups = self.get_groups(self.q_size-1, self.q_size-1)
        loss = [self.compute_loss(q, groups) for q in range(self.q_size)]
        self.record.optimal_grouping_skyline.append(np.array(loss))

    def compute_e_q_star_diff(self):
        q_star = self.q_sa[-1]
        e_q_star_diff = [np.sqrt(np.mean((q - q_star) ** 2)) for q in self.q_sa[:-1]] + [0.0]
        self.record.e_q_star_diff = np.array(e_q_star_diff)


    def get_br_ranking(self):
        br = [np.sqrt(np.sum((self.q_sa[q] - self.r_plus_vfsp[q]) ** 2)) for q in range(self.q_size)]
        br_rank = np.argsort(br)
        self.record.bellman_residual = br
        self.record.record_ranking(br_rank)
        return br_rank

class Hopper_edi(ABC):

    def __init__(self,device,parameter_list,parameter_name_list,policy_training_parameter_map,method_name_list,self_method_name,batch_size = 32,
                 process_num=9, sa_evaluate_times = 10,
                traj_sa_number = 10000,gamma=0.99,trajectory_num=24,
                 max_timestep = 100, total_select_env_number=1,
                 env_name = "Hopper-v4",k=5,num_runs = 20):
        # self.policy_choose = policy_choose
        self.k = k
        self.sa_evaluate_time = sa_evaluate_times
        self.traj_sa_number = traj_sa_number
        self.target_traj_sa_number = traj_sa_number
        self.target_trajectory_num = trajectory_num
        self.process_num = process_num
        self.device = device
        self.batch_size = batch_size
        self.q_functions = []
        self.method_name_list = method_name_list
        self.max_timestep = max_timestep
        self.env_name = env_name
        self.parameter_list = parameter_list
        self.parameter_name_list = parameter_name_list
        self.unique_numbers = []
        self.env_list = []
        self.q_name_functions = []
        self.policy_total_step = policy_training_parameter_map["policy_total_step"]
        self.policy_episode_step = policy_training_parameter_map["policy_episode_step"]
        self.policy_saving_number = policy_training_parameter_map["policy_saving_number"]
        self.policy_learning_rate = policy_training_parameter_map["policy_learning_rate"]
        self.policy_hidden_layer = policy_training_parameter_map["policy_hidden_layer"]
        self.algorithm_name_list = policy_training_parameter_map["algorithm_name_list"]
        self.policy_list = []
        self.policy_name_list = []
        self.env_name_list = []
        self.data = []
        self.gamma = gamma
        self.self_method_name = self_method_name
        self.trajectory_num = trajectory_num
        self.true_env_num = 0
        self.num_runs = num_runs
        for h in range(len(self.parameter_list)):
            for i in range((len(self.parameter_list[h]))):
                current_env = gymnasium.make(self.env_name)
                name = f"{self.env_name}"
                for param_name, param_value in zip(self.parameter_name_list[h], self.parameter_list[h][i]):
                    setattr(current_env.unwrapped.model.opt, param_name, param_value)
                    name += f"_{param_name}_{str(param_value)}"
                self.env_name_list.append(name)

                # print(current_env.unwrapped.model.opt)
                self.env_list.append(current_env)
        self.para_map = {index: item for index, item in enumerate(self.parameter_list)}
        self.q_sa = []
        self.r_plus_vfsp = []
        self.data_size = 0
        self.active_threads = 0
        self.lock = threading.Lock()
    def get_policy(self,env_index,algorithm_name):
        Policy_operation_folder = "Policy_operation"
        Policy_saving_folder = os.path.join(Policy_operation_folder, "Policy_trained")
        self.create_folder(Policy_saving_folder)

        current_env = self.env_list[env_index]
        policy_folder_name = self.env_name_list[env_index]

        policy_saving_path = os.path.join(Policy_saving_folder, policy_folder_name)

        num_epoch = int(self.policy_total_step / self.policy_episode_step)
        buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=1000000, env=current_env)
        explorer = d3rlpy.algos.ConstantEpsilonGreedy(0.3)
        checkpoint_list = []

        policy_model_name = f"{algorithm_name}_{str(self.policy_total_step)}_{str(self.policy_learning_rate)}_{str(self.policy_hidden_layer)}.d3"
        policy_path = policy_saving_path + "_" + policy_model_name
        policy_path = policy_path[:-3] + "_" + str(self.policy_total_step) + "step.d3"
        policy = d3rlpy.load_learnable(policy_path,device=self.device)
        return policy
    def get_policy_path(self,env_index,algorithm_name):
        Policy_operation_folder = "Policy_operation"
        Policy_saving_folder = os.path.join(Policy_operation_folder, "Policy_trained")
        self.create_folder(Policy_saving_folder)

        current_env = self.env_list[env_index]
        policy_folder_name = self.env_name_list[env_index]

        policy_saving_path = os.path.join(Policy_saving_folder, policy_folder_name)

        policy_model_name = f"{algorithm_name}_{str(self.policy_learning_rate)}_{str(self.policy_hidden_layer)}.d3"
        policy_path = policy_saving_path + "_" + policy_model_name
        policy_path = policy_path[:-3] + "_" + str(self.policy_total_step) + "step.d3"
        return policy_path

    def get_policy_name(self,env_index,algorithm_name):


        current_env = self.env_list[env_index]
        policy_folder_name = self.env_name_list[env_index]


        num_epoch = int(self.policy_total_step / self.policy_episode_step)

        policy_model_name = f"{algorithm_name}_{str(self.policy_learning_rate)}_{str(self.policy_hidden_layer)}.d3"
        policy_path = self.env_name_list[env_index]+ "_" + policy_model_name
        policy_path = policy_path[:-3] + "_" + str(self.policy_total_step) + "step.d3"
        return policy_path

    def delete_files_in_folder_r(self, folder_path):
        if not os.path.exists(folder_path):
            print("The folder does not exist.")
            return
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    if not re.match(r'.*\d{1}\.pkl$', filename):
                        os.unlink(file_path)
                        print(f"Deleted {file_path}")
                    else:
                        print(f"Skipping {file_path} (matches the pattern of ending with a number and '.pkl')")
                elif os.path.isdir(file_path):
                    print(f"Skipping directory {file_path}")
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    def delete_as_pkl(self,path):
        os.unlink(path+".pkl")
    def delete_files_in_folder(self,folder_path):
        if not os.path.exists(folder_path):
            print("The folder does not exist.")
            return
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    print(f"Deleted {file_path}")
                elif os.path.isdir(file_path):
                    print(f"Skipping directory {file_path}")
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    def create_folder(self,folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    def save_as_pkl(self,file_path, list_to_save):
        full_path = f"{file_path}.pkl"
        with open(full_path, 'wb') as file:
            pickle.dump(list_to_save, file)

    def save_as_txt(self,file_path, list_to_save):
        full_path = f"{file_path}.txt"
        with open(full_path, 'w') as file:
            for item in list_to_save:
                file.write(f"{item}\n")

    def save_dict_as_txt(self,file_path, dict_to_save):
        full_path = f"{file_path}.txt"
        with open(full_path, 'w') as file:
            for key, value in dict_to_save.items():
                file.write(f"{key}:{value}\n")

    def remove_duplicates(self,lst):
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def load_dict_from_txt(self,file_path):
        with open(file_path, 'r') as file:
            return {line.split(':', 1)[0]: line.split(':', 1)[1].strip() for line in file}

    def list_to_dict(self,name_list, reward_list):
        return dict(zip(name_list, reward_list))

    def load_from_pkl(self,file_path):
        full_path = f"{file_path}.pkl"
        with open(full_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def whether_file_exists(self,checkpoint_path):
        if not os.path.exists(checkpoint_path):
            return False
        return True



    def generate_unique_numbers(self,n, range_start, range_end):
        if n > (range_end - range_start + 1):
            raise ValueError("The range is too small to generate the required number of unique numbers.")

        unique_numbers = random.sample(range(range_start, range_end + 1), n)
        return unique_numbers


    @abstractmethod
    def select_Q(self,q_list,r_plus_vfsp,policy_namei):
        pass
    def remove_duplicates(self,lst):
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    def generate_unique_colors(self,number_of_colors):

        cmap = plt.get_cmap('tab20')

        if number_of_colors <= 20:
            colors = [cmap(i) for i in range(number_of_colors)]
        else:
            colors = [cmap(i) for i in np.linspace(0, 1, number_of_colors)]
        return colors
    def pick_policy(self):
        Policy_operation_folder = "Policy_operation"
        Policy_trained_folder = os.path.join(Policy_operation_folder, "Policy_trained")

        policy_name_list = []
        policy_list = []

        for file_name in os.listdir(Policy_trained_folder):
            if file_name.endswith(".d3"):
                policy_name = file_name  # Remove the last three characters ".d3"

                if policy_name not in self.policy_name_list:
                    self.policy_name_list.append(policy_name[:-3])

                    policy_path = os.path.join(Policy_trained_folder, file_name)
                    policy = d3rlpy.load_learnable(policy_path, device=self.device)

                    self.policy_list.append(policy)

    def find_prefix_suffix(self,folder_path, prefix, suffix):
        if not os.path.exists(folder_path):
            raise ValueError(f"The folder path {folder_path} does not exist.")

        for file_name in os.listdir(folder_path):
            if file_name.startswith(prefix) and file_name.endswith(suffix):
                return file_name

        return None

    def find_folder_prefix_suffix(self,folder_path, prefix, suffix):

        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path) and item.startswith(prefix) and item.endswith(suffix):
                return item
        return None

    def find_position(self,lst, item):
        try:
            return lst.index(item)
        except ValueError:
            return -1
    def get_data_ranking(self, data_address, policy_name_list, true_env_number,true_policy_number):
        Offline_data_folder = "Offline_data"
        true_env_name = self.env_name_list[true_env_number]
        prefix =  f"target_{self.target_trajectory_num}_trajectories_{self.target_traj_sa_number}_sa"

        ranking_data_folder = os.path.join("Exp_result", f"{true_env_name}")

        data_folder = os.path.join(Offline_data_folder, f"{true_env_name}_Policy_{self.policy_name_list[true_policy_number]}")
        seeds = self.load_from_pkl(os.path.join(data_folder,self.find_prefix_suffix(data_folder, prefix, "seeds.pkl")[:-4]))
        ranking_folder = os.path.join(ranking_data_folder,
                                    self.find_folder_prefix_suffix(ranking_data_folder, prefix, data_address))
        policy_performance = []
        for i in range(len(policy_name_list)):
            ranking_path = os.path.join(ranking_folder,self.policy_name_list[i])
            env_name = self.load_from_pkl(ranking_path)
            policy = self.load_policy(i)
            policy_performance.append(self.evaluate_policy_on_seeds(policy=policy,env=self.env_list[self.find_position(self.env_name_list,env_name)],seeds=seeds))


        # Return the ranking of the policy names
        return self.rank_elements_larger_higher(policy_performance)

    def evaluate_policy_on_seeds(self, policy, env, seeds):
        total_rewards = 0

        for i in range(self.num_runs):
            for seed in seeds:
                rewards = self.evaluate_policy_on_seed(policy, env, int(seed))
                total_rewards += rewards


        return total_rewards / len(seeds) / self.num_runs

    def evaluate_policy_on_seed(self, policy, env, seed):
        env.reset(seed=seed)
        obs, info = env.reset(seed=seed)

        total_rewards = 0
        discount_factor = 1
        max_iteration = self.max_timestep

        for _ in range(max_iteration):
            action = policy.predict(np.array([obs]))[0]
            ui = env.step(action)
            obs, reward, done = ui[0], ui[1],ui[2]
            total_rewards += reward * discount_factor
            discount_factor *= self.gamma

            if done:
                break

        return total_rewards

    def load_policy_performance(self, policy_name, true_env_index, true_policy_index):

        policy_performance_folder = os.path.join("Policy_operation", "Policy_performance",
                                                 f"{self.env_name_list[true_env_index]}_Policy_{self.policy_name_list[true_policy_index]}")
        self.create_folder(policy_performance_folder)
        policy_performance_path = os.path.join(policy_performance_folder, "performance")

        if os.path.exists(policy_performance_path):
            performance_map = self.load_from_pkl(policy_performance_path)
            if policy_name in performance_map:
                return performance_map[policy_name]

        self.get_policy_performance(true_env_index=true_env_index, true_policy_index=true_policy_index)
        performance_map = self.load_from_pkl(policy_performance_path)
        return performance_map.get(policy_name, None)
    def calculate_top_k_precision(self, true_env_index,true_policy_index, policy_name_list, rank_list, k=2):

        device = self.device

        policy_performance_list = [self.load_policy_performance(policy_name=policy_name,true_env_index=true_env_index,true_policy_index=true_policy_index) for policy_name in policy_name_list]
        policy_ranking_groundtruth = self.rank_elements_larger_higher(policy_performance_list)

        k_precision_list = []
        for i in range(1, k + 1):
            proportion = 0
            for pos in rank_list:
                if (rank_list[pos - 1] <= i - 1 and policy_ranking_groundtruth[pos - 1] <= i - 1):
                    proportion += 1
            proportion = proportion / i
            k_precision_list.append(proportion)
        return k_precision_list


    def calculate_top_k_normalized_regret(self,ranking_list, policy_name_list, true_env_index,true_policy_index, k=2):
        print("calcualte top k normalized regret")
        policy_performance_list = [self.load_policy_performance(policy_name=policy_name,true_env_index=true_env_index,true_policy_index=true_policy_index) for policy_name in policy_name_list]

        ground_truth_value = max(policy_performance_list)
        worth_value = min(policy_performance_list)
        if ((ground_truth_value - worth_value) == 0):
            return 99999
        k_regret_list = []
        for j in range(1, k + 1):
            gap_list = []
            for i in range(len(ranking_list)):
                if (ranking_list[i] <= j):
                    value = policy_performance_list[i]
                    norm = (ground_truth_value - value) / (ground_truth_value - worth_value)
                    gap_list.append(norm)
            k_regret_list.append(min(gap_list))
        return k_regret_list
    def calculate_statistics(self,data_list):
        mean = np.mean(data_list)
        std_dev = np.std(data_list, ddof=1)
        sem = std_dev / np.sqrt(len(data_list))
        ci = 2 * sem
        return mean, ci
    def calculate_k(self, true_data_list,k,plot_name_list):
        Ranking_list = []
        Policy_name_list = []

        run_list = []
        for i in range(len(true_data_list)):
            for j in range(len(self.policy_list)):
                run_list.append([i,j])

        data_address_lists = self.remove_duplicates(self.method_name_list)

        for i in range(len(data_address_lists)):
            Ranking_list.append([])

        for runs in range(len(run_list)):
            self.pick_policy()
            Policy_name_list.append(self.policy_name_list)

            for data_address_index in range(len(data_address_lists)):
                Ranking_list[data_address_index].append(
                    self.get_data_ranking(data_address_lists[data_address_index], self.policy_name_list,run_list[runs][0],run_list[runs][1]))


        Precision_list = []
        Regret_list = []
        for index in range(len(data_address_lists)):
            Precision_list.append([])
            Regret_list.append([])

        for i in range(len(run_list)):
            for num_index in range(len(Ranking_list)):
                Precision_list[num_index].append(
                    self.calculate_top_k_precision(true_env_index=run_list[i][0],true_policy_index=run_list[i][1],
                                                   policy_name_list=Policy_name_list[i],rank_list=Ranking_list[num_index][i],k=k))
                Regret_list[num_index].append(
                    self.calculate_top_k_normalized_regret(ranking_list = Ranking_list[num_index][i],
                                                           policy_name_list=Policy_name_list[i],
                                                    true_env_index=run_list[i][0],
                                                           true_policy_index=run_list[i][1],
                                                           k=k))

        Precision_k_list = []
        Regret_k_list = []
        for iu in range(len(Ranking_list)):
            Precision_k_list.append([])
            Regret_k_list.append([])

        for i in range(k):
            for ku in range(len(Ranking_list)):
                k_precision = []
                k_regret = []
                for j in range(len(run_list)):
                    k_precision.append(Precision_list[ku][j][i])
                    k_regret.append(Regret_list[ku][j][i])
                Precision_k_list[ku].append(k_precision)
                Regret_k_list[ku].append(k_regret)

        precision_mean_list = []
        regret_mean_list = []
        precision_ci_list = []
        regret_ci_list = []
        for i in range(len(Precision_list)):
            current_precision_mean_list = []
            current_regret_mean_list = []
            current_precision_ci_list = []
            current_regret_ci_list = []
            for j in range(k):
                current_precision_mean, current_precision_ci = self.calculate_statistics(Precision_k_list[i][j])
                current_regret_mean, current_regret_ci = self.calculate_statistics(Regret_k_list[i][j])
                current_precision_mean_list.append(current_precision_mean)
                current_precision_ci_list.append(current_precision_ci)
                current_regret_mean_list.append(current_regret_mean)
                current_regret_ci_list.append(current_regret_ci)
            precision_mean_list.append(current_precision_mean_list)
            regret_mean_list.append(current_regret_mean_list)
            precision_ci_list.append(current_precision_ci_list)
            regret_ci_list.append(current_regret_ci_list)
        policy_ranking_saving_place = 'Policy_operation'
        k_saving_folder = 'K_statistics'
        saving_folder = os.path.join(policy_ranking_saving_place, k_saving_folder)
        saving_path = os.path.join(saving_folder,
                                          f"target_{self.target_trajectory_num}_trajectories_{self.target_traj_sa_number}")

        self.create_folder(saving_path)
        # Bvft_k_save_path = os.path.join(Bvft_saving_place, Bvft_k)
        # if not os.path.exists(Bvft_k_save_path):
        #     os.makedirs(Bvft_k_save_path)
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        plot_name = "plots"
        k_precision_name = str(k) + "_mean_precision_" + str(len(run_list))
        k_regret_name = str(k) + "_mean_regret" + str(len(run_list))
        precision_ci_name = str(k) + "_CI_precision" + str(len(run_list))
        regret_ci_name = str(k) + "_CI_regret" + str(len(run_list))

        k_precision_mean_saving_path = os.path.join(saving_path, k_precision_name)
        k_regret_mean_saving_path = os.path.join(saving_path, k_regret_name)
        k_precision_ci_saving_path = os.path.join(saving_path, precision_ci_name)
        k_regret_ci_saving_path = os.path.join(saving_path, regret_ci_name)
        plot_name_saving_path = os.path.join(saving_path, plot_name)

        self.save_as_pkl(k_precision_mean_saving_path, precision_mean_list)
        self.save_as_pkl(k_regret_mean_saving_path, regret_mean_list)
        self.save_as_pkl(k_precision_ci_saving_path, precision_ci_list)
        self.save_as_pkl(k_regret_ci_saving_path, regret_ci_list)
        self.save_as_pkl(plot_name_saving_path, plot_name_list)

        self.save_as_txt(k_precision_mean_saving_path, precision_mean_list)
        self.save_as_txt(k_regret_mean_saving_path, regret_mean_list)
        self.save_as_txt(k_precision_ci_saving_path, precision_ci_list)
        self.save_as_txt(k_regret_ci_saving_path, regret_ci_list)
        self.save_as_txt(plot_name_saving_path, plot_name_list)

        return precision_mean_list, regret_mean_list, precision_ci_list, regret_ci_list, plot_name_list

    def draw_figure_6L(self,true_data_list):
        Result_saving_place = 'Policy_operation'
        Result_k = 'K_statistics'
        Result_k_save_folder = os.path.join(Result_saving_place, Result_k)
        Result_k_save_path = os.path.join(Result_k_save_folder,
                                          f"target_{self.target_trajectory_num}_trajectories_{self.target_traj_sa_number}")

        self.create_folder(Result_k_save_path)
        num_runs = len(true_data_list) * len(self.policy_list)

        k_precision_name = str(self.k) + "_mean_precision_" + str(num_runs)
        k_regret_name = str(self.k) + "_mean_regret_" + str(num_runs)
        precision_ci_name = str(self.k) + "_CI_precision_" + str(num_runs)
        regret_ci_name = str(self.k) + "_CI_regret_" + str(num_runs)
        plot_name = "plots"

        k_precision_mean_saving_path = os.path.join(Result_k_save_path, k_precision_name)
        k_regret_mean_saving_path = os.path.join(Result_k_save_path, k_regret_name)
        k_precision_ci_saving_path = os.path.join(Result_k_save_path, precision_ci_name)
        k_regret_ci_saving_path = os.path.join(Result_k_save_path, regret_ci_name)
        plot_name_saving_path = os.path.join(Result_k_save_path, plot_name)

        if os.path.exists(k_precision_mean_saving_path):
            precision_mean_list = self.load_from_pkl(k_precision_mean_saving_path)
            regret_mean_list = self.load_from_pkl(k_regret_mean_saving_path)
            precision_ci_list = self.load_from_pkl(k_precision_ci_saving_path)
            regret_ci_list = self.load_from_pkl(k_regret_ci_saving_path)
            line_name_list = self.load_from_pkl(plot_name_saving_path)
        else:
            precision_mean_list, regret_mean_list, precision_ci_list, regret_ci_list, line_name_list = self.calculate_k(
                true_data_list = true_data_list, k = self.k,plot_name_list = self.method_name_list
            )

        plot_mean_list = [precision_mean_list, regret_mean_list]
        plot_ci_list = [precision_ci_list, regret_ci_list]

        plot_folder = os.path.join(Result_k_save_path, "Figure_6L_plot")
        self.create_folder(plot_folder)

        y_axis_names = ["k precision", "k regret"]
        colors = self.generate_unique_colors(len(plot_mean_list[0]))

        self.plot_subplots(data=plot_mean_list, save_path=plot_folder, y_axis_names=y_axis_names,
                           line_names=line_name_list, colors=colors, ci=plot_ci_list)

    def plot_subplots(self, data, save_path, y_axis_names, line_names, colors, ci):
        num_subplots = len(data)
        fig, axes = plt.subplots(num_subplots, figsize=(10, 5 * num_subplots), squeeze=False)

        for i, subplot_data in enumerate(data):
            for j, line_data in enumerate(subplot_data):
                x_values = list(range(1, len(line_data) + 1))

                top = []
                bot = []

                for z in range(len(line_data)):
                    top.append(line_data[z] + ci[i][j][z])
                    bot.append(line_data[z] - ci[i][j][z])

                axes[i, 0].plot(x_values, line_data, label=line_names[j], color=colors[j])
                axes[i, 0].fill_between(x_values, bot, top, color=colors[j], alpha=0.2)

            axes[i, 0].set_ylabel(y_axis_names[i])
            axes[i, 0].legend()

        plt.tight_layout()
        saving_path = os.path.join(save_path, "regret_precision.png")
        plt.savefig(saving_path)
        plt.close()

    def load_policy(self, policy_index):
        Policy_operation = "Policy_operation"
        Policy_trained = "Policy_trained"
        policy_folder = os.path.join(Policy_operation, Policy_trained)
        self.create_folder(policy_folder)
        policy_name = self.policy_name_list[policy_index] + ".d3"
        policy_path = os.path.join(policy_folder, policy_name)
        return d3rlpy.load_learnable(policy_path, device=self.device)

    def generate_one_trajectory(self, env_number, max_time_step, policy_index, unique_seed):
        policy = self.load_policy(policy_index)
        env = self.env_list[env_number]
        obs, info = env.reset(seed=unique_seed)

        observations = []
        rewards = []
        actions = []
        dones = []
        next_steps = []
        episode_data = {}
        observations.append(obs)
        total_state_number = 1  # Initialize the state counter

        for _ in range(max_time_step):
            action = policy.predict(np.array([obs]))[0]
            ui = env.step(action)
            next_obs, reward, done = ui[0],ui[1],ui[2]
            rewards.append(reward)
            actions.append(action)
            dones.append(done)
            next_steps.append(next_obs)
            total_state_number += 1

            if done or _ == max_time_step - 1:
                break

            obs = next_obs
            observations.append(obs)

        episode_data = {
            "observations": observations,
            "rewards": rewards,
            "actions": actions,
            "dones": dones,
            "next_steps": next_steps,
            "total_state_number": total_state_number
        }

        return episode_data
    def desire_exists(self,data_folder):
        existing_files = [f[:-4] for f in os.listdir(data_folder) if
                          not f.endswith(('q.pkl', 'r.pkl', 'seeds.pkl', 'size.pkl'))]
        for file in existing_files:
            parts = file.split('_')
            if int(parts[1]) == self.target_trajectory_num and int(parts[3]) == self.target_traj_sa_number:
                return True
        return False
    def load_files(self,data_folder):
        existing_files = [f[:-4] for f in os.listdir(data_folder) if
                          not f.endswith(('q.pkl', 'r.pkl', 'seeds.pkl', 'size.pkl'))]
        result = ""
        for file in existing_files:
            parts = file.split('_')
            if int(parts[1]) == self.target_trajectory_num and int(parts[3]) == self.target_traj_sa_number:
                return file



    def load_offline_data(self, max_time_step, policy_index, true_env_number):
        self.print_environment_parameters()
        self.true_env_num = true_env_number
        Offline_data_folder = "Offline_data"
        self.create_folder(Offline_data_folder)
        data_folder_name = f"{self.env_name_list[true_env_number]}_Policy_{self.policy_name_list[policy_index]}"
        data_folder = os.path.join(Offline_data_folder, data_folder_name)
        self.create_folder(data_folder)


        if self.desire_exists(data_folder):
            file_name = self.load_files(data_folder)
            data_path = os.path.join(data_folder, file_name)
            data_seeds_name = file_name + "_seeds"
            data_seeds_path = os.path.join(data_folder, data_seeds_name)
            self.data = self.load_from_pkl(data_path)
            self.data_size = self.data.size
            self.unique_numbers = self.load_from_pkl(data_seeds_path)
            self.q_sa = [np.zeros(self.data.size) for _ in range(len(self.env_list) * len(self.policy_name_list))]
            self.r_plus_vfsp = [np.zeros(self.data.size) for _ in range(len(self.env_list) * len(self.policy_name_list))]
        else:
            self.generate_offline_data(max_time_step, policy_index, true_env_number)
    def generate_all_offline_data(self):
        self.train_policy()
        Offline_data_folder = "Offline_data"
        self.create_folder(Offline_data_folder)
        for i in range(len(self.env_list)):
            for j in range(len(self.policy_list)):
                self.get_policy_performance(true_env_index=i,true_policy_index=j)
                data_folder_name = f"{self.env_name_list[i]}_Policy_{self.policy_name_list[j]}"
                data_folder = os.path.join(Offline_data_folder, data_folder_name)
                self.create_folder(data_folder)

                existing_files = [f[:-4] for f in os.listdir(data_folder) if
                                  not f.endswith(('q.pkl', 'r.pkl', 'seeds.pkl', 'size.pkl'))]
                total_trajectory_num = 0
                total_sa_num = 0
                final_data = []
                self.unique_numbers = []

                # Merge existing files
                exist = False
                for file in existing_files:
                    parts = file.split('_')
                    if int(parts[1]) == self.target_trajectory_num and int(parts[3]) == self.target_traj_sa_number:
                        exist = True
                if not exist:
                    dudu = f"target_{self.target_trajectory_num}_trajectories_{self.target_traj_sa_number}_"
                    dudu_path = os.path.join(data_folder,dudu)
                    self.save_as_pkl(dudu_path,"1")
                    #
                    # for file in existing_files:
                    #     parts = file.split('_')
                    #     traj_num = int(parts[6])
                    #     sa_num = int(parts[8])
                    #     total_trajectory_num += traj_num
                    #     total_sa_num += sa_num
                    #     print(file)
                    #     data = self.load_from_pkl(os.path.join(data_folder, file)).dataset
                    #     final_data.extend(data)
                    #     seeds = self.load_from_pkl(os.path.join(data_folder, f"{file}_seeds"))
                    #     self.unique_numbers.extend(seeds)
                    #
                    #     if total_sa_num >= self.target_traj_sa_number and total_trajectory_num >= self.target_trajectory_num:
                    #         break

                    # If not satisfied, keep generating

                    while total_sa_num < self.target_traj_sa_number or total_trajectory_num < self.target_trajectory_num:
                        new_seed = self.generate_unique_seed()
                        new_trajectory = self.generate_one_trajectory(i, self.max_timestep, j,
                                                                      new_seed)
                        final_data.append(new_trajectory)
                        self.unique_numbers.append(new_seed)
                        total_sa_num += new_trajectory["total_state_number"]
                        total_trajectory_num += 1
                    self.trajectory_num = total_trajectory_num
                    self.traj_sa_number = total_sa_num

                    trajectory_saving_name = f"target_{self.target_trajectory_num}_trajectories_{self.target_traj_sa_number}_sa_normal_{total_trajectory_num}_trajectories_{total_sa_num}_sa"
                    trajectory_data_path = os.path.join(data_folder, trajectory_saving_name)
                    initial_state_seeds_path = os.path.join(data_folder, f"{trajectory_saving_name}_seeds")

                    self.data = CustomDataLoader(final_data)
                    self.data_size = self.data.size
                    self.q_sa = [np.zeros(self.data.size) for _ in
                                 range(len(self.env_list) * len(self.policy_name_list))]
                    self.r_plus_vfsp = [np.zeros(self.data.size) for _ in
                                        range(len(self.env_list) * len(self.policy_name_list))]
                    # print(self.data.dataset)
                    self.save_as_pkl(trajectory_data_path, self.data)
                    self.save_as_pkl(initial_state_seeds_path, self.unique_numbers)
                    self.delete_as_pkl(dudu_path)





    def generate_offline_data(self, max_time_step, policy_index, true_env_number):
        self.print_environment_parameters()
        self.true_env_num = true_env_number
        Offline_data_folder = "Offline_data"
        data_folder_name = f"{self.env_name_list[true_env_number]}_Policy_{self.policy_name_list[policy_index]}"
        data_folder = os.path.join(Offline_data_folder, data_folder_name)
        self.create_folder(data_folder)

        existing_files = [f[:-4] for f in os.listdir(data_folder) if not f.endswith(('q.pkl', 'r.pkl', 'seeds.pkl','size.pkl'))]
        total_trajectory_num = 0
        total_sa_num = 0
        final_data = []
        self.unique_numbers = []

        # Merge existing files
        for file in existing_files:
            parts = file.split('_')
            traj_num = int(parts[6])
            sa_num = int(parts[8])
            total_trajectory_num += traj_num
            total_sa_num += sa_num
            print(file)
            data = self.load_from_pkl(os.path.join(data_folder, file)).dataset
            final_data.extend(data)
            seeds = self.load_from_pkl(os.path.join(data_folder, f"{file}_seeds"))
            self.unique_numbers.extend(seeds)

            if total_sa_num >= self.target_traj_sa_number and total_trajectory_num >= self.target_trajectory_num:
                break

        # If not satisfied, keep generating

        while total_sa_num < self.target_traj_sa_number or total_trajectory_num < self.target_trajectory_num:
            new_seed = self.generate_unique_seed()
            new_trajectory = self.generate_one_trajectory(true_env_number, max_time_step, policy_index, new_seed)
            final_data.append(new_trajectory)
            self.unique_numbers.append(new_seed)
            total_sa_num += new_trajectory["total_state_number"]
            total_trajectory_num += 1
        self.trajectory_num = total_trajectory_num
        self.traj_sa_number = total_sa_num

        trajectory_saving_name = f"target_{self.target_trajectory_num}_trajectories_{self.target_traj_sa_number}_sa_normal_{total_trajectory_num}_trajectories_{total_sa_num}_sa"
        trajectory_data_path = os.path.join(data_folder, trajectory_saving_name)
        initial_state_seeds_path = os.path.join(data_folder, f"{trajectory_saving_name}_seeds")

        self.data = CustomDataLoader(final_data)
        self.data_size = self.data.size
        self.q_sa = [np.zeros(self.data.size) for _ in range(len(self.env_list) * len(self.policy_name_list))]
        self.r_plus_vfsp = [np.zeros(self.data.size) for _ in range(len(self.env_list) * len(self.policy_name_list))]
        # print(self.data.dataset)
        self.save_as_pkl(trajectory_data_path, self.data)
        self.save_as_pkl(initial_state_seeds_path, self.unique_numbers)
        print("after saving")

    def merge_trajectories(self, trajectories):
        merged_data = {
            "observations": [],
            "rewards": [],
            "actions": [],
            "dones": [],
            "next_steps": [],
            "total_state_number": 0
        }

        for trajectory in trajectories:
            merged_data["observations"].extend(trajectory["observations"])
            merged_data["rewards"].extend(trajectory["rewards"])
            merged_data["actions"].extend(trajectory["actions"])
            merged_data["dones"].extend(trajectory["dones"])
            merged_data["next_steps"].extend(trajectory["next_steps"])
            merged_data["total_state_number"] += trajectory["total_state_number"]

        return merged_data

    def generate_unique_seed(self):
        return np.random.randint(0, 1000000)


    def rank_elements_larger_higher(self,lst):
        sorted_pairs = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
        ranks = [0] * len(lst)
        for rank, (original_index, _) in enumerate(sorted_pairs, start=1):
            ranks[original_index] = rank
        return ranks

    def rank_elements_lower_higher(self,lst):
        sorted_pairs = sorted(enumerate(lst), key=lambda x: x[1], reverse=False)
        ranks = [0] * len(lst)
        for rank, (original_index, _) in enumerate(sorted_pairs, start=1):
            ranks[original_index] = rank
        return ranks

    def print_environment_parameters(self):
        print(f"{self.parameter_name_list} parameters of environments in current class :")
        for key, value in self.para_map.items():
            print(f"{key}: {value}")

    def train_policy(self):
        Policy_operation_folder = "Policy_operation"
        Policy_saving_folder = os.path.join(Policy_operation_folder,"Policy_trained")
        self.create_folder(Policy_saving_folder)
        Policy_checkpoints_folder = os.path.join(Policy_operation_folder,"Policy_checkpoints")
        self.create_folder(Policy_checkpoints_folder)
        # while(True):
        for i in range(len(self.env_name_list)):
            current_env = self.env_list[i]
            policy_folder_name = self.env_name_list[i]

            print(f"start training {policy_folder_name} with algorithm {str(self.algorithm_name_list)}")
            policy_saving_path = os.path.join(Policy_saving_folder, policy_folder_name)
            policy_checkpoints_path = os.path.join(Policy_checkpoints_folder, policy_folder_name)

            self.create_folder(policy_checkpoints_path)
            num_epoch = int(self.policy_total_step / self.policy_episode_step)
            buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=1000000, env=current_env)
            explorer = d3rlpy.algos.ConstantEpsilonGreedy(0.3)
            checkpoint_list = []
            for algorithm_name in self.algorithm_name_list:
                checkpoint_path = os.path.join(policy_checkpoints_path,f"{algorithm_name}_checkpoints.d3")
                checkpoint_list_path = os.path.join(policy_checkpoints_path, f"{algorithm_name}_checkpoints")
                policy_model_name = f"{algorithm_name}_{str(self.policy_learning_rate)}_{str(self.policy_hidden_layer)}.d3"
                policy_path = policy_saving_path+"_"+policy_model_name
                if(not self.whether_file_exists(policy_path[:-3]+"_"+str(self.policy_total_step)+"step.d3")):
                    print(f"{policy_path} not exists")
                    if (self.whether_file_exists(checkpoint_path)):
                        policy = d3rlpy.load_learnable(checkpoint_path, device=self.device)
                        checkpoint_list = self.load_from_pkl(checkpoint_list_path)
                        print(f"enter self checkpoints {checkpoint_path} with epoch {str(checkpoint_list[-1])}")
                        for epoch in range(checkpoint_list[-1] + 1, int(num_epoch)):
                            policy.fit_online(env=current_env,
                                              buffer=buffer,
                                              explorer=explorer,
                                              n_steps=self.policy_episode_step,
                                              eval_env=current_env,
                                              with_timestamp=False,
                                              )
                            policy.save(checkpoint_path)
                            checkpoint_list.append(epoch)
                            self.save_as_pkl(checkpoint_list_path, checkpoint_list)
                            if ((epoch + 1) % self.policy_saving_number == 0):
                                policy.save(policy_path[:-3] + "_" + str(
                                    (epoch + 1) * self.policy_episode_step) + "step" + ".d3")
                                self.policy_list.append(policy)
                                self.policy_name_list.append(policy_folder_name+"_"+policy_model_name[:-3]+"_"+str(self.policy_total_step)+"step")
                    else:
                        self_class = getattr(d3rlpy.algos, algorithm_name + "Config")
                        policy = self_class(
                            actor_encoder_factory=d3rlpy.models.VectorEncoderFactory(
                                hidden_units=self.policy_hidden_layer),
                            critic_encoder_factory=d3rlpy.models.VectorEncoderFactory(
                                hidden_units=self.policy_hidden_layer),
                            actor_learning_rate=self.policy_learning_rate,
                            critic_learning_rate=self.policy_learning_rate,
                        ).create(device=self.device)
                        for epoch in range(num_epoch):
                            policy.fit_online(env=current_env,
                                              buffer=buffer,
                                              explorer=explorer,
                                              n_steps=self.policy_episode_step,
                                              eval_env=current_env,
                                              with_timestamp=False,
                                              )
                            policy.save(checkpoint_path)
                            checkpoint_list.append(epoch)
                            self.save_as_pkl(checkpoint_list_path, checkpoint_list)
                            if ((epoch + 1) % self.policy_saving_number == 0):
                                policy.save(policy_path[:-3] + "_" + str(
                                    (epoch + 1) * self.policy_episode_step) + "step" + ".d3")
                                self.policy_list.append(policy)
                                self.policy_name_list.append(
                                    policy_folder_name+"_"+policy_model_name[:-3] + "_" + str(self.policy_total_step) + "step")
                    if os.path.exists(checkpoint_list_path + ".pkl"):
                        os.remove(checkpoint_list_path + ".pkl")
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                    print(f"end training {policy_folder_name} with algorithm {str(self.algorithm_name_list)}")
                else:
                    policy_path = policy_path[:-3]+"_"+str(self.policy_total_step)+"step.d3"
                    policy = d3rlpy.load_learnable(policy_path, device=self.device)
                    self.policy_list.append(policy)
                    self.policy_name_list.append(policy_folder_name+"_"+policy_model_name[:-3] + "_" + str(self.policy_total_step) + "step")
                    print("beegin load policy : ",str(policy_path))
            # print("sleep now")
            # time.sleep(600)
    # def get_policy_per(self,policy,environment):
    #     total_rewards = 0
    #     max_iteration = 1000
    #     env = copy.deepcopy(environment)
    #     for num in range(100):
    #         num_step = 0
    #         discount_factor = 1
    #         observation, info = env.reset(seed=12345)
    #         action = policy.predict(np.array([observation]))
    #         ui = env.step(action[0])
    #         state = ui[0]
    #         reward = ui[1]
    #         done = ui[2]
    #         while ((not done) and (num_step < 1000)):
    #             action = policy.predict(np.array([state]))
    #             ui = env.step(action[0])
    #             state = ui[0]
    #             reward = ui[1]
    #             done = ui[2]
    #             total_rewards += reward * discount_factor
    #             discount_factor *= self.gamma
    #             num_step += 1
    #     total_rewards = total_rewards / 100
    #     return total_rewards
    def get_seeds(self,true_env_index,true_policy_index):
        folder = os.path.join("Offline_data",f"{self.env_name_list[true_env_index]}_Policy_{self.policy_name_list[true_policy_index]}")
        seeds_name = self.find_prefix_suffix(folder_path=folder,prefix = f"target_{self.target_trajectory_num}_trajectories_{self.target_traj_sa_number}_sa",suffix="seeds.pkl")
        seed_path = os.path.join(folder,seeds_name)[:-3]
        return self.load_from_pkl(seed_path)



    def get_policy_performance(self,true_env_index,true_policy_index):
        Policy_operation_folder = "Policy_operation"
        Policy_performance = os.path.join(Policy_operation_folder, "Policy_performance")
        Policy_performance_folder = os.path.join(Policy_performance,f"{self.env_name_list[true_env_index]}_Policy_{self.policy_name_list[true_policy_index]}")
        self.create_folder(Policy_performance_folder)


        seeds = self.get_seeds(true_env_index=true_env_index,true_policy_index=true_policy_index)

        performance_folder_name = "performance"
        policy_performance_path = os.path.join(Policy_performance_folder, performance_folder_name)

        # Check if performance.pkl exists, if not initialize an empty dictionary
        if os.path.exists(policy_performance_path + '.pkl'):
            final_result_dict = self.load_from_pkl(policy_performance_path)
        else:
            final_result_dict = {}

        for i in range(len(self.env_list)):
            env = self.env_list[i]
            for algorithm_name in self.algorithm_name_list:
                policy_path = self.get_policy_path(env_index=i, algorithm_name=algorithm_name)
                policy_name = self.get_policy_name(env_index=i, algorithm_name=algorithm_name)[:-3]
                if os.path.exists(policy_path) and policy_name not in final_result_dict:
                    policy = d3rlpy.load_learnable(policy_path, device=self.device)
                    performance = self.evaluate_policy_on_seeds(policy, env,seeds)
                    final_result_dict[policy_name] = performance
        #print(final_result_dict)
        self.save_as_pkl(policy_performance_path, final_result_dict)
        self.save_as_txt(policy_performance_path, final_result_dict)



    def run_simulation(self, state_action_policy_env_batch):
        dudu_time = time.time()
        # print("run simulation batch size 100 one time : ",dudu_time)
        states, actions, policy, envs = state_action_policy_env_batch
        # Initialize environments with states
        for env, state in zip(envs, states):
            env.reset()
            env.observation = state

        total_rewards = np.zeros(len(states))
        num_steps = np.zeros(len(states))
        discount_factors = np.ones(len(states))
        done_flags = np.zeros(len(states), dtype=bool)

        while not np.all(done_flags) and np.any(num_steps < self.max_timestep):
            actions_batch = policy.predict(np.array(states))
            for idx, (env, action) in enumerate(zip(envs, actions_batch)):
                if not done_flags[idx]:
                    next_state, reward, done, _, _ = env.step(action)
                    total_rewards[idx] += reward * discount_factors[idx]
                    discount_factors[idx] *= self.gamma
                    num_steps[idx] += 1
                    states[idx] = next_state
                    done_flags[idx] = done
        # print("finish one simulation batch size 100 run simulation time :",time.time()-dudu_time)
        # sys.exit()

        return total_rewards

    def get_qa(self, policy_number, env_copy_list, states, actions, batch_size=8):
        policy = self.policy_list[policy_number]
        total_len = len(states)
        results = []

        for i in range(0, total_len, batch_size):
            actual_batch_size = min(batch_size, total_len - i)
            state_action_policy_env_pairs = (
                states[i:i + actual_batch_size], actions[i:i + actual_batch_size], policy,
                env_copy_list[:actual_batch_size])
            batch_results = self.run_simulation(state_action_policy_env_pairs)
            results.extend(batch_results)

        return np.array(results)

    def create_deep_copies(self, env, batch_size):
        return [copy.deepcopy(env) for _ in range(batch_size)]

    def process_env_policy_combination(self, env_index, policy_index, policy_list, env_list, data, gamma, batch_size,
                                       trajectory_num, data_size):
        logging.info(f"Starting processing for env_index {env_index}, policy_index {policy_index}")
        start_time = time.time()

        q_sa = np.zeros(data_size)
        r_plus_vfsp = np.zeros(data_size)
        local_data_size = 0

        env = env_list[env_index]
        env_copy_list = [copy.deepcopy(env) for _ in range(batch_size)]
        logging.info(f"Memory usage after creating environment copies: {memory_usage()} MB")

        ptr = 0
        trajectory_length = 0
        while ptr < trajectory_num:
            length = data.get_iter_length(ptr)
            state, action, next_state, reward, done = data.sample(ptr)

            q_values = self.get_qa(policy_index, env_copy_list, state, action, batch_size)
            if q_values.shape[0] != length:
                raise ValueError(f"Shape mismatch: q_values.shape[0]={q_values.shape[0]}, length={length}")

            q_sa[trajectory_length:trajectory_length + length] = q_values

            vfsp_values = (reward + self.get_qa(policy_index, env_copy_list, next_state,
                                                policy_list[policy_index].predict(next_state), batch_size) *
                           (1 - np.array(done)) * gamma)
            if vfsp_values.shape[0] != length:
                raise ValueError(f"Shape mismatch: vfsp_values.shape[0]={vfsp_values.shape[0]}, length={length}")

            r_plus_vfsp[trajectory_length:trajectory_length + length] = vfsp_values.flatten()[:length]
            trajectory_length += length
            ptr += 1

        local_data_size += trajectory_length

        end_time = time.time()
        logging.info(
            f"Finished processing for env_index {env_index}, policy_index {policy_index} in {end_time - start_time} seconds")
        logging.info(
            f"Memory usage after processing env_index {env_index}, policy_index {policy_index}: {memory_usage()} MB")

        return q_sa, r_plus_vfsp, local_data_size

    def get_whole_qa(self, env_index, policy_index):
        Offline_data_folder = "Offline_data"
        self.create_folder(Offline_data_folder)
        data_folder_name = f"{self.env_name_list[env_index]}_Policy_{self.policy_name_list[policy_index]}"
        data_folder = os.path.join(Offline_data_folder,data_folder_name)
        self.create_folder(data_folder)
        trajectory_name = f"target_{self.target_trajectory_num}_trajectories_{self.target_traj_sa_number}_sa_normal_{self.trajectory_num}_trajectory_{self.traj_sa_number}_sa"

        data_q_name = trajectory_name + "_q"
        data_q_path = os.path.join(data_folder, data_q_name)
        data_r_name = trajectory_name + "_r"
        data_r_path = os.path.join(data_folder, data_r_name)
        data_size_name = trajectory_name + "_size"
        data_size_path = os.path.join(data_folder, data_size_name)

        if not self.whether_file_exists(data_q_path + ".pkl"):
            logging.info("Enter get qa calculate loop")
            start_time = time.time()

            threading_start_time = time.time()
            print("self process num: ", self.process_num)
            # results = []
            # for iteration in range(self.sa_evaluate_time):
            #     for i in range(len(self.env_list)):
            #         for j in range(len(self.policy_list)):
            #             result = self.process_env_policy_combination(
            #                 i, j, self.policy_list, self.env_list, self.data, self.gamma, self.batch_size,
            #                 self.trajectory_num, self.data.size)
            #             results.append(result)

            # q_sa_aggregated = [np.zeros(self.data.size) for _ in range(len(self.env_list) * len(self.policy_list))]
            # r_plus_vfsp_aggregated = [np.zeros(self.data.size) for _ in
            #                           range(len(self.env_list) * len(self.policy_list))]
            # data_size_aggregated = 0
            with Pool(self.process_num) as pool:
                results = []
                for iteration in range(self.sa_evaluate_time):
                    results.extend([pool.apply_async(self.process_env_policy_combination, args=(
                        i, j, self.policy_list, self.env_list, self.data, self.gamma, self.batch_size,
                        self.trajectory_num, self.data.size))
                        for i in range(len(self.env_list)) for j in range(len(self.policy_list))])

                q_sa_aggregated = [np.zeros(self.data.size) for _ in range(len(self.env_list) * len(self.policy_list))]
                r_plus_vfsp_aggregated = [np.zeros(self.data.size) for _ in range(len(self.env_list) * len(self.policy_list))]
                data_size_aggregated = 0

                for idx, result in enumerate(results):
                    q_sa_partial, r_plus_vfsp_partial, data_size_partial = result.get()
                    index = idx % (len(self.env_list) * len(self.policy_list))
                    q_sa_aggregated[index] += q_sa_partial
                    r_plus_vfsp_aggregated[index] += r_plus_vfsp_partial
                    data_size_aggregated += data_size_partial

                q_sa_aggregated = [q_sa / self.sa_evaluate_time for q_sa in q_sa_aggregated]
                r_plus_vfsp_aggregated = [r_plus_vfsp / self.sa_evaluate_time for r_plus_vfsp in r_plus_vfsp_aggregated]

            self.q_sa = q_sa_aggregated
            self.r_plus_vfsp = r_plus_vfsp_aggregated
            self.data_size = data_size_aggregated

            threading_end_time = time.time()
            logging.info(f"Threading (env only) time: {threading_end_time - threading_start_time} seconds")
            logging.info(f"Memory usage after processing all environments: {memory_usage()} MB")

            end_time = time.time()
            logging.info(f"Total running time get_qa: {end_time - start_time} seconds")

            self.save_as_pkl(data_q_path, self.q_sa)
            self.save_as_pkl(data_r_path, self.r_plus_vfsp)
            self.save_as_pkl(data_size_path, self.data_size)
        else:
            self.q_sa = self.load_from_pkl(data_q_path)
            self.r_plus_vfsp = self.load_from_pkl(data_r_path)
            self.data_size = self.load_from_pkl(data_size_path)

            logging.info(f"Memory usage after loading from pickle: {memory_usage()} MB")



    def get_ranking(self,env_index):
        Bvft_folder = "Bvft_Records"

        Q_result_folder = "Exp_result"
        folder_name = f"{self.env_name_list[env_index]}"
        Q_saving_folder = os.path.join(Q_result_folder,folder_name)
        self.create_folder(Q_saving_folder)
        method_folder_name = f"target_{self.target_trajectory_num}_trajectories_{self.target_traj_sa_number}_sa_normal_{self.trajectory_num}_trajectory_{self.traj_sa_number}_sa_{self.self_method_name}"
        method_folder_name = os.path.join(Q_saving_folder,method_folder_name)
        self.create_folder(method_folder_name)
        for j in range(len(self.policy_list)):
            # print("len policy list : ",len(self.policy_list))
            policy_name = self.policy_name_list[j]
            # print("policy name : ",policy_name)
            # print("len policy name list : ",len(self.policy_name_list))
            # print("policy name  list : ",self.policy_name_list)
            Q_result_saving_path = os.path.join(method_folder_name,policy_name)
            q_list = []
            r_plus_vfsp = []
            for i in range(len(self.env_list)):
                q_list.append(self.q_sa[(i)*len(self.policy_list)+(j+1)-1])
                r_plus_vfsp.append(self.r_plus_vfsp[(i)*len(self.policy_list)+(j+1)-1])
            result = self.select_Q(q_list=q_list,r_plus_vfsp=r_plus_vfsp,policy_namei=policy_name)
            index = np.argmin(result)
            save_list = [self.env_name_list[index]]
            self.save_as_txt(Q_result_saving_path, save_list)
            self.save_as_pkl(Q_result_saving_path, save_list)
            self.delete_files_in_folder(Bvft_folder)



    def run(self,true_data_list):
        start_time = time.time()
        self.train_policy()

        # for j in range(len(true_data_list)):
        before_for_time = time.time()
        for j in range(len(true_data_list)):
            for i in range(len(self.policy_list)):
                self.get_policy_performance(true_env_index=j,true_policy_index=i)
                # self.load_offline_data(max_time_step=self.max_timestep,algorithm_name=self.algorithm_name_list[i],
                #                        true_env_number=true_data_list[j])
                # load_time = time.time()
                # print(f"load_time use : {load_time - before_for_time}")
                # self.get_whole_qa(i)
                # get_whole_end = time.time()
                # print(f"after whole end time : {get_whole_end - load_time}")
                # self.get_ranking(i)
                # get_ranking_end = time.time()
                # print(f"get ranking time : {get_ranking_end - get_whole_end}")
                # print(f"one for loop total time : {get_ranking_end - before_for_time}")
                # sys.exit()
                self.load_offline_data(max_time_step=self.max_timestep,policy_index = j,
                                       true_env_number=true_data_list[j])
                # if self.policy_choose == 0 :
                #     for h in range(len(self.policy_list)):
                #         self.policy_list[h] = RandomPolicy(self.env_list[0].action_space)

                self.get_whole_qa(env_index = j, policy_index = i)
            self.get_ranking(env_index=j)


        end_time = time.time()
        # self.delete_files_in_folder_r("Offline_data")
        return (end_time - before_for_time)






