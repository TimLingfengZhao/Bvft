
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
# class CustomDataLoader:
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.current = 0
#         self.size = 0
#         for i in range(len(dataset)):
#             self.size += len(dataset[i]["action"])
#
#     def get_iter_length(self,iteration_number):
#         return len(self.dataset[iteration_number]["state"])
#     def get_state_shape(self):
#         first_state = self.dataset.observations[0]
#         return np.array(first_state).shape
#     def sample(self, iteration_number):
#         dones =np.array(self.dataset[iteration_number]["done"])
#         states = np.array(self.dataset[iteration_number]["state"])
#         actions =  np.array(self.dataset[iteration_number]["action"])
#         padded_next_states =  np.array(self.dataset[iteration_number]["next_state"])
#         rewards =np.array( self.dataset[iteration_number]["rewards"])
#
#         return states, actions, padded_next_states, rewards, dones


class CustomDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.current = 0
        self.size = 0
        self.length = len(dataset)
        for i in range(len(dataset)):
            self.size += len(dataset[i]["action"])

    def get_iter_length(self, iteration_number):
        return len(self.dataset[iteration_number]["state"])

    def get_state_shape(self):
        first_state = self.dataset[0]["state"]
        return np.array(first_state).shape

    def sample(self, iteration_number):
        dones = np.array(self.dataset[iteration_number]["done"])
        states = np.array(self.dataset[iteration_number]["state"])
        actions = np.array(self.dataset[iteration_number]["action"])
        padded_next_states = np.array(self.dataset[iteration_number]["next_state"])
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

    def __init__(self,device,parameter_list,parameter_name_list,policy_training_parameter_map,method_name_list,self_method_name,batch_size = 32,process_num=5,
                traj_sa_number = 10000,gamma=0.99,trajectory_num=10,
                 max_timestep = 100, total_select_env_number=2,
                 env_name = "Hopper-v4"):
        # self.policy_choose = policy_choose
        self.traj_sa_number = traj_sa_number
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
        for i in range(len(self.parameter_list)):
            current_env = gymnasium.make(self.env_name)
            name = f"{self.env_name}"
            for param_name, param_value in zip(self.parameter_name_list, self.parameter_list[i]):
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
    def select_Q(self,q_sa,r_plus_q,policy_namei):
        pass
    def remove_duplicates(self,lst):
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    def generate_one_trajectory(self,env_number,max_time_step,algorithm_name,unique_seed):
        Policy_operation_folder = "Policy_operation"
        Policy_saving_folder = os.path.join(Policy_operation_folder,"Policy_trained")
        self.create_folder(Policy_saving_folder)
        policy_folder_name = f"{self.env_name}"
        for j in range(len(self.parameter_list[env_number])):
            param_name = self.parameter_name_list[j]
            param_value = self.parameter_list[env_number][j].tolist()
            policy_folder_name += f"_{param_name}_{str(param_value)}"
        policy_saving_path = os.path.join(Policy_saving_folder, policy_folder_name)
        policy_model_name = f"{algorithm_name}_{str(self.policy_total_step)}_{str(self.policy_learning_rate)}_{str(self.policy_hidden_layer)}_{self.policy_total_step}step.d3"
        policy_path = os.path.join(policy_saving_path, policy_model_name)
        policy = d3rlpy.load_learnable(policy_path, device=self.device)
        env = self.env_list[env_number]
        obs,info = env.reset(seed=unique_seed)

        observations = []
        rewards = []
        actions = []
        dones = []
        next_steps = []
        episode_data = {}
        observations.append(obs)
        # print("initial obs : ",obs)
        for t in range(max_time_step):

            action = policy.predict(np.array([obs]))
            # print("action after prediction : ",action)
            state, reward, done, truncated, info = env.step(action[0])
            actions.append(action[0])
            rewards.append(reward)
            dones.append(done)
            next_steps.append(state)
            if((t != max_time_step-1) and done == False):
                observations.append(state)

            obs = state
            # print("state in env step : ",state)

            if done or truncated:
                break
        episode_data["action"] = actions
        episode_data["state"] = observations
        episode_data["rewards"] = rewards
        episode_data["done"] = dones
        episode_data["next_state"] = next_steps
        return episode_data
    def load_offline_data(self,max_time_step,algorithm_name,true_env_number):
        self.print_environment_parameters()
        self.true_env_num = true_env_number
        Offine_data_folder = "Offline_data"
        self.create_folder(Offine_data_folder)
        data_folder_name = f"{algorithm_name}_{self.env_name}"
        for j in range(len(self.parameter_list[true_env_number])):
            param_name = self.parameter_name_list[j]
            param_value = self.parameter_list[true_env_number][j].tolist()
            data_folder_name += f"_{param_name}_{str(param_value)}"
        data_folder_name += f"_{max_time_step}_maxStep_{self.trajectory_num}_trajectory_{self.true_env_num}"
        data_path = os.path.join(Offine_data_folder, data_folder_name)
        data_seeds_name = data_folder_name + "_seeds"
        data_seeds_path = os.path.join(Offine_data_folder,data_seeds_name)
        if os.path.exists(data_path):
            self.data = self.load_from_pkl(data_path)
            self.data_size = self.data.size
            self.unique_numbers = self.load_from_pkl(data_seeds_path)
            self.q_sa = [np.zeros(self.data.size) for _ in range(len(self.env_list)  * len(self.env_list))]
            self.r_plus_vfsp = [np.zeros(self.data.size) for _ in range(len(self.env_list)  * len(self.env_list))]
        else:
            self.generate_offline_data(max_time_step,algorithm_name,true_env_number)

    def generate_offline_data(self,max_time_step,algorithm_name,true_env_number):
        self.print_environment_parameters()
        self.true_env_num = true_env_number
        unique_numbers = self.generate_unique_numbers(self.trajectory_num, 1, 12345)
        self.unique_numbers = unique_numbers
        final_data = []
        for i in range(self.trajectory_num):
            one_episode_data = self.generate_one_trajectory(true_env_number,max_time_step,algorithm_name,unique_numbers[i])
            final_data.append(one_episode_data)

        Offine_data_folder = "Offline_data"
        self.create_folder(Offine_data_folder)
        data_folder_name = f"{algorithm_name}_{self.env_name}"
        for j in range(len(self.parameter_list[true_env_number])):
            param_name = self.parameter_name_list[j]
            param_value = self.parameter_list[true_env_number][j].tolist()
            data_folder_name += f"_{param_name}_{str(param_value)}"
        data_folder_name += f"_{max_time_step}_maxStep_{self.trajectory_num}_trajectory_{self.true_env_num}"
        data_path = os.path.join(Offine_data_folder,data_folder_name)
        data_seeds_name = data_folder_name + "_seeds"
        data_seeds_path = os.path.join(Offine_data_folder,data_seeds_name)
        self.data = CustomDataLoader(final_data)
        self.data_size = self.data.size
        self.q_sa = [np.zeros(self.data.size) for _ in range(len(self.env_list)*len(self.env_list) )]
        self.r_plus_vfsp = [np.zeros(self.data.size) for _ in range(len(self.env_list)*len(self.env_list) )]
        self.save_as_pkl(data_path,self.data)
        self.save_as_pkl(data_seeds_path, self.unique_numbers)


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
        for i in range(len(self.parameter_list)):
            current_env = self.env_list[i]
            policy_folder_name = f"{self.env_name}"
            for j in range(len(self.parameter_list[i])):
                param_name = self.parameter_name_list[j]
                param_value = self.parameter_list[i][j].tolist()
                policy_folder_name += f"_{param_name}_{str(param_value)}"
            print(f"start training {policy_folder_name} with algorithm {str(self.algorithm_name_list)}")
            policy_saving_path = os.path.join(Policy_saving_folder, policy_folder_name)
            policy_checkpoints_path = os.path.join(Policy_checkpoints_folder, policy_folder_name)
            self.create_folder(policy_saving_path)
            self.create_folder(policy_checkpoints_path)
            num_epoch = int(self.policy_total_step / self.policy_episode_step)
            buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=1000000, env=current_env)
            explorer = d3rlpy.algos.ConstantEpsilonGreedy(0.3)
            checkpoint_list = []
            for algorithm_name in self.algorithm_name_list:
                checkpoint_path = os.path.join(policy_checkpoints_path,f"{algorithm_name}_checkpoints.d3")
                checkpoint_list_path = os.path.join(policy_checkpoints_path, f"{algorithm_name}_checkpoints")
                policy_model_name = f"{algorithm_name}_{str(self.policy_total_step)}_{str(self.policy_learning_rate)}_{str(self.policy_hidden_layer)}.d3"
                policy_path = os.path.join(policy_saving_path, policy_model_name)
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
    def get_policy_per(self,policy,environment):
        total_rewards = 0
        max_iteration = 1000
        env = copy.deepcopy(environment)
        for num in range(100):
            num_step = 0
            discount_factor = 1
            observation, info = env.reset(seed=12345)
            action = policy.predict(np.array([observation]))
            ui = env.step(action[0])
            state = ui[0]
            reward = ui[1]
            done = ui[2]
            while ((not done) and (num_step < 1000)):
                action = policy.predict(np.array([state]))
                ui = env.step(action[0])
                state = ui[0]
                reward = ui[1]
                done = ui[2]
                total_rewards += reward * discount_factor
                discount_factor *= self.gamma
                num_step += 1
        total_rewards = total_rewards / 100
        return total_rewards
    def get_policy_performance(self):
        Policy_operation_folder = "Policy_operation"
        Policy_performance_folder = os.path.join(Policy_operation_folder,"Policy_performance")
        self.create_folder(Policy_performance_folder)
        performance_folder_name = "performance"
        policy_performance_path = os.path.join(Policy_performance_folder,performance_folder_name)
        # while(True):
        final_result_list = []
        for i in range(len(self.policy_list)):
            result_list = []
            policy = self.policy_list[i]
            for current_env in range(len(self.env_list)):
                result_list.append(self.get_policy_per(policy=policy,environment=self.env_list[current_env]))
            final_result_list.append(result_list)
        self.save_as_pkl(policy_performance_path,final_result_list)
        self.save_as_txt(policy_performance_path,final_result_list)
    def run_simulation(self, state_action_policy_env_batch):
        states, actions, policy, envs = state_action_policy_env_batch
        # Initialize environments with states
        for env, state in zip(envs, states):
            env.reset()
            env.observation = state

        total_rewards = np.zeros(len(states))
        num_steps = np.zeros(len(states))
        discount_factors = np.ones(len(states))
        done_flags = np.zeros(len(states), dtype=bool)

        print_idx = 0
        pr_i = 0
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




        return total_rewards
    # def run_simulation(self, state_action_policy_env_batch):
    #     states, actions, policy, envs = state_action_policy_env_batch
    #     before_ini = time.time()
    #     # Initialize environments with states
    #     for env, state in zip(envs, states):
    #         env.reset()
    #         env.observation = state
    #     after_ini = time.time()
    #
    #     total_rewards = np.zeros(len(states))
    #     num_steps = np.zeros(len(states))
    #     discount_factors = np.ones(len(states))
    #     done_flags = np.zeros(len(states), dtype=bool)
    #
    #     after_setup = time.time()
    #     print_idx = 0
    #     pr_i = 0
    #     while not np.all(done_flags) and np.any(num_steps < self.max_timestep):
    #         before_predict = time.time()
    #
    #         actions_batch = policy.predict(np.array(states))
    #
    #         after_predict = time.time()
    #
    #
    #         for idx, (env, action) in enumerate(zip(envs, actions_batch)):
    #             dudu_time = time.time()
    #             if not done_flags[idx]:
    #                 before_env_step = time.time()
    #                 next_state, reward, done, _, _ = env.step(action)
    #                 after_env_step = time.time()
    #                 total_rewards[idx] += reward * discount_factors[idx]
    #                 discount_factors[idx] *= self.gamma
    #                 num_steps[idx] += 1
    #                 states[idx] = next_state
    #                 done_flags[idx] = done
    #                 after_four_index_calculation = time.time()
    #                 if(print_idx == 0) :
    #                     print(f"current run simulation time use : \n"
    #                           f"initialize time : {after_ini - before_ini} \n"
    #                           f"set up time : {after_setup - after_ini}\n"
    #                           f"predict time : {after_predict - before_predict} \n"
    #                           f"env step time : {after_env_step - before_env_step} \n"
    #                           f"cauculation five thing time : {after_four_index_calculation - after_env_step}")
    #                     print_idx += 1
    #             if idx == 0 :
    #                 print(f"one iter in for loop time : {time.time() - dudu_time}")
    #         after_for_loop_time = time.time()
    #         if pr_i ==0:
    #             print(f"for loop time : {time.time() - after_predict}")
    #             pr_i += 1
    #     print(f"after while loop time : {time.time() - after_setup}")
    #     sys.exit()
    #
    #
    #
    #
    #     return total_rewards

    def get_qa(self, policy_number, env_copy_list, states, actions, batch_size=8):
        policy = self.policy_list[policy_number]
        results = []

        total_len = len(states)
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


    def process_env_policy_combination(self,env_index, policy_index, policy_list, env_list, data, gamma, batch_size,
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

    def get_whole_qa(self, algorithm_index):
        Offline_data_folder = "Offline_data"
        self.create_folder(Offline_data_folder)
        data_folder_name = f"{self.algorithm_name_list[algorithm_index]}_{self.env_name}"
        for j in range(len(self.parameter_list[self.true_env_num])):
            param_name = self.parameter_name_list[j]
            param_value = self.parameter_list[self.true_env_num][j].tolist()
            data_folder_name += f"_{param_name}_{str(param_value)}"
        data_folder_name += f"_{self.max_timestep}_maxStep_{self.trajectory_num}_trajectory_{self.true_env_num}"
        data_q_name = data_folder_name + "_q"
        data_q_path = os.path.join(Offline_data_folder, data_q_name)
        data_r_name = data_folder_name + "_r"
        data_r_path = os.path.join(Offline_data_folder, data_r_name)
        data_size_name = data_folder_name + "_size"
        data_size_path = os.path.join(Offline_data_folder, data_size_name)

        if not self.whether_file_exists(data_q_path + ".pkl"):
            logging.info("Enter get qa calculate loop")
            start_time = time.time()

            threading_start_time = time.time()
            print("self process num : ",self.process_num)
            with Pool(pathos.multiprocessing.cpu_count()) as pool:
                results = [pool.apply_async(self.process_env_policy_combination, args=(
                    i, j, self.policy_list, self.env_list, self.data, self.gamma, self.batch_size,
                    self.trajectory_num,
                    self.data.size)) for i in range(len(self.env_list)) for j in range(len(self.policy_list))]

                q_sa_aggregated = [np.zeros(self.data.size) for _ in range(len(self.env_list) * len(self.policy_list))]
                r_plus_vfsp_aggregated = [np.zeros(self.data.size) for _ in
                                          range(len(self.env_list) * len(self.policy_list))]
                data_size_aggregated = 0

                for idx, result in enumerate(results):
                    q_sa_partial, r_plus_vfsp_partial, data_size_partial = result.get()
                    q_sa_aggregated[idx] += q_sa_partial
                    r_plus_vfsp_aggregated[idx] += r_plus_vfsp_partial
                    data_size_aggregated += data_size_partial


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


        # def run_simulation(self, state_action_policy_env_batch):
    #     states, actions, policy, envs = state_action_policy_env_batch
    #
    #     # Initialize environments with states
    #     for env, state in zip(envs, states):
    #         env.reset()
    #         env.observation = state
    #
    #     total_rewards = np.zeros(len(states))
    #     num_steps = np.zeros(len(states))
    #     discount_factors = np.ones(len(states))
    #     done_flags = np.zeros(len(states), dtype=bool)
    #
    #     while not np.all(done_flags) and np.any(num_steps < self.max_timestep):
    #         actions_batch = policy.predict(np.array(states))
    #         for idx, (env, action) in enumerate(zip(envs, actions_batch)):
    #             if not done_flags[idx]:
    #                 next_state, reward, done, _, _ = env.step(action)
    #                 total_rewards[idx] += reward * discount_factors[idx]
    #                 discount_factors[idx] *= self.gamma
    #                 num_steps[idx] += 1
    #                 states[idx] = next_state
    #                 done_flags[idx] = done
    #
    #     return total_rewards
    #
    # def get_qa(self, policy_number, env_copy_list, states, actions, batch_size=8):
    #     policy = self.policy_list[policy_number]
    #     results = []
    #
    #     total_len = len(states)
    #     for i in range(0, total_len, batch_size):
    #         actual_batch_size = min(batch_size, total_len - i)
    #         state_action_policy_env_pairs = (states[i:i + actual_batch_size], actions[i:i + actual_batch_size], policy,
    #                                          env_copy_list[:actual_batch_size])
    #         batch_results = self.run_simulation(state_action_policy_env_pairs)
    #         results.extend(batch_results)
    #
    #     return results
    #
    # def create_deep_copies(self, env, batch_size):
    #     return [copy.deepcopy(env) for _ in range(batch_size)]
    #
    # def process_env(self, env_index, policy_index, env_copy_lists_name, policy_list_name, data_name, env_list_name):
    #     start_time = time.time()
    #     logging.info(f"Starting processing for env_index {env_index}, policy_index {policy_index}")
    #
    #     # Attach to existing shared memory blocks
    #     shm_env_copy_lists = shared_memory.SharedMemory(name=env_copy_lists_name)
    #     env_copy_lists = dill.loads(bytes(shm_env_copy_lists.buf))
    #
    #     shm_policy_list = shared_memory.SharedMemory(name=policy_list_name)
    #     policy_list = dill.loads(bytes(shm_policy_list.buf))
    #
    #     shm_data = shared_memory.SharedMemory(name=data_name)
    #     data = dill.loads(bytes(shm_data.buf))
    #
    #     shm_env_list = shared_memory.SharedMemory(name=env_list_name)
    #     env_list = dill.loads(bytes(shm_env_list.buf))
    #
    #     env = env_list[env_index]
    #     env_copy_list = env_copy_lists[env_index][policy_index]
    #     policy = policy_list[policy_index]
    #     ptr = 0
    #     trajectory_length = 0
    #     while ptr < self.data.size:
    #         length = data[ptr]["state"].shape[0]  # or use self.data.get_iter_length(ptr) if it fits better
    #         state, action, next_state, reward, done = data[ptr]
    #         self.q_sa[(env_index + 1) * len(policy_list) + (policy_index + 1) - 1][
    #         trajectory_length:trajectory_length + length] = self.get_qa(policy_index, env_copy_list, state, action,
    #                                                                     self.batch_size)
    #         vfsp = (reward + self.get_qa(policy_index, env_copy_list, next_state,
    #                                      policy.predict(next_state), self.batch_size) * (
    #                         1 - np.array(done)) * self.gamma)
    #         self.r_plus_vfsp[(env_index + 1) * (policy_index + 1) - 1][
    #         trajectory_length:trajectory_length + length] = vfsp.flatten()[:length]
    #         trajectory_length += length
    #         ptr += 1
    #     self.data_size = trajectory_length
    #
    #     end_time = time.time()
    #     logging.info(
    #         f"Finished processing for env_index {env_index}, policy_index {policy_index} in {end_time - start_time} seconds")
    #
    #     # Cleanup shared memory in worker
    #     shm_env_copy_lists.close()
    #     shm_policy_list.close()
    #     shm_data.close()
    #     shm_env_list.close()
    #
    #     # Force garbage collection
    #     gc.collect()
    #
    # def get_whole_qa(self, algorithm_index):
    #     Offline_data_folder = "Offline_data"
    #     self.create_folder(Offline_data_folder)
    #     data_folder_name = f"{self.algorithm_name_list[algorithm_index]}_{self.env_name}"
    #     for j in range(len(self.parameter_list[self.true_env_num])):
    #         param_name = self.parameter_name_list[j]
    #         param_value = self.parameter_list[self.true_env_num][j].tolist()
    #         data_folder_name += f"_{param_name}_{str(param_value)}"
    #     data_folder_name += f"_{self.max_timestep}_maxStep_{self.trajectory_num}_trajectory_{self.true_env_num}"
    #     data_q_name = data_folder_name + "_q"
    #     data_q_path = os.path.join(Offline_data_folder, data_q_name)
    #     data_r_name = data_folder_name + "_r"
    #     data_r_path = os.path.join(Offline_data_folder, data_r_name)
    #     data_size_name = data_folder_name + "_size"
    #     data_size_path = os.path.join(Offline_data_folder, data_size_name)
    #
    #     if not self.whether_file_exists(data_q_path + ".pkl"):
    #         logging.info("Enter get qa calculate loop")
    #         start_time = time.time()
    #
    #         env_copy_start_time = time.time()
    #         env_copy_lists = np.array(
    #             [[self.create_deep_copies(env, self.batch_size) for env in self.env_list] for _ in self.policy_list],
    #             dtype=object)
    #         env_copy_serialized = dill.dumps(env_copy_lists)
    #         env_copy_shm = shared_memory.SharedMemory(create=True, size=len(env_copy_serialized))
    #         env_copy_shm.buf[:len(env_copy_serialized)] = env_copy_serialized
    #
    #         policy_list_serialized = dill.dumps(self.policy_list)
    #         policy_list_shm = shared_memory.SharedMemory(create=True, size=len(policy_list_serialized))
    #         policy_list_shm.buf[:len(policy_list_serialized)] = policy_list_serialized
    #
    #         data_list = [self.data.sample(i) for i in range(len(self.data.dataset))]
    #         data_serialized = dill.dumps(data_list)
    #         data_shm = shared_memory.SharedMemory(create=True, size=len(data_serialized))
    #         data_shm.buf[:len(data_serialized)] = data_serialized
    #
    #         env_list_serialized = dill.dumps(self.env_list)
    #         env_list_shm = shared_memory.SharedMemory(create=True, size=len(env_list_serialized))
    #         env_list_shm.buf[:len(env_list_serialized)] = env_list_serialized
    #
    #         env_copy_end_time = time.time()
    #         logging.info(f"Environment copy time: {env_copy_end_time - env_copy_start_time} seconds")
    #
    #         # Determine total memory and set maximum memory usage to 90% of it
    #         total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Total memory in GB
    #         max_memory_usage = total_memory * 0.9  # Set to 90% of total memory
    #         logging.info(f"Total memory: {total_memory} GB, Max memory usage: {max_memory_usage} GB")
    #
    #         threading_start_time = time.time()
    #         with Pool(processes=multiprocessing.cpu_count()) as pool:
    #             results = [pool.apply_async(self.process_env, args=(
    #                 i, j, env_copy_shm.name, policy_list_shm.name, data_shm.name, env_list_shm.name)) for i in
    #                        range(len(self.env_list))
    #                        for j in range(len(self.policy_list))]
    #
    #             # Memory usage check loop
    #             while results:
    #                 if self.get_memory_usage() > max_memory_usage:
    #                     logging.info(f"Memory usage exceeded {max_memory_usage}GB, waiting for tasks to complete...")
    #                     results.pop(0).get()  # Wait for the first task in the list to complete
    #                 else:
    #                     break  # Exit loop if memory usage is within limits
    #
    #             # Ensure all tasks are completed
    #             for result in results:
    #                 result.get()
    #
    #         threading_end_time = time.time()
    #         logging.info(f"Threading (env-policy) time: {threading_end_time - threading_start_time} seconds")
    #
    #         end_time = time.time()
    #         logging.info(f"Total running time get_qa: {end_time - start_time} seconds")
    #
    #         self.save_as_pkl(data_q_path, self.q_sa)
    #         self.save_as_pkl(data_r_path, self.r_plus_vfsp)
    #         self.save_as_pkl(data_size_path, self.data_size)
    #
    #         # Cleanup shared memory
    #         env_copy_shm.close()
    #         env_copy_shm.unlink()
    #         policy_list_shm.close()
    #         policy_list_shm.unlink()
    #         data_shm.close()
    #         data_shm.unlink()
    #         env_list_shm.close()
    #         env_list_shm.unlink()
    #
    #         # Force garbage collection
    #         gc.collect()
    #     else:
    #         self.q_sa = self.load_from_pkl(data_q_path)
    #         self.r_plus_vfsp = self.load_from_pkl(data_r_path)
    #         self.data_size = self.load_from_pkl(data_size_path)
    #
    # def get_memory_usage(self):
    #     # 获取当前进程的内存使用情况（以GB为单位）
    #     process = psutil.Process()
    #     mem_info = process.memory_info()
    #     return mem_info.rss / (1024 ** 3)



    # def run_simulation(self, state_action_policy_env_batch):
    #     states, actions, policy, envs = state_action_policy_env_batch
    #
    #     # Initialize environments with states
    #     for env, state in zip(envs, states):
    #         env.reset()
    #         env.observation = state
    #
    #     total_rewards = np.zeros(len(states))
    #     num_steps = np.zeros(len(states))
    #     discount_factors = np.ones(len(states))
    #     done_flags = np.zeros(len(states), dtype=bool)
    #
    #     while not np.all(done_flags) and np.any(num_steps < self.max_timestep):
    #         actions_batch = policy.predict(np.array(states))
    #         for idx, (env, action) in enumerate(zip(envs, actions_batch)):
    #             if not done_flags[idx]:
    #                 next_state, reward, done, _, _ = env.step(action)
    #                 total_rewards[idx] += reward * discount_factors[idx]
    #                 discount_factors[idx] *= self.gamma
    #                 num_steps[idx] += 1
    #                 states[idx] = next_state
    #                 done_flags[idx] = done
    #
    #     return total_rewards
    #
    # # def run_simulation(self, state_action_policy_env_batch):
    # #     states, actions, policy, envs = state_action_policy_env_batch
    # #
    # #     # Initialize environments with states
    # #     for env, state in zip(envs, states):
    # #         env.reset()
    # #         env.observation = state
    # #
    # #     total_rewards = np.zeros(len(states))
    # #     num_steps = np.zeros(len(states))
    # #     discount_factors = np.ones(len(states))
    # #     done_flags = np.zeros(len(states), dtype=bool)
    # #
    # #     while not np.all(done_flags) and np.any(num_steps < self.max_timestep):
    # #         actions_batch = policy.predict(np.array(states))
    # #         step_results = []
    # #
    # #         with concurrent.futures.ThreadPoolExecutor() as executor:
    # #             for env, action in zip(envs, actions_batch):
    # #                 step_results.append(executor.submit(env.step, action))
    # #
    # #         for idx, future in enumerate(step_results):
    # #             if not done_flags[idx]:
    # #                 next_state, reward, done, _, _ = future.result()
    # #                 total_rewards[idx] += reward * discount_factors[idx]
    # #                 discount_factors[idx] *= self.gamma
    # #                 num_steps[idx] += 1
    # #                 states[idx] = next_state
    # #                 done_flags[idx] = done
    # #
    # #     return total_rewards
    #
    # def get_qa(self, policy_number, env_copy_list, states, actions, batch_size=8):
    #     policy = self.policy_list[policy_number]
    #     results = []
    #
    #     total_len = len(states)
    #     for i in range(0, total_len, batch_size):
    #         actual_batch_size = min(batch_size, total_len - i)
    #         state_action_policy_env_pairs = (states[i:i + actual_batch_size], actions[i:i + actual_batch_size], policy,
    #                                          env_copy_list[:actual_batch_size])
    #         batch_results = self.run_simulation(state_action_policy_env_pairs)
    #         results.extend(batch_results)
    #
    #     return results
    #
    # def create_deep_copies(self, env, batch_size):
    #     return [copy.deepcopy(env) for _ in range(batch_size)]
    #
    # def get_whole_qa(self,algorithm_index):
    #     Offine_data_folder = "Offline_data"
    #     self.create_folder(Offine_data_folder)
    #     data_folder_name = f"{self.algorithm_name_list[algorithm_index]}_{self.env_name}"
    #     for j in range(len(self.parameter_list[self.true_env_num])):
    #         param_name = self.parameter_name_list[j]
    #         param_value = self.parameter_list[self.true_env_num][j].tolist()
    #         data_folder_name += f"_{param_name}_{str(param_value)}"
    #     data_folder_name += f"_{self.max_timestep}_maxStep_{self.trajectory_num}_trajectory_{self.true_env_num}"
    #     data_q_name = data_folder_name  + "_q"
    #     data_q_path = os.path.join(Offine_data_folder,data_q_name)
    #     data_r_name = data_folder_name  + "_r"
    #     data_r_path = os.path.join(Offine_data_folder,data_r_name)
    #     data_size_name = data_folder_name  + "_size"
    #     data_size_path = os.path.join(Offine_data_folder,data_size_name)
    #     if(not self.whether_file_exists(data_q_path+".pkl")):
    #         print("enter get qa calculate loop")
    #         ptr = 0
    #         gamma = self.gamma
    #         start_time = time.time()
    #         trajectory_length = 0
    #         env_copy_lists = []
    #         for i in range(len(self.env_list)):
    #             env = self.env_list[i]
    #             env_copy_lists.append(self.create_deep_copies(env, self.batch_size))
    #         env_copy_time = time.time()
    #         print(f"full env copy time : {env_copy_time - start_time} for batch size : {self.batch_size}")
    #
    #         while ptr < self.trajectory_num:  # for everything in data size
    #             length = self.data.get_iter_length(ptr)
    #             sample_state = time.time()
    #             state, action, next_state, reward, done = self.data.sample(ptr)
    #             print(f"sample time : {time.time() - sample_state}")
    #
    #             for i in range(len(self.env_list)):
    #                 env = self.env_list[i]
    #                 # env_copy_list = [copy.deepcopy(env) for _ in range(self.batch_size)]
    #                 env_copy_list = env_copy_lists[i]
    #
    #                 total_st = 0
    #                 for j in range(len(self.policy_list)):
    #                     #self.q_sa[(i + 1) * len(self.policy_list) + (j + 1) - 1][
    #                     # trajectory_length:trajectory_length + length]  = self.get_qa(policy_number=j,
    #                     #                                                             environment_number=i,
    #                     #                                                             states=state, actions=action)
    #                     # q_time = time.time()
    #                     self.q_sa[(i + 1) * len(self.policy_list)+ (j + 1) - 1][trajectory_length:trajectory_length + length] = self.get_qa(policy_number=j,env_copy_list=env_copy_list,states=state,actions=action,batch_size=self.batch_size)
    #                     # vfsp = (reward + self.get_qa(policy_number=j, environment_number=i, states=next_state,
    #                     #                              actions=self.policy_list[j].predict(next_state)
    #                     #                              ) * (1 - np.array(done)) * self.gamma)
    #                     # q_end = time.time()
    #                     # print(f"q sa time : {q_end - q_time}")
    #                     vfsp = (reward + self.get_qa(policy_number=j, env_copy_list=env_copy_list, states=next_state, actions=self.policy_list[j].predict(next_state),batch_size=self.batch_size) * (1 - np.array(done)) * self.gamma)
    #                     vfsp_end = time.time()
    #                     # print(f"vfsp time : {vfsp_end - q_end}")
    #                     self.r_plus_vfsp[(i + 1) * (j + 1) - 1][trajectory_length:trajectory_length + length] = vfsp.flatten()[:length]
    #                     # r_plus_end = time.time()
    #                     # print(f"r plus vfsp end : {r_plus_end - vfsp_end}")
    #                     # print(f"total time : {r_plus_end - q_time}")
    #                     # sys.exit()
    #             trajectory_length += length
    #             ptr += 1
    #         end_time = time.time()
    #         print(f"running time get_qa : {self.batch_size} , running time : {end_time - start_time}")
    #         self.data_size = trajectory_length
    #         self.save_as_pkl(data_q_path,self.q_sa)
    #         self.save_as_pkl(data_r_path,self.r_plus_vfsp)
    #         self.save_as_pkl(data_size_path,self.data_size)
    #     else:
    #         self.q_sa  = self.load_from_pkl(data_q_path)
    #         self.r_plus_vfsp = self.load_from_pkl(data_r_path)
    #         self.data_size = self.load_from_pkl(data_size_path)

    def get_ranking(self,algorithm_index):
        Bvft_folder = "Bvft_Records"

        Q_result_folder = "Exp_result"
        Q_saving_folder = os.path.join(Q_result_folder,self.self_method_name)

        data_folder_name = f"{self.algorithm_name_list[algorithm_index]}_{self.env_name}"
        for j in range(len(self.parameter_list[self.true_env_num])):
            param_name = self.parameter_name_list[j]
            param_value = self.parameter_list[self.true_env_num][j].tolist()
            data_folder_name += f"_{param_name}_{str(param_value)}"
        data_folder_name += f"_{self.max_timestep}_maxStep_{self.trajectory_num}_trajectory_{self.true_env_num}"
        Q_saving_folder_data = os.path.join(Q_saving_folder,data_folder_name)
        self.create_folder(Q_saving_folder_data)
        for j in range(len(self.policy_list)):
            # print("len policy list : ",len(self.policy_list))
            policy_name = self.policy_name_list[j]
            # print("policy name : ",policy_name)
            # print("len policy name list : ",len(self.policy_name_list))
            # print("policy name  list : ",self.policy_name_list)
            Q_result_saving_path = os.path.join(Q_saving_folder_data,policy_name)
            q_list = []
            r_plus_vfsp = []
            print("q sa : ",self.q_sa[0])
            for i in range(len(self.env_list)):
                q_list.append(self.q_sa[(i)*len(self.policy_list)+(j+1)-1])
                r_plus_vfsp.append(self.r_plus_vfsp[(i)*len(self.policy_list)+(j+1)-1])
            result = self.select_Q(q_list,r_plus_vfsp,policy_name)
            index = np.argmin(result)
            save_list = [self.env_name_list[index]]
            self.save_as_txt(Q_result_saving_path, save_list)
            self.save_as_pkl(Q_result_saving_path, save_list)
            self.delete_files_in_folder(Bvft_folder)

    # def draw_figure_6R(self):
        # means = []
        # SE = []
        # labels = []
        # self_data_saving_path = self.remove_duplicates(self.data_saving_path)
        # max_step = str(max(self.FQE_saving_step_list))
        # for i in range(len(self_data_saving_path)):
        #     repo_name = self_data_saving_path[i]
        #     NMSE,standard_error = self.get_NMSE(repo_name)
        #     means.append(NMSE)
        #     SE.append(standard_error)
        #     labels.append(self_data_saving_path[i]+"_"+max_step)
        # name_list = ['hopper-medium-v2']
        #
        # FQE_returned_folder = "Policy_ranking_saving_place/Policy_k_saving_place/Figure_6R_plot"
        # if not os.path.exists(FQE_returned_folder):
        #     os.makedirs(FQE_returned_folder)
        # plot = "NMSE_plot"
        # Figure_saving_path = os.path.join(FQE_returned_folder, plot)
        # #
        # colors = self.generate_unique_colors(len(self_data_saving_path))
        # figure_name = 'Normalized e MSE of FQE min max'
        # filename = "Figure6R_max_min_NMSE_graph" + "_" + str(self.FQE_saving_step_list)
        # if self.normalization_factor == 1:
        #     figure_name = 'Normalized MSE of FQE groundtruth variance'
        #     filename = "Figure6R_groundtruth_variance_NMSE_graph" + "_" + str(self.FQE_saving_step_list)
        # self.draw_mse_graph(combinations=name_list, means=means, colors=colors, standard_errors=SE,
        #                labels=labels, folder_path=Figure_saving_path, FQE_step_list=self.FQE_saving_step_list,
        #                filename=filename, figure_name=figure_name)

    def run(self,true_data_list):
        start_time = time.time()
        self.train_policy()
        self.get_policy_performance()

        # for j in range(len(true_data_list)):
        before_for_time = time.time()
        for j in range(len(true_data_list)):
            for i in range(len(self.algorithm_name_list)):
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
                self.load_offline_data(max_time_step=self.max_timestep,algorithm_name=self.algorithm_name_list[i],
                                       true_env_number=true_data_list[j])
                # if self.policy_choose == 0 :
                #     for h in range(len(self.policy_list)):
                #         self.policy_list[h] = RandomPolicy(self.env_list[0].action_space)
                self.get_whole_qa(i)
                self.get_ranking(i)
        end_time = time.time()
        self.delete_files_in_folder_r("Offline_data")
        return (end_time - before_for_time)






