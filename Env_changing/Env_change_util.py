from abc import ABC, abstractmethod
import numpy as np
import sys
import os
import pickle
import heapq
from typing import Sequence
from d3rlpy.datasets import get_d4rl
import gym
import random
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
import pickle
import heapq
from typing import Sequence
from d3rlpy.datasets import get_d4rl
import gym
import random
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
class CustomDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.current = 0
        self.size = 0
        for i in range(len(dataset)):
            self.size += len(dataset[i]["action"])

    def get_iter_length(self,iteration_number):
        return len(self.dataset[iteration_number]["state"])
    def get_state_shape(self):
        first_state = self.dataset.observations[0]
        return np.array(first_state).shape
    def sample(self, iteration_number):
        dones =np.array(self.dataset[iteration_number]["done"])
        states = np.array(self.dataset[iteration_number]["state"])
        actions =  np.array(self.dataset[iteration_number]["action"])
        padded_next_states =  np.array(self.dataset[iteration_number]["next_state"])
        rewards =np.array( self.dataset[iteration_number]["rewards"])

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
        self.q_size = len(q_functions)                                          #how many (s,a) pairs (q function length)
        self.verbose = verbose                                                  #if true, print log
        if bins is None:
            bins = [2,  4, 5,  7, 8,  10, 11, 12, 16, 19, 22,23]
        self.bins = bins                                                        #used for discretizing Q-values
        self.q_sa = q_sa                                                    #all trajectory q s a
        self.r_plus_vfsp = r_plus_vfsp                                                 #reward
        self.q_functions = q_functions                                          #all q functions
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
            print(F"Being discretizing outputs of Q functions on batch data with resolution = {resolution}")
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

    def __init__(self,device,parameter_list,parameter_name_list,policy_training_parameter_map,method_name_list,self_method_name,gamma=0.99,trajectory_num=10,
                 max_timestep = 100, total_select_env_number=2,
                 env_name = "Hopper-v4"):
        self.device = device
        self.q_functions = []
        self.method_name_list = method_name_list
        self.max_timestep = max_timestep
        self.env_name = env_name
        self.parameter_list = parameter_list
        self.parameter_name_list = parameter_name_list
        self.unique_numbers = []
        self.env_list = []
        self.policy_total_step = policy_training_parameter_map["policy_total_step"]
        self.policy_episode_step = policy_training_parameter_map["policy_episode_step"]
        self.policy_saving_number = policy_training_parameter_map["policy_saving_number"]
        self.policy_learning_rate = policy_training_parameter_map["policy_learning_rate"]
        self.policy_hidden_layer = policy_training_parameter_map["policy_hidden_layer"]
        self.algorithm_name_list = policy_training_parameter_map["algorithm_name_list"]
        self.policy_list = []
        self.policy_name_list = []
        self.data = []
        self.gamma = gamma
        self.self_method_name = self_method_name
        self.trajectory_num = trajectory_num
        self.true_env_num = 0
        for i in range(len(self.parameter_list)):
            current_env = gymnasium.make(self.env_name)
            for param_name, param_value in zip(self.parameter_name_list, self.parameter_list[i]):
                setattr(current_env.unwrapped.model.opt, param_name, param_value)
            # print(current_env.unwrapped.model.opt)
            self.env_list.append(current_env)
        self.para_map = {index: item for index, item in enumerate(self.parameter_list)}
        self.q_sa = []
        self.r_plus_vfsp = []
        self.data_size = 0



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
            self.unique_numbers = self.load_from_pkl(data_seeds_path)
            self.q_sa = [np.zeros(self.data.size) for _ in range(len(self.env_list) * 2 * len(self.env_list))]
            self.r_plus_vfsp = [np.zeros(self.data.size) for _ in range(len(self.env_list) * 2 * len(self.env_list))]
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
        self.q_sa = [np.zeros(self.data.size) for _ in range(len(self.env_list)*2*len(self.env_list) )]
        self.r_plus_vfsp = [np.zeros(self.data.size) for _ in range(len(self.env_list)*2*len(self.env_list) )]
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
                                self.policy_name_list.append(policy_model_name[:-3]+"_"+str(self.policy_total_step)+"step")
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
                                    policy_model_name[:-3] + "_" + str(self.policy_total_step) + "step")
                    if os.path.exists(checkpoint_list_path + ".pkl"):
                        os.remove(checkpoint_list_path + ".pkl")
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                    print(f"end training {policy_folder_name} with algorithm {str(self.algorithm_name_list)}")
                else:
                    policy_path = policy_path[:-3]+"_"+str(self.policy_total_step)+"step.d3"
                    policy = d3rlpy.load_learnable(policy_path, device=self.device)
                    self.policy_list.append(policy)
                    self.policy_name_list.append(policy_model_name[:-3] + "_" + str(self.policy_total_step) + "step")
                    print("beegin load policy : ",str(policy_path))
            # print("sleep now")
            # time.sleep(600)
    def get_policy_per(self,policy,environment):
        total_rewards = 0
        max_iteration = 1000
        env = environment
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

    def get_qa(self,policy_number,environment_number,states,actions):
        env = self.env_list[environment_number]
        policy = self.policy_list[policy_number]
        result_list = []
        for i in range(len(states)):
            total_rewards = 0
            for j in range(1):
                num_step = 0
                discount_factor = 1
                # print("len states : ",len(states))p
                # print("len actions : ",len(actions)
                observation =states[i]
                action = actions[i]
                env.reset()
                #print("before env.observation : ",observation)
                env.observation= observation
                # env.state = observation
                # print("env.observation : ", env.observation)
                # sys.exit()
                ui = env.step(action)
                state = ui[0]
                reward = ui[1]
                total_rewards += reward
                done = ui[2]
                while ((not done) and (num_step < self.max_timestep)):
                    action = policy.predict(np.array([state]))
                    # print("predicted actioin : ",action)
                    ui = env.step(action[0])
                    state = ui[0]
                    reward = ui[1]
                    done = ui[2]
                    # print("state=E fr e step : ",state)
                    total_rewards += reward * discount_factor
                    discount_factor *= self.gamma
                    num_step += 1
            total_rewards = total_rewards
            result_list.append(total_rewards)
        return result_list

    def get_whole_qa(self,algorithm_index):
        Offine_data_folder = "Offline_data"
        self.create_folder(Offine_data_folder)
        data_folder_name = f"{self.algorithm_name_list[algorithm_index]}_{self.env_name}"
        for j in range(len(self.parameter_list[self.true_env_num])):
            param_name = self.parameter_name_list[j]
            param_value = self.parameter_list[self.true_env_num][j].tolist()
            data_folder_name += f"_{param_name}_{str(param_value)}"
        data_folder_name += f"_{self.max_timestep}_maxStep_{self.trajectory_num}_trajectory_{self.true_env_num}"
        data_q_name = data_folder_name  + "_q"
        data_q_path = os.path.join(Offine_data_folder,data_q_name)
        data_r_name = data_folder_name  + "_r"
        data_r_path = os.path.join(Offine_data_folder,data_r_name)
        data_size_name = data_folder_name  + "_size"
        data_size_path = os.path.join(Offine_data_folder,data_size_name)
        if(not self.whether_file_exists(data_q_path+".pkl")):
            ptr = 0
            gamma = self.gamma
            trajectory_length = 0
            while ptr < self.trajectory_num:  # for everything in data size
                length = self.data.get_iter_length(ptr)
                state, action, next_state, reward, done = self.data.sample(ptr)
                for i in range(len(self.env_list)):
                    for j in range(len(self.policy_list)):
                        self.q_sa[(i + 1) * len(self.policy_list)+ (j + 1) - 1][trajectory_length:trajectory_length + length] = self.get_qa(policy_number=j,environment_number=i,states=state,actions=action)
                        # print("actions : ",[self.policy_list[j].predict(next_state)])
                        # print("len next state : ",len(next_state))o
                        # print("len actions : ",len(action))i
                        # print("next actions : ",self.policy_list[j].predict(next_state))
                        vfsp = (reward + self.get_qa(j, i, next_state, self.policy_list[j].predict(next_state)) * (1 - np.array(done)) * gamma)

                        self.r_plus_vfsp[(i + 1) * (j + 1) - 1][trajectory_length:trajectory_length + length] = vfsp.flatten()[:length]
                trajectory_length += length
                ptr += 1
            self.data_size = trajectory_length
            print("self q_sa: ", self.q_sa)
            print("lesa :  ", len(self.q_sa))
            self.save_as_pkl(data_q_path,self.q_sa)
            self.save_as_pkl(data_r_path,self.r_plus_vfsp)
            self.save_as_pkl(data_size_path,self.data_size)
        else:
            self.q_sa  = self.load_from_pkl(data_q_path)
            self.r_plus_vfsp = self.load_from_pkl(data_r_path)
            self.data_size = self.load_from_pkl(data_size_path)
    def get_ranking(self,algorithm_index):
        Q_result_folder = "Exp_result"
        Q_saving_folder = os.path.join(Q_result_folder,self.self_method_name)

        data_folder_name = f"{self.algorithm_name_list[algorithm_index]}_{self.env_name}"
        for j in range(len(self.parameter_list[self.true_env_num])):
            param_name = self.parameter_name_list[j]
            param_value = self.parameter_list[self.true_env_num][j].tolist()
            data_folder_name += f"_{param_name}_{str(param_value)}"
        data_folder_name += f"_{self.max_timestep}_maxStep_{self.trajectory_num}_trajectory_{self.true_env_num}"
        Q_saving_folder_data = os.path.join(Q_saving_folder,data_folder_name)
        self.whether_file_exists(Q_saving_folder_data)
        for j in range(len(self.policy_list)):
            policy_name = self.policy_name_list[j]
            Q_result_saving_path = os.path.join(Q_saving_folder_data,policy_name)
            q_list = []
            r_plus_vfsp = []
            for i in range(len(self.env_list)):
                q_list.append(self.q_sa[(i+1)*len(self.policy_list)+(j+1)-1])
                r_plus_vfsp.append(self.r_plus_vfsp[(i+1)*len(self.policy_list)+(j+1)-1])
            result = self.select_Q(q_list,r_plus_vfsp,policy_namei)
            index = np.argmin(result)
            save_list = [q_name_functions[index]]
            self.save_as_txt(Q_result_saving_path, save_list)
            self.save_as_pkl(Q_result_saving_path, save_list)
            self.delete_files_in_folder(Bvft_folder)



    def run(self,true_data_list):
        self.train_policy()
        self.get_policy_performance()
        for j in range(len(true_data_list)):
            for i in range(len(self.algorithm_name_list)):
                self.load_offline_data(max_time_step=self.max_timestep,algorithm_name=self.algorithm_name_list[i],
                                       true_env_number=true_data_list[j])
                self.get_whole_qa(i)
                self.get_ranking(i)






