
from abc import ABC, abstractmethod
import numpy as np
import sys
from multiprocessing import shared_memory
import os
import re
import multiprocess.context as ctx
import pickle
from itertools import product
import ast
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




class Hopper_edi(ABC):

    def __init__(self
                 ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def get_env_list(self,env_parameter_map_o):
        env_parameter_map = copy.deepcopy(env_parameter_map_o)
        env_name_list = []
        env_list = []
        for h in range(len(env_parameter_map["parameter_list"])):
            for i in range((len(env_parameter_map["parameter_list"][h]))):
                current_env = gymnasium.make(env_parameter_map["env_name"])
                name = f"{env_parameter_map['env_name']}"
                for param_name, param_value in zip(env_parameter_map["parameter_name_list"][h],env_parameter_map["parameter_list"][h][i]):
                    setattr(current_env.unwrapped.model.opt, param_name, param_value)
                    name += f"_{param_name}_{str(param_value)}"
                env_name_list.append(name)

                # print(current_env.unwrapped.model.opt)
                env_list.append(current_env)
        return env_list,env_name_list



    def get_policy_name(self,env_name,algorithm_name,learning_rate,hidden_layer,total_step):
        return f"{env_name}_{algorithm_name}_{learning_rate}_{str(hidden_layer)}_{total_step}step"

    def get_policy_path(self,env_name,algorithm_name,learning_rate,hidden_layer,total_step):
        Policy_operation = "Policy_operation"
        Policy_trained = "Policy_trained"
        policy_name = self.get_policy_name(env_name,algorithm_name,learning_rate,hidden_layer,total_step)
        return os.path.join(Policy_operation,Policy_trained,f"{policy_name}.d3")

    def parse_param_value(self,param_value):
        # Remove brackets and split by spaces or commas
        param_value = param_value.strip('[]')
        elements = param_value.split()
        if len(elements) == 1:
            elements = param_value.split(',')

        # Convert elements to floats
        try:
            float_elements = [float(el) for el in elements]
            if len(float_elements) > 1:
                return np.array(float_elements)
            else:
                return float_elements[0]
        except ValueError:
            raise ValueError(f"Could not convert param_value '{param_value}' to float or list of floats")

    def get_env(self, env_name):
        parts = env_name.split('_')
        base_env_name = parts[0]
        param_dict = {}

        for i in range(1, len(parts), 2):
            param_name = parts[i]
            param_value = parts[i + 1]
            if param_value.startswith('[') and param_value.endswith(']'):
                # Handle list-like parameter values
                param_value = self.parse_param_value(param_value)
            else:
                # Handle single numeric values
                param_value = float(param_value)
            param_dict[param_name] = param_value

        env = gymnasium.make(base_env_name)

        # Set parameters
        for param_name, param_value in param_dict.items():
            setattr(env.unwrapped.model.opt, param_name, param_value)

        return env

    def train_single_policy(self,env_name,algorithm_name,learning_rate,hidden_layer,total_step):
        Policy_saving_folder = "Policy_operation"
        Policy_trained_folder = os.path.join(Policy_saving_folder, "Policy_trained")
        Policy_checkpoints_folder = os.path.join(Policy_saving_folder, "Policy_checkpoints")

        self.create_folder(Policy_checkpoints_folder)
        self.create_folder(Policy_trained_folder)

        env = self.get_env(env_name)

        policy_saving_path = os.path.join(Policy_trained_folder, env_name)
        policy_checkpoints_path = os.path.join(Policy_checkpoints_folder, env_name)

        self.create_folder(policy_checkpoints_path)

        num_epoch = 10  # Assuming 1000 steps per episode
        buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=1500000, env=env)
        explorer = d3rlpy.algos.ConstantEpsilonGreedy(0.3)
        checkpoint_list = []

        # Prepare paths for saving checkpoints and final policy
        policy_model_name = f"{algorithm_name}_{learning_rate}_{str(hidden_layer)}_{total_step}step.d3"
        policy_path = policy_saving_path + "_" + policy_model_name
        checkpoint_path = os.path.join(policy_checkpoints_path, f"{algorithm_name}_{learning_rate}_{str(hidden_layer)}_{total_step}_checkpoints.d3")
        checkpoint_list_path = os.path.join(policy_checkpoints_path, f"{algorithm_name}_{learning_rate}_{str(hidden_layer)}_{total_step}_checkpoints")

        # Check if the policy already exists
        if not self.whether_file_exists(f"{policy_path}"):
            print(f"{policy_path} does not exist, starting training")

            self_class = getattr(d3rlpy.algos, f"{algorithm_name}Config")
            policy = self_class(
                actor_encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=hidden_layer),
                critic_encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=hidden_layer),
                actor_learning_rate=learning_rate,
                critic_learning_rate=learning_rate
            ).create(device=self.device)

            for epoch in range(num_epoch):
                policy.fit_online(env=env,
                                  buffer=buffer,
                                  explorer=explorer,
                                  n_steps=total_step/num_epoch,
                                  eval_env=env,
                                  with_timestamp=False)
                policy.save(checkpoint_path)
                checkpoint_list.append(epoch)
                self.save_as_pkl(checkpoint_list_path, checkpoint_list)

                if (epoch + 1) % num_epoch == 0:
                    policy.save(f"{policy_path}")

            if os.path.exists(f"{checkpoint_list_path}.pkl"):
                os.remove(f"{checkpoint_list_path}.pkl")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

            print(f"Finished training {env_name} with algorithm {algorithm_name}")

    def get_policy(self,env_name,algorithm_name,learning_rate,hidden_layer,total_step):
        policy_path = self.get_policy_path(env_name,algorithm_name,learning_rate,hidden_layer,total_step)
        if os.path.exists(policy_path):
            return d3rlpy.load_learnable(policy_path,
                                         device = self.device)
        else:
            self.train_single_policy(env_name,algorithm_name,learning_rate,hidden_layer,total_step)
            return d3rlpy.load_learnable(policy_path,
                                         device=self.device)


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

    def draw_Bvft_resolution_loss_graph(self,Bvft_final_resolution_loss,
                                      resolution_list,
                                        line_name_list, group_list,plot_saving_path):
        fig, ax = plt.subplots(figsize=(10, 6))

        x_list = []
        for i in range(len(resolution_list)):
            x_list.append(str(resolution_list[i]) + "res" + "_" + str(group_list[i]) + "groups")
        x_positions = list(range(len(x_list)))
        for index, y_values in enumerate(Bvft_final_resolution_loss):
            ax.plot(x_positions, y_values, label=line_name_list[index])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        ax.set_xticks(x_positions)  # Set the x positions
        ax.set_xticklabels(x_list, rotation=45, ha="right")
        ax.set_title("mean loss with different policy")
        ax.set_ylabel('Bvft_loss')
        ax.set_xlabel('resolutions_groups')

        Policy_ranking_saving_place = "Policy_ranking_saving_place"
        Policy_k_saving_place = "Policy_k_saving_place"
        policy_result = os.path.join(Policy_ranking_saving_place,Policy_k_saving_place)

        Plot_saving_path = plot_saving_path

        plt.savefig(plot_saving_path, bbox_inches='tight')
        plt.close()
    @abstractmethod
    def select_Q(self,q_list,q_prime,policy_namei,dataset,env,gamma=0.99,line_name_list = [],res_plot_save_path ="e"):
        pass
    def remove_duplicates(self,lst):
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def normalized_mean_square_error_with_error_bar(self,actual, predicted, NMSE_normalization_factor):

        if len(actual) != len(predicted):
            raise ValueError("The length of actual and predicted values must be the same.")

        squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]

        mse = sum(squared_errors) / len(actual)

        mean_actual = sum(actual) / len(actual)
        mean_predicted = sum(predicted) / len(predicted)
        range_squared = (max(actual) - min(actual)) ** 2
        if (NMSE_normalization_factor == 1):
            range_squared = sum((x - mean_actual) ** 2 for x in actual) / len(actual)
        # range_squared = mean_actual * mean_predicted
        if range_squared == 0:
            raise ValueError("The range of actual values is zero. NMSE cannot be calculated.")

        nmse = mse / range_squared

        mean_squared_errors = mse
        variance_squared_errors = sum((se - mean_squared_errors) ** 2 for se in squared_errors) / (
                    len(squared_errors) - 1)

        sd_mse = variance_squared_errors / len(squared_errors)

        se_mse = sd_mse ** 0.5
        se_mse = se_mse / range_squared
        return nmse, se_mse
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

    def whether_prefix_suffix(self, folder_path, prefix, suffix):
        if not os.path.exists(folder_path):
            raise ValueError(f"The folder path {folder_path} does not exist.")

        for file_name in os.listdir(folder_path):
            if file_name.startswith(prefix) and file_name.endswith(suffix):
                return True

        return False

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
    #self.get_data_ranking(data_address_lists[data_address_index], policy_name_list[0],run_list[runs][0],run_list[runs][1]))
    def get_data_ranking(self,experiment_name,method_name ,policy_name_list, evaluate_time=30,max_timestep=1000,gamma=0.99):
        Offline_data_folder = "Offline_data"
        #true_env_name,algorithm_trajectory_list,target_env_parameter_map,target_policy_parameter_map
        exp_parameter_path = os.path.join("Exp_result",experiment_name,method_name)


        policy_folder = os.path.join("Policy_operation","Policy_trained")

        policy_performance = []

        for i in range(len(policy_name_list)):
            policy_path= os.path.join(policy_folder,policy_name_list[i])
            policy = d3rlpy.load_learnable(policy_path+".d3",self.device)

            current_env_path = os.path.join("Exp_result",experiment_name,method_name,policy_name_list[i])
            current_env_name = self.load_from_pkl(current_env_path)
            # print(current_env_name)
            current_env_name = current_env_name[0]
            env = self.get_env(current_env_name)
            policy_performance.append(self.evaluate_policy( policy, env, evaluate_time,max_timestep,gamma))


        # Return the ranking of the policy names
        # print("policy performance : ",policy_performance)
        return self.rank_elements_larger_higher(policy_performance)

    # policy_name = policy_name, true_env_name = true_env_name, true_policy_name = true_policy_name

    def load_policy_performance(self, policy_name, true_env_name):

        policy_performance_path = os.path.join("Policy_operation", "Policy_performance",
                                                 true_env_name,policy_name)


        if os.path.exists(policy_performance_path):
            return self.load_from_pkl(policy_performance_path)
        else:
            env = self.get_env(true_env_name)
            policy_path = os.path.join("Policy_operation","Policy_trained",policy_name)
            policy = d3rlpy.load_learnable(policy_path+".d3",device=self.device)
            return self.evaluate_policy(policy=policy,env=env)

#(experiment_name=run_list[i],ranking_list = Ranking_list[num_index][i],
                                                         #  Policy_name_list = Policy_name_list[i],
                                                          # k=k)
    def calculate_top_k_precision(self, experiment_name, rank_list, Policy_name_list,k=5):
        device = self.device

        exp_parameter_path = os.path.join("Exp_result", experiment_name, "parameter")
        exp_parameter = self.load_from_pkl(exp_parameter_path)
        true_env_name = exp_parameter[0]

        policy_performance_list = [self.load_policy_performance(true_env_name=true_env_name, policy_name=policy_name) for
                                   policy_name in Policy_name_list]

        ground_truth = self.rank_elements_larger_higher(policy_performance_list)

        k_precision_list = []
        for i in range(1, k + 1):
            proportion = 0
            for pos in rank_list:
                if (rank_list[pos-1 ] <= i - 1 and ground_truth[pos-1 ] <= i - 1):
                    proportion += 1
            proportion = proportion / i
            k_precision_list.append(proportion)
        return k_precision_list


    def calculate_top_k_normalized_regret(self,experiment_name,rank_list, Policy_name_list, k=5):
        print("calcualte top k normalized regret")
        exp_parameter_path = os.path.join("Exp_result",experiment_name,"parameter")
        exp_parameter = self.load_from_pkl(exp_parameter_path)
        true_env_name = exp_parameter[0]

        policy_performance_list = [self.load_policy_performance(true_env_name = true_env_name,policy_name = policy_name) for policy_name in Policy_name_list]

        ground_truth_value = max(policy_performance_list)
        worth_value = min(policy_performance_list)
        if ((ground_truth_value - worth_value) == 0):
            return 99999
        k_regret_list = []
        for j in range(1, k + 1):
            gap_list = []
            for i in range(len(rank_list)):
                if (rank_list[i] <= j):
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
    def draw_mse_graph(self,combinations, means, colors, standard_errors, labels, folder_path,
                       filename="FQE_MSES.png", figure_name='Normalized MSE of FQE'):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        group_width = 1.0
        n_bars = len(labels)
        bar_width = group_width / n_bars

        indices = np.arange(1)

        fig, ax = plt.subplots()

        for i in range(n_bars):
            bar_x_positions = indices - (group_width - bar_width) / 2 + i * bar_width

            errors_below = standard_errors[i]
            errors_above = standard_errors[i]

            ax.bar(bar_x_positions, means[i], yerr=[[errors_below], [errors_above]],
                   capsize=5, alpha=0.7, color=colors[i], label=labels[i], width=bar_width)

        ax.set_ylabel('Normalized MSE')
        ax.set_title(figure_name )

        ax.set_xticks(indices)
        ax.set_xticklabels(combinations)

        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, filename))
        plt.close()
    def calculate_k(self, experiment_name_list,saving_folder_name, k ,plot_name_list,evaluate_time=30,max_timestep=1000,gamma=0.99):
        Ranking_list = []
        Policy_name_list = []


        Policy_name_list = []
        for i in range(len(experiment_name_list)):
            exp_parameter_path = os.path.join("Exp_result",experiment_name_list[i],"parameter")
            #true_env_name, algorithm_trajectory_list, target_env_parameter_map, target_policy_parameter_map
            para_list = self.load_from_pkl(exp_parameter_path)
            current_target_env_list,current_target_env_name_list = self.get_env_list(para_list[2])
            current_target_policy_parameters = self.generate_policy_parameter_tuples(current_target_env_name_list,para_list[3])
            policy_name_list = []
            for j in range(len(current_target_policy_parameters)):
                dudu = copy.deepcopy(current_target_policy_parameters[j])
                dudu[0] = current_target_env_name_list[current_target_policy_parameters[j][0]]
                policy_name_list.append(self.get_policy_name(*dudu))
            Policy_name_list.append(policy_name_list)

        run_list = []
        for i in range(len(experiment_name_list)):
            experiment_name = experiment_name_list[i]
            exp_parameter_path = os.path.join("Exp_result", experiment_name, "parameter")
            # true_env_name,algorithm_trajectory_list,target_env_parameter_map,target_policy_parameter_map
            exp_parameter = self.load_from_pkl(exp_parameter_path)

            true_env_name = exp_parameter[0]
            run_list.append(experiment_name_list[i])


        data_address_lists = copy.deepcopy(plot_name_list)
        for i in range(len(data_address_lists)):
            Ranking_list.append([])

        for runs in range(len(run_list)):
            for data_address_index in range(len(data_address_lists)):
                Ranking_list[data_address_index].append(
                    #当前algorithm的排名
                    self.get_data_ranking(experiment_name=run_list[runs],method_name = data_address_lists[data_address_index],
                                          policy_name_list=Policy_name_list[runs]))
            # print("data address lists : ",data_address_lists)
            # print(Ranking_list)
            # sys.exit()


        Precision_list = []
        Regret_list = []
        for index in range(len(data_address_lists)):
            Precision_list.append([])
            Regret_list.append([])

        for i in range(len(run_list)):
            for num_index in range(len(Ranking_list)):

                Precision_list[num_index].append(
                    self.calculate_top_k_precision(experiment_name=run_list[i],rank_list=Ranking_list[num_index][i],Policy_name_list=Policy_name_list[i],k=k))
                Regret_list[num_index].append(
                    self.calculate_top_k_normalized_regret(experiment_name=run_list[i],rank_list = Ranking_list[num_index][i],
                                                           Policy_name_list = Policy_name_list[i],
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


        policy_ranking_saving_place = 'Exp_result'
        k_saving_folder = 'K_statistics'
        saving_folder = os.path.join(policy_ranking_saving_place, k_saving_folder)
        saving_path = os.path.join(saving_folder,saving_folder_name)

        self.create_folder(saving_path)
        # Bvft_k_save_path = os.path.join(Bvft_saving_place, Bvft_k)
        # if not os.path.exists(Bvft_k_save_path):
        #     os.makedirs(Bvft_k_save_path)
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        plot_name = "plots"
        k_precision_name = str(k) + "_mean_precision_" + str(len(experiment_name_list))
        k_regret_name = str(k) + "_mean_regret" + str(len(experiment_name_list))
        precision_ci_name = str(k) + "_CI_precision" + str(len(experiment_name_list))
        regret_ci_name = str(k) + "_CI_regret" + str(len(experiment_name_list))

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
    def get_NMSE(self,experiment_name,algorithm_name,normalization_factor):
        print("Plot FQE MSE")


        policy_folder = os.path.join("Policy_operation","Policy_trained")
        self.create_folder(policy_folder)

        policy_performance_folder = os.path.join("Policy_operation","Policy_performance")
        policy_total_dictionary = self.load_from_pkl(policy_performance_folder)

        true_list = []
        prediction_list = []

        exp_parameter_path = os.path.join("Exp_result", experiment_name, "parameter")
        # true_env_name, algorithm_trajectory_list, target_env_parameter_map, target_policy_parameter_map
        para_list = self.load_from_pkl(exp_parameter_path)
        current_target_env_list, current_target_env_name_list = self.get_env_list(para_list[2])
        current_target_policy_parameters = self.generate_policy_parameter_tuples(current_target_env_name_list,
                                                                                 para_list[3])

        true_env_name = para_list[0]
        policy_performance_env_folder = os.path.join(policy_performance_folder, true_env_name)

        target_policy_folder = os.path.join("Exp_result",experiment_name,algorithm_name)

        for policy_file_name in os.listdir(target_policy_folder):
            policy_name = policy_file_name[:-4]
            env_path = os.path.join(target_policy_folder,policy_name)

            current_target_env = self.load_from_pkl(env_path)
            prediction_list.append(self.load_policy_performance(policy_name,current_target_env))
            true_list.append(self.load_policy_performance(policy_name,true_env_name))
        NMSE, standard_error = self.normalized_mean_square_error_with_error_bar(true_list, prediction_list,
                                                                           normalization_factor)
        return NMSE, standard_error

    ##true_env_name,algorithm_trajectory_list,target_env_parameter_map,target_policy_parameter_map
    def draw_figure_6R(self,saving_folder_name,experiment_name_list,method_name_list,normalization_factor):
        means = []
        SE = []
        labels = []
        self_data_saving_path = method_name_list
        for i in range(len(experiment_name_list)):
            for j in range(len(self_data_saving_path)):
                repo_name = self_data_saving_path[j]
                experiment_name = experiment_name_list[i]
                NMSE,standard_error = self.get_NMSE(experiment_name=experiment_name,
                                                    algorithm_name = repo_name)
                means.append(NMSE)
                SE.append(standard_error)
                labels.append(self_data_saving_path[j]+"_"+experiment_name_list[i])
        name_list = ['hopper-v4']

        Figure_saving_path = os.path.join("Exp_result", "K_statistics",saving_folder_name,"Figure_6L_plot")
        #
        colors = self.generate_unique_colors(len(self_data_saving_path))
        figure_name = 'Normalized e MSE of FQE min max'
        filename = "Figure6R_max_min_NMSE_graph"
        if normalization_factor == 1:
            figure_name = 'Normalized MSE of FQE groundtruth variance'
            filename = "Figure6R_groundtruth_variance_NMSE_graph"
        self.draw_mse_graph(combinations=name_list, means=means, colors=colors, standard_errors=SE,
                       labels=labels, folder_path=Figure_saving_path,
                       filename=filename, figure_name=figure_name)
    def draw_figure_6L(self,saving_folder_name,experiment_name_list,method_name_list,k=5,evaluate_time=30,max_timestep=1000,gamma=0.99):
        Result_saving_place = 'Exp_result'
        Result_k = 'K_statistics'
        Result_k_save_path = os.path.join(Result_saving_place,Result_k,saving_folder_name)
        self.create_folder(Result_k_save_path)

        num_runs = str(experiment_name_list)

        k_precision_name = str(k) + "_mean_precision_" + str(num_runs)
        k_regret_name = str(k) + "_mean_regret_" + str(num_runs)
        precision_ci_name = str(k) + "_CI_precision_" + str(num_runs)
        regret_ci_name = str(k) + "_CI_regret_" + str(num_runs)
        plot_name = "plots"

        k_precision_mean_saving_path = os.path.join(Result_k_save_path, k_precision_name)
        k_regret_mean_saving_path = os.path.join(Result_k_save_path, k_regret_name)
        k_precision_ci_saving_path = os.path.join(Result_k_save_path, precision_ci_name)
        k_regret_ci_saving_path = os.path.join(Result_k_save_path, regret_ci_name)
        plot_name_saving_path = os.path.join(Result_k_save_path, plot_name)

        # if os.path.exists(k_precision_mean_saving_path):
        #     precision_mean_list = self.load_from_pkl(k_precision_mean_saving_path)
        #     regret_mean_list = self.load_from_pkl(k_regret_mean_saving_path)
        #     precision_ci_list = self.load_from_pkl(k_precision_ci_saving_path)
        #     regret_ci_list = self.load_from_pkl(k_regret_ci_saving_path)
        #     line_name_list = self.load_from_pkl(plot_name_saving_path)
        # else:
        precision_mean_list, regret_mean_list, precision_ci_list, regret_ci_list, line_name_list = self.calculate_k(
           experiment_name_list = experiment_name_list, saving_folder_name = saving_folder_name,k = k,plot_name_list = method_name_list,
            evaluate_time =evaluate_time,max_timestep = max_timestep, gamma=gamma
        )

        plot_mean_list = [precision_mean_list, regret_mean_list]
        plot_ci_list = [precision_ci_list, regret_ci_list]

        plot_folder = os.path.join(Result_k_save_path, "Figure_6L_plot")
        self.create_folder(plot_folder)

        y_axis_names = ["k precision", "k regret"]
        colors = self.generate_unique_colors(len(plot_mean_list[0]))

        self.plot_subplots(data=plot_mean_list, save_path=plot_folder, y_axis_names=y_axis_names,
                           line_names=line_name_list, colors=colors, ci=plot_ci_list)
    def plot_subplots(self,data, save_path, y_axis_names, line_names, colors, ci):
        num_subplots = len(data)
        fig, axes = plt.subplots(num_subplots, figsize=(10, 5 * num_subplots), squeeze=False)
        print(data)
        print(ci)
        for i, subplot_data in enumerate(data):

            for j, line_data in enumerate(subplot_data):
                # print("len line data : ", len(line_data))
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
        saving_path = "regret_precision.png"
        saving_path = os.path.join(save_path, saving_path)
        plt.savefig(saving_path)
        plt.close()

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
    def generate_one_trajectory(self, env, policy, unique_seed,Offline_trajectory_max_timestep=1000):
        obs, info = env.reset(seed=unique_seed)

        observations = []
        rewards = []
        actions = []
        dones = []
        next_steps = []
        episode_data = {}
        observations.append(obs)
        total_state_number = 1  # Initialize the state counter

        for _ in range(Offline_trajectory_max_timestep):
            action = policy.predict(np.array([obs]))[0]
            ui = env.step(action)
            next_obs, reward, done = ui[0],ui[1],ui[2]
            rewards.append(reward)
            actions.append(action)
            dones.append(done)
            next_steps.append(next_obs)

            if done or _ == Offline_trajectory_max_timestep - 1:
                break

            obs = next_obs
            observations.append(obs)
            total_state_number += 1


        episode_data = {
            "observations": observations,
            "rewards": rewards,
            "actions": actions,
            "dones": dones,
            "next_steps": next_steps,
            "total_state_number": total_state_number
        }

        return episode_data
    def generate_trajectory(self,env,trajectory_number,policy,Offline_trajectory_max_timestep=1000):
        trajectories = []
        max_timestep = Offline_trajectory_max_timestep
        total_sa = 0
        for _ in range(trajectory_number):
            unique_seed = self.generate_unique_seed()
            episode = self.generate_one_trajectory( env, policy, unique_seed,Offline_trajectory_max_timestep=Offline_trajectory_max_timestep)
            trajectories.append(episode)
            total_sa += episode["total_state_number"]

        return trajectories,total_sa
    def generate_offline_data(self, trajectory_number, true_environment_parameter_list, behaviroal_policy_parameter_map,
                              Offline_trajectory_max_timestep=1000):
        folder_name = "Offline_data"
        env_list, env_name_list = self.get_env_list(true_environment_parameter_list)
        parameter_tuples = self.generate_policy_parameter_tuples(env_name_list, behaviroal_policy_parameter_map)
        unique_id = time.strftime("%Y%m%d-%H%M%S")
        for params in parameter_tuples:
            env = env_list[params[0]]
            params[0] = env_name_list[params[0]]
            env_name = params[0]
            policy_name = self.get_policy_name(*params)
            trajectory_folder = os.path.join(folder_name,env_name,policy_name)
            self.create_folder(trajectory_folder)

            print(f"start generate environment   {env_name}  \n"
                  f"policy {policy_name}  \n"
                  f"total trajectory : {trajectory_number} \n"
                  f"time id series : {unique_id}")
            policy = self.get_policy(*params)
            dataset,total_sa = self.generate_trajectory(env=env,trajectory_number = trajectory_number,
                                                        policy=policy,
                                                        Offline_trajectory_max_timestep=Offline_trajectory_max_timestep)

            trajectory_name = f"{trajectory_number}_trajectory_{unique_id}"
            # trajectory_name = f"{trajectory_number}_trajectory_{total_sa}_states"
            trajectory_path = os.path.join(trajectory_folder,trajectory_name)
            self.save_as_pkl(trajectory_path,dataset)
            self.save_as_txt(trajectory_path,dataset)
            print(f"finish generate environment   {env_name}  \n"
                  f"policy {policy_name}  \n"
                  f"total trajectory : {trajectory_number} \n"
                  f"time id series : {unique_id}")







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

    def train_policy(self,env_parameter_map,target_policy_training_parameter_map):
        Policy_operation_folder = "Policy_operation"
        Policy_saving_folder = os.path.join(Policy_operation_folder,"Policy_trained")
        self.create_folder(Policy_saving_folder)
        Policy_checkpoints_folder = os.path.join(Policy_operation_folder,"Policy_checkpoints")
        self.create_folder(Policy_checkpoints_folder)
        # while(True):
        env_list, env_name_list = self.get_env_list(env_parameter_map)
        for i in range(len(env_name_list)):
            current_env = env_list[i]
            policy_folder_name = env_name_list[i]

            print(f"start training {policy_folder_name} with algorithm {str(target_policy_training_parameter_map['algorithm_name_list'])}")
            policy_saving_path = os.path.join(Policy_saving_folder, policy_folder_name)
            policy_checkpoints_path = os.path.join(Policy_checkpoints_folder, policy_folder_name)

            self.create_folder(policy_checkpoints_path)
            num_epoch = int(target_policy_training_parameter_map['policy_total_step'] / target_policy_training_parameter_map['policy_episode_step'])
            buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=1500000, env=current_env)
            explorer = d3rlpy.algos.ConstantEpsilonGreedy(0.3)
            checkpoint_list = []
            for algorithm_name in target_policy_training_parameter_map['algorithm_name_list']:
                checkpoint_path = os.path.join(policy_checkpoints_path,f"{algorithm_name}_checkpoints.d3")
                checkpoint_list_path = os.path.join(policy_checkpoints_path, f"{algorithm_name}_checkpoints")
                policy_model_name = f"{algorithm_name}_{str(target_policy_training_parameter_map['policy_learning_rate'])}_{str(target_policy_training_parameter_map['policy_hidden_layer'])}.d3"
                policy_path = policy_saving_path+"_"+policy_model_name

                if((not self.whether_prefix_suffix(Policy_saving_folder,policy_folder_name+"_"+policy_model_name[:-3],"step.pkl") ) and
                        (not self.whether_file_exists(policy_path[:-3]+"_"+str(target_policy_training_parameter_map["policy_total_step"])+"step.d3"))):
                    self.save_as_pkl(policy_path[:-3]+"step",2)
                    print(f"{policy_path} not exists")
                    if (self.whether_file_exists(checkpoint_path)):
                        policy = d3rlpy.load_learnable(checkpoint_path, device=self.device)
                        checkpoint_list = self.load_from_pkl(checkpoint_list_path)
                        print(f"enter self checkpoints {checkpoint_path} with epoch {str(checkpoint_list[-1])}")
                        for epoch in range(checkpoint_list[-1] + 1, int(num_epoch)):
                            policy.fit_online(env=current_env,
                                              buffer=buffer,
                                              explorer=explorer,
                                              n_steps=target_policy_training_parameter_map['policy_episode_step'],
                                              eval_env=current_env,
                                              with_timestamp=False,
                                              )
                            policy.save(checkpoint_path)
                            checkpoint_list.append(epoch)
                            self.save_as_pkl(checkpoint_list_path, checkpoint_list)
                            if ((epoch + 1) % target_policy_training_parameter_map['policy_saving_number'] == 0):
                                policy.save(policy_path[:-3] + "_" + str(
                                    (epoch + 1) * target_policy_training_parameter_map['policy_episode_step']) + "step" + ".d3")
                    else:
                        self_class = getattr(d3rlpy.algos, algorithm_name + "Config")
                        policy = self_class(
                            actor_encoder_factory=d3rlpy.models.VectorEncoderFactory(
                                hidden_units=target_policy_training_parameter_map['policy_hidden_layer']),
                            critic_encoder_factory=d3rlpy.models.VectorEncoderFactory(
                                hidden_units=target_policy_training_parameter_map['policy_hidden_layer']),
                            actor_learning_rate=target_policy_training_parameter_map['policy_learning_rate'],
                            critic_learning_rate=target_policy_training_parameter_map['policy_learning_rate'],
                        ).create(device=self.device)
                        for epoch in range(num_epoch):
                            policy.fit_online(env=current_env,
                                              buffer=buffer,
                                              explorer=explorer,
                                              n_steps=target_policy_training_parameter_map['policy_episode_step'],
                                              eval_env=current_env,
                                              with_timestamp=False,
                                              )
                            policy.save(checkpoint_path)
                            checkpoint_list.append(epoch)
                            self.save_as_pkl(checkpoint_list_path, checkpoint_list)
                            if ((epoch + 1) % target_policy_training_parameter_map['policy_saving_number'] == 0):
                                policy.save(policy_path[:-3] + "_" + str(
                                    (epoch + 1) * target_policy_training_parameter_map['policy_episode_step']) + "step" + ".d3")
                    self.delete_as_pkl(policy_path[:-3]+"step")
                    if os.path.exists(checkpoint_list_path + ".pkl"):
                        os.remove(checkpoint_list_path + ".pkl")
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                    print(f"end training {policy_folder_name} with algorithm {str(target_policy_training_parameter_map['algorithm_name_list'])}")


    def get_policy_parameters_from_map(self,policy_parameter_map):

        return  (policy_parameter_map["policy_total_step"],policy_parameter_map["policy_episode_step"],policy_parameter_map["policy_saving_number"],policy_parameter_map["policy_learning_rate"],

                 policy_parameter_map["policy_hidden_layer"], policy_parameter_map["algorithm_name_list"])

    def get_trained_policy_path(self,env_name,algorithm_name,learning_rate,hidden_layer,total_step):
        Policy_operation_folder = "Policy_operation"
        Policy_performance = os.path.join(Policy_operation_folder, "Policy_performance")
        Policy_performance_folder = os.path.join(Policy_performance, env_name)
        self.create_folder(Policy_performance_folder)

        policy_name = self.get_policy_name(env_name, algorithm_name, learning_rate, hidden_layer, total_step)
        policy_performance_path = os.path.join(Policy_performance_folder, policy_name)
        return policy_performance_path

    def generate_policy_parameter_tuples(self,env_name_list, policy_parameter_map):
        parameter_tuples = []
        for env_name_index, env_name in enumerate(env_name_list):
            for algorithm_name in policy_parameter_map["algorithm_name_list"]:
                for learning_rate in [policy_parameter_map["policy_learning_rate"]]:
                    for hidden_layer in [policy_parameter_map["policy_hidden_layer"]]:
                        for total_step in [policy_parameter_map["policy_total_step"]]:
                            parameter_tuples.append(
                                [env_name_index, algorithm_name, learning_rate, hidden_layer, total_step])
        return parameter_tuples
    def train_policy_performance(self,evaluate_env_parameter_map,policy_env_parameter_map,policy_parameter_map,policy_evaluation_parameter_map):
        evaluate_env_list, evaluate_env_name_list = self.get_env_list(evaluate_env_parameter_map)
        env_list, env_name_list = self.get_env_list(policy_env_parameter_map)
        parameter_tuples = self.generate_policy_parameter_tuples(env_name_list,policy_parameter_map)
        for params in parameter_tuples:
            for env in evaluate_env_list:
                input_tup = copy.deepcopy(params)
                input_tup[0]= env_name_list[params[0]]
                policy_performance_path = self.get_trained_policy_path(*input_tup)
                performance = [0]
                if not os.path.exists(policy_performance_path + '.pkl'):
                    self.save_as_pkl(policy_performance_path, performance)
                    self.save_as_txt(policy_performance_path, performance)
                    performance = self.load_from_pkl(policy_performance_path)
                    policy = self.get_policy(*input_tup)
                    performance[0] = self.evaluate_policy(policy=policy, env=env, **policy_evaluation_parameter_map)
                    # print(final_result_dict)
                    self.save_as_pkl(policy_performance_path, performance)
                    self.save_as_txt(policy_performance_path, performance)
                    print(f"finished evaluated policy {policy_performance_path}")
                else:
                    print(f"already exist policy performance {policy_performance_path}")





    def get_policy_performance(self,env_name,algorithm_name,learning_rate,hidden_layer,total_step,evaluate_time=30,
                               max_timestep=1000,
                               gamma=0.99):
        Policy_operation_folder = "Policy_operation"
        Policy_performance = os.path.join(Policy_operation_folder, "Policy_performance")
        Policy_performance_folder = os.path.join(Policy_performance,env_name)
        self.create_folder(Policy_performance_folder)

        policy_name = self.get_policy_name(env_name,algorithm_name,learning_rate,hidden_layer,total_step)
        policy_performance_path = os.path.join(Policy_performance_folder, policy_name)
        performance = 0
        if os.path.exists(policy_performance_path + '.pkl'):
            performance = self.load_from_pkl(policy_performance_path)
        else:
            env = gymnasium.make(env_name)
            policy = self.get_policy(env_name,algorithm_name,learning_rate,hidden_layer,total_step)
            performance = self.evaluate_policy(policy=policy, env=env, evaluate_time=evaluate_time,max_timestep=max_timestep,gamma=gamma)
            #print(final_result_dict)
            self.save_as_pkl(policy_performance_path, performance)
            self.save_as_txt(policy_performance_path, performance)
        return performance

    def evaluate_policy(self, policy, env, evaluate_time=30,max_timestep=1000,gamma=0.99):
        total_rewards = 0

        for i in range(evaluate_time):
            rewards = self.evaluate_single(env=env,policy=policy,max_timestep=max_timestep,gamma=gamma)
            total_rewards += rewards


        return total_rewards /  evaluate_time

    def evaluate_single(self, env,policy,max_timestep,gamma):
        obs, info = env.reset()
        total_rewards = 0
        discount_factor = 1
        max_iteration = max_timestep

        for _ in range(max_iteration):
            action = policy.predict(np.array([obs]))[0]
            ui = env.step(action)
            obs, reward, done = ui[0], ui[1],ui[2]
            total_rewards += reward * discount_factor
            discount_factor *= gamma

            if done:
                break

        return total_rewards

    def run_simulation(self, state_action_policy_env_batch,max_timestep=1000,gamma=0.99):
        states, actions, policy, envs = state_action_policy_env_batch
        for env, state in zip(envs, states):
            env.reset()
            env.observation = state

        total_rewards = np.zeros(len(states))
        num_steps = np.zeros(len(states))
        discount_factors = np.ones(len(states))
        done_flags = np.zeros(len(states), dtype=bool)

        while not np.all(done_flags) and np.any(num_steps < max_timestep):
            actions_batch = policy.predict(np.array(states))
            for idx, (env, action) in enumerate(zip(envs, actions_batch)):
                if not done_flags[idx]:
                    next_state, reward, done, _, _ = env.step(action)
                    total_rewards[idx] += reward * discount_factors[idx]
                    discount_factors[idx] *= gamma
                    num_steps[idx] += 1
                    states[idx] = next_state
                    done_flags[idx] = done

        return total_rewards

    def load_offline_data(self,env_name,Policy_name,offline_trajectory_name):
        file_path = os.path.join("Offline_data",env_name,Policy_name,offline_trajectory_name[:-4])
        return self.load_from_pkl(file_path)

    def calculate_r_plus_vfsp(self, q_sa, q_prime, dataset, gamma=0.99):
        r_plus_vfsp = []
        for i in range(len(q_sa)):
            r_plus_vfsp.append([])
        for h in range(len(q_sa)):
            index = 0

            for j in range(len(dataset)):
                length = len(dataset[j]["observations"])
                rewards = dataset[j]["rewards"]
                dones = dataset[j]["dones"]
                for i in range(length):
                    reward = rewards[i]
                    q_p = q_prime[h][index]
                    done = dones[i]
                    if (done):
                        done = 1
                    else:
                        done = 0
                    r_plus_vfsp[h].append(reward + gamma * q_p * (1 - done))
                    index += 1

        return r_plus_vfsp

    def create_deep_copies(self, env, batch_size):
        env_copies = [copy.deepcopy(env) for _ in range(batch_size)]
        return env_copies

    def train_qa(self, offline_trajectory_name, data_env_name, behaviroal_policy_parameter, target_env_parameter_map,
                 target_policy_parameter, batch_size=100, max_timestep=1000, gamma=0.99):
        behaviroal_policy_name = self.get_policy_name(*behaviroal_policy_parameter)
        policy = self.get_policy(*target_policy_parameter)

        folder_path = os.path.join("Offline_data", data_env_name, behaviroal_policy_name)
        self.create_folder(folder_path)

        data_file_path = os.path.join(folder_path, offline_trajectory_name)
        target_env_list, target_env_name_list = self.get_env_list(target_env_parameter_map)

        if self.whether_file_exists(data_file_path):
            trajectory = self.load_from_pkl(data_file_path[:-4])
            total_leng = sum(traj["total_state_number"] for traj in trajectory)

            print("len total : ", total_leng)

            for h, target_env_name in enumerate(target_env_name_list):
                prifix = f"{target_env_name}_Policy_{self.get_policy_name(*target_policy_parameter)}_{offline_trajectory_name[:-4]}q.pkl"
                dudu = os.path.join(folder_path, prifix)

                if not (self.whether_file_exists(dudu) or self.whether_file_exists(dudu[:-5] + "_q.pkl")):
                    print("enter not exists")
                    env_copy_list = self.create_deep_copies(env=target_env_list[h], batch_size=batch_size)
                    check_saving_path = os.path.join(folder_path, prifix)
                    self.save_as_pkl(check_saving_path, [1])

                    qa_results = []
                    q_prime_results = []

                    for traj in trajectory:
                        total_len = len(traj["observations"])
                        states = traj["observations"]
                        actions = traj["actions"]
                        next_states = traj["next_steps"]
                        next_actions = policy.predict(np.array(next_states))

                        for j in range(0, total_len, batch_size):
                            actual_batch_size = min(batch_size, total_len - j)
                            state_action_policy_env_pairs = (
                                states[j:j + actual_batch_size], actions[j:j + actual_batch_size], policy,
                                env_copy_list[:actual_batch_size])
                            batch_results = self.run_simulation(state_action_policy_env_pairs,
                                                                max_timestep=max_timestep, gamma=gamma)

                            next_state_action_policy_env_pairs = (
                                next_states[j:j + actual_batch_size], next_actions[j:j + actual_batch_size], policy,
                                env_copy_list[:actual_batch_size])
                            prime_results = self.run_simulation(next_state_action_policy_env_pairs,
                                                                max_timestep=max_timestep, gamma=gamma)

                            qa_results.extend(batch_results)
                            q_prime_results.extend(prime_results)

                    target_policy_name = self.get_policy_name(*target_policy_parameter)

                    qa_saving_name = f"{target_env_name}_Policy_{target_policy_name}_{offline_trajectory_name[:-4]}_q"
                    qa_save_path = os.path.join("Offline_data", data_env_name, behaviroal_policy_name, qa_saving_name)

                    prime_saving_name = f"{target_env_name}_Policy_{target_policy_name}_{offline_trajectory_name[:-4]}_q_prime"
                    prime_saving_path = os.path.join("Offline_data", data_env_name, behaviroal_policy_name,
                                                     prime_saving_name)

                    self.delete_as_pkl(check_saving_path)
                    self.save_as_pkl(qa_save_path, np.array(qa_results))
                    self.save_as_pkl(prime_saving_path, np.array(q_prime_results))

                    print(f"finish train q(s, a) , q (s', pi(a)) for env {target_env_name_list[h]} \n "
                          f"behaviroal policy {behaviroal_policy_name} \n"
                          f"dataset name {offline_trajectory_name} \n"
                          f"target policy name {self.get_policy_name(*target_policy_parameter)} \n \n")

                else:
                    print(f" q(s, a) , q (s', pi(a)) for env {target_env_name_list[h]} \n "
                          f"behaviroal policy {behaviroal_policy_name} \n"
                          f"dataset name {offline_trajectory_name} \n"
                          f"target policy name {self.get_policy_name(*target_policy_parameter)} \n"
                          f"already Exist \n \n")

        else:
            print(f"  Data file: env {data_env_name} \n "
                  f"behaviroal policy {behaviroal_policy_name} \n"
                  f"dataset name {offline_trajectory_name} "
                  f"does not exist \n \n")

    def train_whole_qa(self, offline_trajectory_name_list, behavioral_env_parameter_map,
                       behavioral_policy_parameter_map, target_env_parameter_map,
                       target_parameter_map, batch_size=100, max_timestep=1000,
                       gamma=0.99):
        behavioral_env_list, behavioral_env_name_list = self.get_env_list(behavioral_env_parameter_map)
        behavioral_parameter = self.generate_policy_parameter_tuples(behavioral_env_name_list,
                                                                     behavioral_policy_parameter_map)

        target_env_list, target_env_name_list = self.get_env_list(target_env_parameter_map)
        target_parameter = self.generate_policy_parameter_tuples(target_env_name_list, target_parameter_map)

        all_combinations = product(behavioral_parameter, target_parameter, offline_trajectory_name_list)

        for behavioral_param, target_param, trajectory_name in all_combinations:
            behavioral_env_name = behavioral_env_name_list[behavioral_param[0]]
            behavioral_input = copy.deepcopy(behavioral_param)
            behavioral_input[0] = behavioral_env_name

            target_env_name = target_env_name_list[target_param[0]]
            target_input = copy.deepcopy(target_param)
            target_input[0] = target_env_name


            self.train_qa(offline_trajectory_name=trajectory_name,
                          data_env_name=behavioral_env_name,
                          behaviroal_policy_parameter=behavioral_input,
                          target_env_parameter_map = target_env_parameter_map,
                          target_policy_parameter=target_input,
                          batch_size=batch_size,
                          max_timestep=max_timestep,
                          gamma=gamma)

    def train_one_qa(self, environment,environment_name,dataset_path,  target_policy_parameter, batch_size=100, max_timestep=1000,
                     gamma=0.99):
        trajectory = self.load_from_pkl(dataset_path[:-4])
        qa_results = []
        q_prime_results = []
        policy = self.get_policy(*target_policy_parameter)

        env_name_parts = dataset_path.split(os.sep)
        data_env_name = env_name_parts[1]
        behavioral_policy_name = env_name_parts[2]
        offline_trajectory_name = env_name_parts[3]

        env_copy_list = self.create_deep_copies(env=environment, batch_size=batch_size)

        folder_path = os.path.join("Offline_data", data_env_name, behavioral_policy_name)
        self.create_folder(folder_path)

        print(f"start train q(s, a) , q (s', pi(a)) for env {environment_name} \n"
              f"behaviroal policy {behavioral_policy_name} \n"
              f"dataset name {offline_trajectory_name} \n"
              f"target policy name {self.get_policy_name(*target_policy_parameter)}")

        for i in range(len(trajectory)):
            total_len = len(trajectory[i]["observations"])
            states = trajectory[i]["observations"]
            actions = trajectory[i]["actions"]

            next_states = trajectory[i]["next_steps"]
            next_actions = policy.predict(np.array(next_states))
            for j in range(0, total_len, batch_size):
                actual_batch_size = min(batch_size, total_len - j)
                state_action_policy_env_pairs = (
                    states[j:j + actual_batch_size], actions[j:j + actual_batch_size], policy,
                    env_copy_list[:actual_batch_size])
                batch_results = self.run_simulation(state_action_policy_env_pairs, max_timestep=max_timestep,
                                                    gamma=gamma)

                next_state_action_policy_env_pairs = (
                next_states[j:j + actual_batch_size], next_actions[j:j + actual_batch_size], policy,
                env_copy_list[:actual_batch_size])
                prime_results = self.run_simulation(next_state_action_policy_env_pairs, max_timestep=max_timestep,
                                                    gamma=gamma)

                qa_results.extend(batch_results)
                q_prime_results.extend(prime_results)

        target_policy_name = self.get_policy_name(*target_policy_parameter)
        qa_saving_name = f"{environment_name}_Policy_{target_policy_name}_{offline_trajectory_name[:-4]}_q"
        qa_save_path = os.path.join("Offline_data", data_env_name, behavioral_policy_name, qa_saving_name)

        prime_saving_name = f"{environment_name}_Policy_{target_policy_name}_{offline_trajectory_name[:-4]}_q_prime"
        prime_saving_path = os.path.join("Offline_data", data_env_name, behavioral_policy_name, prime_saving_name)

        self.save_as_pkl(qa_save_path, qa_results)
        self.save_as_pkl(prime_saving_path, q_prime_results)


        print(f"finish train q(s, a) , q (s', pi(a)) for env {data_env_name} \n"
              f"behaviroal policy {behavioral_policy_name} \n"
              f"dataset name {offline_trajectory_name} \n"
              f"target policy name {self.get_policy_name(*target_policy_parameter)}")
    def get_data_q(self,environment_name,algorithm_trajectory_list,target_env_parameter_map,target_policy_parameter_map):
        parameter_tuples = self.generate_policy_parameter_tuples([environment_name], algorithm_trajectory_list[0])
        q_list = []
        q_prime_list = []
        data = []
        target_env_list, target_env_name_list = self.get_env_list(target_env_parameter_map)
        target_tuples = self.generate_policy_parameter_tuples(target_env_name_list,target_policy_parameter_map)
        input_target_tuple = copy.deepcopy(target_tuples)
        target_policy_name_list = []

        for hi in range(len(target_tuples)):
            for _ in range(len(target_env_list)):
                q_list.append([])
                q_prime_list.append([])
            current_name = target_env_name_list[target_tuples[hi][0]]
            input_target_tuple[hi][0] = current_name

        for i in range(len(parameter_tuples)):
            input = parameter_tuples[i].copy()
            input[0] = environment_name
            current_policy_name = self.get_policy_name(*input)

            for j in range(len(algorithm_trajectory_list[1])):
                current_offline_data_name = algorithm_trajectory_list[1][j]
                current_offline_data_path = os.path.join("Offline_data",environment_name,current_policy_name,current_offline_data_name)
                if i == 0  and j == 0:
                    data = self.load_from_pkl(current_offline_data_path[:-4])
                else:
                    traj = self.load_from_pkl(current_offline_data_path[:-4])
                    for i in range(len(traj)):
                        data.append(traj[i])
                for h in range(len(input_target_tuple)):
                    for u in range(len(target_env_name_list)):
                        current_target_policy_name = self.get_policy_name(*input_target_tuple[h])
                        if u == 0 :
                            target_policy_name_list.append(current_target_policy_name)
                        target_q_name = f"{target_env_name_list[u]}_Policy_{current_target_policy_name}_{current_offline_data_name[:-4]}_q"
                        target_prime_name = target_q_name + "_prime"
                        target_q_path = os.path.join("Offline_data", environment_name, current_policy_name,
                                                     target_q_name)
                        target_prime_path = os.path.join("Offline_data", environment_name, current_policy_name,
                                                         target_prime_name)
                        target_environment_name = copy.deepcopy(target_env_name_list[u])
                        target_environment = copy.deepcopy(target_env_list[u])
                        if not self.whether_file_exists(target_q_path + ".pkl"):
                            self.train_one_qa(environment=target_environment,environment_name=target_environment_name,dataset_path=current_offline_data_path,
                                              target_policy_parameter=input_target_tuple[h])
                            q_list[h*len(target_env_name_list)+u].extend(self.load_from_pkl(target_q_path))
                            q_prime_list[h*len(target_env_name_list)+u].extend(self.load_from_pkl(target_prime_path))
                        else:
                            print("targetq   length : ",len(self.load_from_pkl(target_q_path)))
                            q_list[h*len(target_env_name_list)+u].extend(self.load_from_pkl(target_q_path))
                            q_prime_list[h*len(target_env_name_list)+u].extend(self.load_from_pkl(target_prime_path))
        return q_list,q_prime_list,data,target_policy_name_list







    #true_env_parameter_map, [behaviroal_map, [offline_trajectory_name]]
    def get_ranking(self, experiment_name, ranking_method_name, true_env_name,
                    algorithm_trajectory_list,
                    target_env_parameter_map, target_policy_parameter_map,gamma=0.99):
        Bvft_folder = "Bvft_Records"
        Q_result_folder = "Exp_result"

        exp_folder = experiment_name
        parameter_path = os.path.join(Q_result_folder,exp_folder,"parameter")
        self.create_folder(os.path.join(Q_result_folder,exp_folder))

        self.save_as_pkl(parameter_path,[true_env_name,algorithm_trajectory_list,target_env_parameter_map,target_policy_parameter_map])
        self.save_as_txt(parameter_path, [true_env_name, algorithm_trajectory_list, target_env_parameter_map,
                                          target_policy_parameter_map])

        target_env_list, target_env_name_list = self.get_env_list(target_env_parameter_map)

        saving_folder = os.path.join(Q_result_folder,exp_folder,ranking_method_name)
        self.create_folder(saving_folder)

        q_list,q_prime_list,data,target_policy_name_list = self.get_data_q(environment_name=true_env_name,
                                                   algorithm_trajectory_list=algorithm_trajectory_list,
                                                   target_env_parameter_map=target_env_parameter_map,
                                                   target_policy_parameter_map=target_policy_parameter_map)


        for j in range(len(target_policy_name_list)):
            policy_name = target_policy_name_list[j]
            Q_result_saving_path = os.path.join(saving_folder, policy_name)
            q_sa = []
            q_prime = []
            for h in range(len(target_env_name_list)):
                q_sa.append(np.array(q_list[j*len(target_env_name_list)+h]))
                q_prime.append(np.array(q_prime_list[j*len(target_env_name_list)+h]))

            env = target_env_list[0]
            line_name_list = []
            for i in range(len(target_env_name_list)):
                line_name_list.append(target_env_name_list[i]+"_Policy_"+policy_name)
            self.create_folder(os.path.join("Exp_result",experiment_name,"BVFT_res_plot"))
            res_plot_save_path = os.path.join("Exp_result",experiment_name,"BVFT_res_plot",policy_name + ".png")
            result = self.select_Q(q_list=q_sa, q_prime=q_prime, policy_namei=policy_name,dataset=data,env=env,gamma=gamma,line_name_list=line_name_list,res_plot_save_path =res_plot_save_path )
            index = np.argmin(result)
            save_list = [target_env_name_list[index]]

            self.save_as_txt(Q_result_saving_path, save_list)
            self.save_as_pkl(Q_result_saving_path, save_list)
            self.delete_files_in_folder(Bvft_folder)









