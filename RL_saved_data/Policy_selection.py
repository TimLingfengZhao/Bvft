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
from BvftUtil import *
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
import numpy as np
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
class policy_select(ABC):

    def __init__(self,device,data_list,data_name_self, whole_dataset,train_episodes,test_episodes,test_data,replay_buffer,env,k,num_runs,FQE_saving_step_list,
                 gamma,initial_state,normalization_factor):
        self.device = device
        self.env = env
        self.data_saving_path = data_list
        self.data_name = data_name_self
        self.k = k
        self.num_runs = num_runs
        self.FQE_saving_step_list = FQE_saving_step_list
        self.whole_dataset = whole_dataset
        self.train_episodes = train_episodes
        self.test_episodes = test_episodes
        self.test_data = test_data
        self.initial_state = initial_state
        self.normalization_factor = normalization_factor
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.trajectory_num = len(self.test_episodes)
        self.data_size = self.get_data_size(test_episodes)

    def draw_Bvft_resolution_loss_graph(self,Bvft_final_resolution_loss, FQE_step_Q_list, resolution_list, save_folder_name,
                                        line_name_list, group_list):
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

        policy_result_path  = "Bvft_res_plot_"+str(resolution_list)
        Res_plot_folder = os.path.join(policy_result,policy_result_path)
        if not os.path.exists(Res_plot_folder):
            os.makedirs(Res_plot_folder)
        plot_name = "Figure_1R_" + save_folder_name  + ".png"
        save_path = os.path.join(Res_plot_folder, plot_name)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
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

    def remove_duplicates(self,lst):
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def draw_mse_graph(self,combinations, means, colors, standard_errors, labels, folder_path, FQE_step_list,
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
        ax.set_title(figure_name + "_" + str(FQE_step_list) + "_steps")

        ax.set_xticks(indices)
        ax.set_xticklabels(combinations)

        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, filename))
        plt.close()
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
    def generate_unique_colors(self,number_of_colors):

        cmap = plt.get_cmap('tab20')

        if number_of_colors <= 20:
            colors = [cmap(i) for i in range(number_of_colors)]
        else:
            colors = [cmap(i) for i in np.linspace(0, 1, number_of_colors)]
        return colors
    def get_data_size(self,episodes):
        size = 0
        for ele in episodes:
            size += len(ele.observations)
        return size

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

    def rank_elements(self,input_list):
        indexed_list = [(value, index) for index, value in enumerate(input_list)]

        sorted_list = sorted(indexed_list, key=lambda x: x[0], reverse=True)

        rank_list = [0] * len(input_list)

        current_rank = 1
        for i, (value, index) in enumerate(sorted_list):
            rank_list[index] = current_rank
            current_rank += 1

        return rank_list
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

    def calculate_top_k_precision(self,initial_state, env, policy_name_list, rank_list, k=2):
        # ranking_list 给了前k个policy的坐标

        device = self.device

        policy_performance_list = self.load_policy_performance(policy_name_list, env)
        policy_ranking_groundtruth = self.rank_elements(policy_performance_list)

        k_precision_list = []
        for i in range(1, k + 1):
            proportion = 0
            for pos in rank_list:
                if (rank_list[pos - 1] <= i - 1 and policy_ranking_groundtruth[pos - 1] <= i - 1):
                    proportion += 1
            proportion = proportion / i
            k_precision_list.append(proportion)
        return k_precision_list
    def calculate_top_k_normalized_regret(self,ranking_list, policy_name_list, env, k=2):
        print("calcualte top k normalized regret")
        policy_performance_list = self.load_policy_performance(policy_name_list, env)

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

    def extract_substrings(self,s):
        parts = s.split('_')

        if len(parts) < 4:
            return None, None

        return parts[1], parts[2]
    "return the ranking list"
    @abstractmethod
    def select_Q(self,q_functions,q_name_functions,policy_name_listi,q_sa,r_plus_vfsp):
        pass

    def get_self_ranking(self):
        Bvft_folder = "Bvft_Records"
        saving_folder = "Policy_ranking_saving_place"
        Q_saving_folder = self.data_name
        self.data_saving_path.append(Q_saving_folder)
        self.data_saving_path = self.remove_duplicates(self.data_saving_path)
        Q_saving_path = os.path.join(saving_folder, Q_saving_folder)
        if not os.path.exists(Q_saving_path):
            os.makedirs(Q_saving_path)
        policy_name_list, policy_list = self.load_policy(self.device)
        Q_FQE, Q_name_list, FQE_step_Q_list = self.load_FQE(policy_name_list, self.FQE_saving_step_list, self.replay_buffer,
                                                       self.device)  # 1d: how many policy #2d: how many step #3d: 4
        FQE_lr_list = [1e-4, 2e-5]
        FQE_hl_list = [[128, 256], [128, 1024]]
        data_size = self.data_size
        line_name_list = []
        for i in range(len(self.FQE_saving_step_list)):
            for j in range(len(FQE_lr_list)):
                for k in range(len(FQE_hl_list)):
                    line_name_list.append('FQE_' + str(FQE_lr_list[j]) + '_' + str(FQE_hl_list[k]) + '_' + str(
                        self.FQE_saving_step_list[i]) + "step")
        for i in range(len(Q_FQE)):
            save_folder_name = Q_name_list[i]
            Q_result_saving_path = os.path.join(Q_saving_path, save_folder_name)
            q_functions = []
            q_name_functions = []
            for j in range(len(Q_FQE[0])):
                for h in range(len(Q_FQE[0][0])):
                    q_functions.append(Q_FQE[i][j][h])
                    q_name_functions.append(FQE_step_Q_list[i][j][h])
            loss_function = []
            q_sa = [np.zeros(data_size) for _ in q_functions]
            r_plus_vfsp = [np.zeros(data_size) for _ in q_functions]
            ptr = 0
            gamma = self.gamma
            while ptr < self.trajectory_num:  # for everything in data size
                length = test_data.get_iter_length(ptr)
                state, action, next_state, reward, done = test_data.sample(ptr)
                for j in range(len(q_functions)):
                    actor = q_functions[j]
                    critic = q_functions[j]
                    q_sa[j][ptr:ptr + length] = critic.predict_value(state, action).flatten()[
                                                     :length]
                    vfsp = (reward.squeeze(-1) + critic.predict_value(next_state, actor.predict(next_state)) *(1- np.array(done)).squeeze(-1) * gamma)

                    r_plus_vfsp[j][ptr:ptr + length] = vfsp.flatten()[:length]
                    # print("self r plus vfsp : ",self.r_plus_vfsp[i][ptr:ptr + 20])
                ptr += 1
            result = self.select_Q(q_functions, q_name_functions,policy_name_list[i], q_sa, r_plus_vfsp)
            index = np.argmin(result)
            save_list = [q_name_functions[index]]
            self.save_as_txt(Q_result_saving_path, save_list)
            self.save_as_pkl(Q_result_saving_path, save_list)
            self.delete_files_in_folder(Bvft_folder)

    def SixR_get_FQE_name(self,policy_name,repo_name):
        ranking_folder = "Policy_ranking_saving_place"
        if not os.path.exists(ranking_folder):
            os.makedirs(ranking_folder)
        folder_path = os.path.join(ranking_folder,repo_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        policy_path = os.path.join(folder_path,policy_name)
        policy_path = policy_path + "_"+str(self.FQE_saving_step_list)
        return self.load_from_pkl(policy_path)[0]
    def get_NMSE(self,repo_name):
        print("Plot FQE MSE")
        whole_dataset = self.whole_dataset
        env = self.env
        train_episodes = self.train_episodes
        test_episodes = self.test_episodes
        replay_buffer = self.replay_buffer


        policy_returned_result_folder = "policy_returned_result"
        if not os.path.exists(policy_returned_result_folder):
            os.makedirs(policy_returned_result_folder)

        FQE_returned_folder = "FQE_returned_result"

        policy_total_model = 'policy_returned_total'
        policy_total_path = os.path.join(policy_returned_result_folder, policy_total_model)
        policy_total_dictionary = self.load_from_pkl(policy_total_path)

        true_list = []
        prediction_list = []
        max_step = max(FQE_saving_step_list)
        for policy_file_name in os.listdir("policy_trained"):
            policy_name = policy_file_name[:-3]

            FQE_model_name = self.SixR_get_FQE_name(policy_name,repo_name)

            FQE_learning_rate, FQE_hidden_layer = self.extract_substrings(FQE_model_name)

            FQE_directory = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer)
            FQE_folder = os.path.join(FQE_returned_folder, FQE_directory)
            if not os.path.exists(FQE_folder):
                os.makedirs(FQE_folder)

            FQE_total_result_folder = "FQE_returned_total"
            FQE_total_path = os.path.join(FQE_folder, FQE_total_result_folder)

            FQE_total_dictionary = self.load_from_pkl(FQE_total_path)

            true_list.append(policy_total_dictionary[policy_name])
            prediction_list.append(FQE_total_dictionary[FQE_model_name])
        NMSE, standard_error = self.normalized_mean_square_error_with_error_bar(true_list, prediction_list,
                                                                           self.normalization_factor)

        return NMSE, standard_error
    def draw_figure_6R(self):
        means = []
        SE = []
        labels = []
        self_data_saving_path = self.remove_duplicates(self.data_saving_path)
        max_step = str(max(self.FQE_saving_step_list))
        for i in range(len(self_data_saving_path)):
            repo_name = self_data_saving_path[i]
            NMSE,standard_error = self.get_NMSE(repo_name)
            means.append(NMSE)
            SE.append(standard_error)
            labels.append(self_data_saving_path[i]+"_"+max_step)
        name_list = ["hopper-medium-expert-v0"]

        FQE_returned_folder = "Policy_ranking_saving_place/Policy_k_saving_place/Figure_6R_plot"
        if not os.path.exists(FQE_returned_folder):
            os.makedirs(FQE_returned_folder)
        plot = "NMSE_plot"
        Figure_saving_path = os.path.join(FQE_returned_folder, plot)
        #
        colors = self.generate_unique_colors(len(self_data_saving_path))
        figure_name = 'Normalized MSE of FQE min max'
        filename = "Figure6R_max_min_NMSE_graph" + "_" + str(self.FQE_saving_step_list)
        if self.normalization_factor == 1:
            figure_name = 'Normalized MSE of FQE groundtruth variance'
            filename = "Figure6R_groundtruth_variance_NMSE_graph" + "_" + str(self.FQE_saving_step_list)
        self.draw_mse_graph(combinations=name_list, means=means, colors=colors, standard_errors=SE,
                       labels=labels, folder_path=Figure_saving_path, FQE_step_list=self.FQE_saving_step_list,
                       filename=filename, figure_name=figure_name)
    def get_min_loss(self,loss_list):  # input 2d list, return 1d list
        # print("loss list : ",loss_list)
        if (len(loss_list) == 1):
            return loss_list[0]
        min_loss = []
        # print("loss list : ",loss_list)
        for i in range(len(loss_list[0])):
            current_loss = []
            for j in range(len(loss_list)):
                current_loss.append(loss_list[j][i])
            min_loss.append(min(current_loss))
        return min_loss
    def load_FQE(self,policy_name_list, FQE_step_list, replay_buffer, device):
        policy_folder = 'policy_trained'
        FQE_lr_list = [1e-4, 2e-5]
        FQE_hl_list = [[128, 256], [128, 1024]]
        Q_FQE = []
        Q_name_list = []
        FQE_step_Q_list = []
        for policy_file_name in policy_name_list:
            policy_path = os.path.join(policy_folder, policy_file_name)
            policy = d3rlpy.load_learnable(policy_path + ".d3", device=device)
            Q_list = []
            inner_list = []
            Q_name_list.append(policy_file_name + "_" + str(FQE_step_list))
            for FQE_step in FQE_step_list:
                step_list = []
                FQE_policy_name = []
                for FQE_learning_rate in FQE_lr_list:
                    for FQE_hidden_layer in FQE_hl_list:

                        FQE_directory = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer)
                        if not os.path.exists(FQE_directory):
                            os.makedirs(FQE_directory)
                        FQE_model_pre = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer) + '_' + str(
                            FQE_step) + "step" + "_"
                        FQE_model_name = FQE_model_pre + policy_file_name
                        FQE_policy_name.append(FQE_model_name)

                        FQE_model_name = FQE_model_name + ".pt"
                        FQE_file_path = os.path.join(FQE_directory, FQE_model_name)
                        fqeconfig = d3rlpy.ope.FQEConfig(
                            learning_rate=FQE_learning_rate,
                            encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=FQE_hidden_layer)
                        )
                        fqe = FQE(algo=policy, config=fqeconfig, device=device)
                        fqe.build_with_dataset(replay_buffer)
                        fqe.load_model(FQE_file_path)
                        step_list.append(fqe)
                inner_list.append(FQE_policy_name)
                Q_list.append(step_list)
            Q_FQE.append(Q_list)
            FQE_step_Q_list.append(inner_list)
        return Q_FQE, Q_name_list, FQE_step_Q_list
    def run(self):
        print("object start run")
        num_runs = self.num_runs
        Result_saving_place = 'Policy_ranking_saving_place'
        Result_k = 'Policy_k_saving_place'
        Result_k_save_folder = os.path.join(Result_saving_place, Result_k)
        if not os.path.exists(Result_k_save_folder):
            os.makedirs(Result_k_save_folder)
        Result_k_save_path = os.path.join(Result_k_save_folder,"k_statistic")
        k_precision_name = str(k) + "_mean_precision_" + str(num_runs)
        k_regret_name = str(k) + "_mean_regret" + str(num_runs)
        precision_ci_name = str(k) + "_CI_precision" + str(num_runs)
        regret_ci_name = str(k) + "_CI_regret" + str(num_runs)
        plot_name = "plots"

        k_precision_mean_saving_path = os.path.join(Result_k_save_path, k_precision_name)
        k_regret_mean_saving_path = os.path.join(Result_k_save_path, k_regret_name)
        k_precision_ci_saving_path = os.path.join(Result_k_save_path, precision_ci_name)
        k_regret_ci_saving_path = os.path.join(Result_k_save_path, regret_ci_name)
        plot_name_saving_path = os.path.join(Result_k_save_path, plot_name)
        precision_path = os.path.join(Result_k_save_path, k_precision_name)
        # if os.path.exists(precision_path):
        #     print("load saved data")
        #     precision_mean_list = self.load_from_pkl(k_precision_mean_saving_path)
        #     regret_mean_list = self.load_from_pkl(k_regret_mean_saving_path)
        #     precision_ci_list = self.load_from_pkl(k_precision_ci_saving_path)
        #     regret_ci_list = self.load_from_pkl(k_regret_ci_saving_path)
        #     line_name_list = self.load_from_pkl(plot_name_saving_path)
        # else:
        precision_mean_list, regret_mean_list, precision_ci_list, regret_ci_list, line_name_list = self.calculate_k(self.data_saving_path,self.data_saving_path,self.FQE_saving_step_list,self.initial_state,self.k,self.num_runs)
        plot_mean_list = [precision_mean_list, regret_mean_list]
        plot_ci_list = [precision_ci_list, regret_ci_list]

        plot_folder = os.path.join(Result_k_save_path, "Figure_6_L_plot")
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        y_axis_names = ["k precision", "k regret"]
        colors = self.generate_unique_colors(len(plot_mean_list[0]))
        line_name = ["hopper-medium-expert-v0", "hopper-medium-expert-v0"]
        print("line name list : ", line_name_list)
        # print("plot mean list : ",plot_mean_list)
        # print("ci lsit : ",plot_ci_list)
        self.plot_subplots(data=plot_mean_list, save_path=plot_folder, y_axis_names=y_axis_names,
                      line_names=line_name_list, colors=colors, ci=plot_ci_list)
        print("plot finished")
    def load_policy(self,device):
        policy_folder = 'policy_trained'
        if not os.path.exists(policy_folder):
            os.makedirs(policy_folder)

        policy_name_list = []
        policy_list = []
        for policy_file_name in os.listdir(policy_folder):  # policy we want to evaluate
            policy_path = os.path.join(policy_folder, policy_file_name)
            policy = d3rlpy.load_learnable(policy_path, device=device)

            policy_name_list.append(policy_file_name[:-3])
            policy_list.append(policy)
        return policy_name_list, policy_list

    def load_policy_performance(self, policy_name_list, env):
        policy_folder = 'policy_trained'

        performance_folder = "policy_returned_result"
        total_name = "policy_returned_total.txt"
        performance_total_path = os.path.join(performance_folder, total_name)
        performance_dict = self.load_dict_from_txt(performance_total_path)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        performance_list = []
        for policy_name in policy_name_list:
            included = False
            print("policy name : ", policy_name)
            print("performance dict : ", performance_dict)
            if policy_name in performance_dict:
                performance_list.append(float(performance_dict[policy_name]))
                print("included")
                included = True
            if not included:
                print("not included")
                policy_path = os.path.join(policy_folder, policy_name)
                policy = d3rlpy.load_learnable(policy_path + ".d3", device=device)
                performance_list.append(calculate_policy_value(env, policy, gamma=0.99, num_run=100))
        return performance_list
    def load_FQE_performance_specific(self,FQE_learning_rate, FQE_hidden_layer, FQE_step, policy_name):
        FQE_returned_result = "FQE_returned_result"
        FQE_folder = "FQE_" + str(FQE_learning_rate) + "_" + str(FQE_hidden_layer)
        FQE_name = FQE_folder + "_" + str(FQE_step) + "step" + "_" + policy_name
        FQE_folder_path = os.path.join(FQE_returned_result, FQE_folder)
        if not os.path.exists(FQE_folder_path):
            os.makedirs(FQE_folder_path)
        FQE_total = "FQE_returned_total"
        FQE_path = os.path.join(FQE_folder_path, FQE_total)
        FQE_dictionary = load_from_pkl(FQE_path)
        FQE_result = FQE_dictionary[FQE_name]
        return FQE_result

    def pick_policy(self,m, device):
        policy_folder = 'policy_trained'
        if not os.path.exists(policy_folder):
            os.makedirs(policy_folder)
        policy_files = sample_files(policy_folder, m)
        policy_name_list = []
        policy_list = []
        for policy_file_name in policy_files:  # policy we want to evaluate
            policy_path = os.path.join(policy_folder, policy_file_name)
            policy = d3rlpy.load_learnable(policy_path, device=device)

            policy_name_list.append(policy_file_name[:-3])
            policy_list.append(policy)
        return policy_name_list, policy_list
    def load_FQE_performance(self,FQE_name):
        FQE_returned_result = "FQE_returned_result"
        FQE_folder = "FQE_0.0001_[128, 1024]"
        if (FQE_name[:20] == "FQE_2e-05_[128, 256]"):
            FQE_folder = "FQE_2e-05_[128, 256]"
        elif (FQE_name[:21] == "FQE_0.0001_[128, 256]"):
            FQE_folder = "FQE_0.0001_[128, 256]"
        elif (FQE_name[:21] == "FQE_2e-05_[128, 1024]"):
            FQE_folder = "FQE_2e-05_[128, 1024]"
        FQE_folder_path = os.path.join(FQE_returned_result, FQE_folder)
        if not os.path.exists(FQE_folder_path):
            os.makedirs(FQE_folder_path)
        FQE_total = "FQE_returned_total"
        FQE_path = os.path.join(FQE_folder_path, FQE_total)
        FQE_dictionary = self.load_from_pkl(FQE_path)
        FQE_result = FQE_dictionary[FQE_name]
        return FQE_result
    def get_ranking(self,data_address,policy_name_list,FQE_saving_step_list):
        env = self.env
        Policy_ranking_saving_place = "Policy_ranking_saving_place"
        ranking_path = os.path.join(Policy_ranking_saving_place,data_address)
        if not os.path.exists(ranking_path):
            os.makedirs(ranking_path)
        FQE_performance_list = []
        for i in range(len(policy_name_list)):
            policy_file_name = policy_name_list[i]
            folder_name = policy_file_name + "_" + str(FQE_saving_step_list)
            FQE_name_path = os.path.join(ranking_path, folder_name)
            FQE_name = self.load_from_pkl(FQE_name_path)[0]
            FQE_performance = self.load_FQE_performance(FQE_name)
            FQE_performance_list.append(FQE_performance)
        FQE_rank_list = self.rank_elements_larger_higher(FQE_performance_list)
        return FQE_rank_list
    def calculate_k(self,data_address_lists,plot_name_list,FQE_saving_step_list,initial_state,k,num_runs):
        """

        :param data_address: 想要计算的方法的文件夹名字(list)，放在Bvft_saving_place下面
        :return: precision, regret 的平均值和 confidence interval,还有要画图的吗名称
        """
        FQE_name_list_new = []
        # whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
        env = self.env
        device = self.device
        Ranking_list = []
        Policy_name_list = []
        self.data_saving_path = self.remove_duplicates(self.data_saving_path)
        data_address_lists = self.remove_duplicates(self.data_saving_path)
        print("data address lists : ",data_address_lists)
        for i in range(len(data_address_lists)):
            Ranking_list.append([])
        for runs in range(num_runs):
            policy_name_list, policy_list = self.pick_policy(15,device)
            for data_address_index in range(len(data_address_lists)):
                # print("ranking : ",self.get_ranking(data_address_lists[data_address_index],policy_name_list,FQE_saving_step_list))
                # sys.exit()
                Ranking_list[data_address_index].append(self.get_ranking(data_address_lists[data_address_index],policy_name_list,FQE_saving_step_list))   #多少个 不同的种类 #多少run #多少个policy ranking
            # print(Ranking_list)
            # performance_list, FQE_name_list = FQE_ranking(policy_name_list,FQE_saving_step_list,env)
            # print(performance_list)
            # print(FQE_name_list)
            Policy_name_list.append(policy_name_list)
        # print("len 0 : ",len(Ranking_list))
        # print("len 1 : ",len(Ranking_list[0]))
        # print("len 2 : ",len(Ranking_list[0][0]))
        #
        # print("ranking list : ",Ranking_list)
        # sys.exit()
        Precision_list = []
        Regret_list = []
        for index in range(len(data_address_lists)):
            Precision_list.append([])
            Regret_list.append([])
        for i in range(num_runs):
            for num_index in range(len(Ranking_list)):
                Precision_list[num_index].append(self.calculate_top_k_precision(initial_state,env,Policy_name_list[i],Ranking_list[num_index][i],k))
                Regret_list[num_index].append(self.calculate_top_k_normalized_regret(Ranking_list[num_index][i],Policy_name_list[i],env,k))
        print("precision list : ",Precision_list)
        print("regret list  :",Regret_list)


        Precision_k_list = []
        Regret_k_list = []
        for iu in range(len(Ranking_list)):
            Precision_k_list.append([])
            Regret_k_list.append([])

        for i in range(k):
            for ku in range(len(Ranking_list)):
                k_precision = []
                k_regret = []
                for j in range(num_runs):
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
        policy_ranking_saving_place = 'Policy_ranking_saving_place'
        k_saving_folder = 'Policy_k_saving_place'
        k_saving_path = os.path.join(policy_ranking_saving_place,k_saving_folder)
        # Bvft_k_save_path = os.path.join(Bvft_saving_place, Bvft_k)
        # if not os.path.exists(Bvft_k_save_path):
        #     os.makedirs(Bvft_k_save_path)


        saving_path = os.path.join(k_saving_path,"k_statistic")
        plot_name = "plots"
        k_precision_name = str(k)+"_mean_precision_"+str(num_runs)
        k_regret_name = str(k)+"_mean_regret"+str(num_runs)
        precision_ci_name = str(k)+"_CI_precision"+str(num_runs)
        regret_ci_name = str(k)+"_CI_regret"+str(num_runs)


        k_precision_mean_saving_path = os.path.join(saving_path,k_precision_name)
        k_regret_mean_saving_path = os.path.join(saving_path,k_regret_name)
        k_precision_ci_saving_path = os.path.join(saving_path,precision_ci_name)
        k_regret_ci_saving_path = os.path.join(saving_path,regret_ci_name)
        plot_name_saving_path = os.path.join(saving_path,plot_name)

        self.save_as_pkl(k_precision_mean_saving_path,precision_mean_list)
        self.save_as_pkl(k_regret_mean_saving_path,regret_mean_list)
        self.save_as_pkl(k_precision_ci_saving_path,precision_ci_list)
        self.save_as_pkl(k_regret_ci_saving_path,regret_ci_list)
        self.save_as_pkl(plot_name_saving_path,plot_name_list)

        self.save_as_txt(k_precision_mean_saving_path,precision_mean_list)
        self.save_as_txt(k_regret_mean_saving_path,regret_mean_list)
        self.save_as_txt(k_precision_ci_saving_path,precision_ci_list)
        self.save_as_txt(k_regret_ci_saving_path,regret_ci_list)
        self.save_as_txt(plot_name_saving_path,plot_name_list)


        return precision_mean_list,regret_mean_list,precision_ci_list,regret_ci_list,plot_name_list
class Bvft_poli(policy_select):
    def select_Q(self, q_functions, q_name_functions, policy_name_listi, q_sa, r_plus_vfsp):
        # resolution_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        resolution_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14, 15, 16,17, 18, 19, 20, 21,22,23]
        rmax, rmin = self.env.reward_range[0], self.env.reward_range[1]
        result_list = []
        Bvft_final_resolution_loss = []
        for i in range(len(FQE_saving_step_list) * 4):
            current_list = []
            Bvft_final_resolution_loss.append(current_list)
        group_list = []
        for resolution in resolution_list:
            record = BvftRecord()
            bvft_instance = BVFT(q_functions, self.test_data, gamma, rmax, rmin, policy_name_listi, record,
                                 "torch_actor_critic_cont", verbose=True, data_size=self.data_size,
                                 trajectory_num=self.trajectory_num)
            bvft_instance.run(resolution=resolution)
            group_list.append(record.group_counts[0])
            for i in range(len(record.losses[0])):
                Bvft_final_resolution_loss[i].append(record.losses[0][i])

            result_list.append(record.losses[0])
        min_loss_list = self.get_min_loss(result_list)
        less_index_list = self.rank_elements_lower_higher(min_loss_list)

        FQE_lr_list = [1e-4, 2e-5]
        FQE_hl_list = [[128, 256], [128, 1024]]
        line_name_list = []
        for i in range(len(self.FQE_saving_step_list)):
            for j in range(len(FQE_lr_list)):
                for k in range(len(FQE_hl_list)):
                    line_name_list.append('FQE_' + str(FQE_lr_list[j]) + '_' + str(FQE_hl_list[k]) + '_' + str(
                        self.FQE_saving_step_list[i]) + "step")
        Policy_ranking_saving_place = "Policy_ranking_saving_place"
        Policy_k_saving_place = "Policy_k_saving_place"
        Res_plot_folder = "Bvft_res_data_"+str(resolution_list)

        Policy_k_saving_path = os.path.join(Policy_ranking_saving_place,Policy_k_saving_place)
        Res_plot_saving_folder = os.path.join(Policy_k_saving_path,Res_plot_folder)
        save_folder_name = policy_name_listi
        if not os.path.exists(Res_plot_saving_folder):
            os.makedirs(Res_plot_saving_folder)
        policy_saving_path = os.path.join(Res_plot_saving_folder,policy_name_listi)
        if not os.path.exists(policy_saving_path):
            os.makedirs(policy_saving_path)
        self.save_as_pkl(policy_saving_path,Bvft_final_resolution_loss)
        self.save_as_txt(policy_saving_path,Bvft_final_resolution_loss)
        self.draw_Bvft_resolution_loss_graph(Bvft_final_resolution_loss, self.FQE_saving_step_list, resolution_list,
                                        save_folder_name, line_name_list, group_list)
        return less_index_list
class Bvft_zero(policy_select):
    def select_Q(self,q_functions,q_name_functions,policy_name_listi,q_sa,r_plus_vfsp):
        resolution_list = np.array([0.00001])
        rmax, rmin = self.env.reward_range[0], self.env.reward_range[1]
        result_list = []
        for resolution in resolution_list:
            record = BvftRecord()
            bvft_instance = BVFT(q_functions, self.test_data, gamma, rmax, rmin, policy_name_listi, record,
                                 "torch_actor_critic_cont", verbose=True, data_size=self.data_size,
                                 trajectory_num=self.trajectory_num)
            bvft_instance.run(resolution=resolution)
            result_list.append(record.losses[0])
        list_to_rank = result_list[0]
        less_index_list = self.rank_elements_lower_higher(list_to_rank)
        return less_index_list


class FQE_zero(policy_select):
    def select_Q(self,q_functions,q_name_functions,policy_name_listi,q_sa,r_plus_vfsp):
        loss = []
        for i in range(len(q_functions)):
            if i % 4 == 0:
                loss.append(self.load_FQE_performance(q_name_functions[i]))
            else:
                loss.append(-9999)
        result = self.rank_elements_larger_higher(loss)
        return result

class FQE_one(policy_select):
    def select_Q(self, q_functions, q_name_functions, policy_name_listi, q_sa, r_plus_vfsp):
        loss = []
        for i in range(len(q_functions)):
            if i % 4 == 1:
                loss.append(self.load_FQE_performance(q_name_functions[i]))
            else:
                loss.append(-9999)
        result = self.rank_elements_larger_higher(loss)
        return result

class FQE_two(policy_select):
    def select_Q(self, q_functions, q_name_functions, policy_name_listi, q_sa, r_plus_vfsp):
        loss = []
        for i in range(len(q_functions)):
            if i % 4 == 2:
                loss.append(self.load_FQE_performance(q_name_functions[i]))
            else:
                loss.append(-9999)
        result = self.rank_elements_larger_higher(loss)
        return result

class FQE_three(policy_select):
    def select_Q(self, q_functions, q_name_functions, policy_name_listi, q_sa, r_plus_vfsp):
        loss = []
        for i in range(len(q_functions)):
            if i % 4 == 3:
                loss.append(self.load_FQE_performance(q_name_functions[i]))
            else:
                loss.append(-9999)
        result = self.rank_elements_larger_higher(loss)
        return result
class Bvft_abs(policy_select):
    def select_Q(self, q_functions, q_name_functions, policy_name_listi, q_sa, r_plus_vfsp):
        loss_function = []
        for i in range(len(q_functions)):
            diff = q_sa[i]-r_plus_vfsp[i]
            loss_function.append(np.abs(np.sum(diff)/len(diff)))
        less_index_list = self.rank_elements_lower_higher(loss_function)
        return less_index_list
class arg_i_max_j(policy_select):
    def select_Q(self, q_functions, q_name_functions, policy_name_listi, q_sa, r_plus_vfsp):
        max_j_list = []
        length = len(q_sa[0])
        for i in range(len(q_functions)):
            loss_list = []
            for j in range(len(q_functions)):
                inner_value = 0
                loss = np.abs(np.sum(q_sa[j]*(q_sa[i]-r_plus_vfsp[i]),axis=1))
                print("loss : ",loss)
                loss_list.append(loss)
            max_j_list.append(max(loss_list))
        return self.rank_elements_lower_higher(max_j_list)
class CustomDataLoader:
    def __init__(self, dataset, batch_size=1024):
        self.dataset = dataset
        self.batch_size = batch_size
        self.current = 0
    def __iter__(self):
        self.current = 0
        np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current >= len(self.dataset):
            raise StopIteration
        return self.sample(self.batch_size)
    def __len__(self):
        return len(self.dataset)

    def get_iter_length(self,iteration_number):
        return len(self.dataset.episodes[iteration_number].observations)
    def get_state_shape(self):
        first_state = self.dataset.observations[0]
        return np.array(first_state).shape
    def sample(self, iteration_number):
        dones = []
        states = self.dataset.episodes[iteration_number].observations
        actions =  self.dataset.episodes[iteration_number].actions
        padded_next_states =  self.dataset.episodes[iteration_number].observations[1:len(self.dataset.episodes[iteration_number].observations)]
        padded_next_states = np.append(padded_next_states, [self.dataset.episodes[iteration_number].observations[-1]], axis=0)
        rewards = self.dataset.episodes[iteration_number].rewards
        done = self.dataset.episodes[iteration_number].terminated
        for i in range(len(states)):
            if(i == len(states)-1):
                if done:
                    dones.append([1])
                else:
                    dones.append([0])
            else:
                dones.append([0])
        # print(dones)
        # print(states)
        # sys.exit()
        return states, actions, padded_next_states, rewards, dones
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# whole_dataset, env = get_d4rl('hopper-medium-v2')
#
# train_episodes = whole_dataset.episodes[0:1500]
# test_episodes = whole_dataset.episodes[1500:2186]
whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
print("len : ",len(whole_dataset.episodes))
train_episodes = whole_dataset.episodes[0:2000]
test_episodes = whole_dataset.episodes[2000:2277]
buffer_one = FIFOBuffer(limit=1000000)
replay_buffer_test = ReplayBuffer(buffer=buffer_one, episodes=test_episodes)
Bvft_batch_dim = 1000
test_data = CustomDataLoader(replay_buffer_test, batch_size=Bvft_batch_dim)
k = 5
num_runs = 300
FQE_saving_step_list = [2000000]
initial_state = 12345
# data_saving_path = ["Bvft_ranking","Bvft_res_0","Bvft_abs"]
data_saving_path = ["Bvft_ranking","Bvft_res_0","FQE_0.0001_256","FQE_0.0001_1024","FQE_0.00002_256","FQE_0.00002_1024","l1_norm","arg_i_max_j"]
normalization_factor = 0
# data_saving_path = ["Bvft_ranking"]
gamma = 0.99
bvft_poli = Bvft_poli(device=device,data_list =data_saving_path,data_name_self = "Bvft_ranking",whole_dataset= whole_dataset,train_episodes=train_episodes,
                     test_episodes=test_episodes,test_data=test_data,replay_buffer=replay_buffer_test,env=env,k=k,
                     num_runs=num_runs,FQE_saving_step_list=FQE_saving_step_list,
                 gamma=gamma,initial_state=initial_state,normalization_factor=normalization_factor)
bvft_obj = Bvft_zero(device=device,data_list =data_saving_path,data_name_self = "Bvft_res_0",whole_dataset= whole_dataset,train_episodes=train_episodes,
                     test_episodes=test_episodes,test_data=test_data,replay_buffer=replay_buffer_test,env=env,k=k,
                     num_runs=num_runs,FQE_saving_step_list=FQE_saving_step_list,
                 gamma=gamma,initial_state=initial_state,normalization_factor=normalization_factor)
FQE_zero = FQE_zero(device=device,data_list =data_saving_path,data_name_self = "FQE_0.0001_256",whole_dataset= whole_dataset,train_episodes=train_episodes,
                     test_episodes=test_episodes,test_data=test_data,replay_buffer=replay_buffer_test,env=env,k=k,
                     num_runs=num_runs,FQE_saving_step_list=FQE_saving_step_list,
                 gamma=gamma,initial_state=initial_state,normalization_factor=normalization_factor)
FQE_one = FQE_one(device=device,data_list =data_saving_path,data_name_self = "FQE_0.0001_1024",whole_dataset= whole_dataset,train_episodes=train_episodes,
                     test_episodes=test_episodes,test_data=test_data,replay_buffer=replay_buffer_test,env=env,k=k,
                     num_runs=num_runs,FQE_saving_step_list=FQE_saving_step_list,
                 gamma=gamma,initial_state=initial_state,normalization_factor=normalization_factor)
FQE_two = FQE_two(device=device,data_list =data_saving_path,data_name_self = "FQE_0.00002_256",whole_dataset= whole_dataset,train_episodes=train_episodes,
                     test_episodes=test_episodes,test_data=test_data,replay_buffer=replay_buffer_test,env=env,k=k,
                     num_runs=num_runs,FQE_saving_step_list=FQE_saving_step_list,
                 gamma=gamma,initial_state=initial_state,normalization_factor=normalization_factor)
FQE_three = FQE_three(device=device,data_list =data_saving_path,data_name_self = "FQE_0.00002_1024",whole_dataset= whole_dataset,train_episodes=train_episodes,
                     test_episodes=test_episodes,test_data=test_data,replay_buffer=replay_buffer_test,env=env,k=k,
                     num_runs=num_runs,FQE_saving_step_list=FQE_saving_step_list,
                 gamma=gamma,initial_state=initial_state,normalization_factor=normalization_factor)
l1_norm = Bvft_abs(device=device,data_list =data_saving_path,data_name_self = "l1_norm",whole_dataset= whole_dataset,train_episodes=train_episodes,
                     test_episodes=test_episodes,test_data=test_data,replay_buffer=replay_buffer_test,env=env,k=k,
                     num_runs=num_runs,FQE_saving_step_list=FQE_saving_step_list,
                 gamma=gamma,initial_state=initial_state,normalization_factor=normalization_factor)
arg_i_max_j = arg_i_max_j(device=device,data_list =data_saving_path,data_name_self = "arg_i_max_j",whole_dataset= whole_dataset,train_episodes=train_episodes,
                     test_episodes=test_episodes,test_data=test_data,replay_buffer=replay_buffer_test,env=env,k=k,
                     num_runs=num_runs,FQE_saving_step_list=FQE_saving_step_list,
                 gamma=gamma,initial_state=initial_state,normalization_factor=normalization_factor)
arg_i_max_j.get_self_ranking()
bvft_poli.get_self_ranking()
bvft_obj.get_self_ranking()
FQE_zero.get_self_ranking()
FQE_one.get_self_ranking()
FQE_two.get_self_ranking()
FQE_three.get_self_ranking()
l1_norm.get_self_ranking()
l1_norm.draw_figure_6R()
l1_norm.run()
# bvft_obj.draw_figure_6R()

