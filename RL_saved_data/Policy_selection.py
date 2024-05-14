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
from top_k_cal import *
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
    def __init__(self,device,data_list, whole_dataset,env,k,num_runs,FQE_saving_step_list,initial_state):
        self.device = device
        self.env = env
        self.data_saving_path = data_list
        self.k = k
        self.num_runs = num_runs
        self.FQE_saving_step_list = FQE_saving_step_list
        self.whole_dataset = whole_dataset

    @abstractmethod
    def select_Q(self):
        pass

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
        Result_saving_place = 'Policy_ranking_saving_place'
        Result_k = 'Policy_k_saving_place'
        Result_k_save_path = os.path.join(Result_saving_place, Result_k)
        if not os.path.exists(Result_k_save_path):
            os.makedirs(Result_k_save_path)
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
        if os.path.exists(precision_path):
            print("load saved data")
            precision_mean_list = load_from_pkl(k_precision_mean_saving_path)
            regret_mean_list = load_from_pkl(k_regret_mean_saving_path)
            precision_ci_list = load_from_pkl(k_precision_ci_saving_path)
            regret_ci_list = load_from_pkl(k_regret_ci_saving_path)
            line_name_list = load_from_pkl(plot_name_saving_path)
        else:
            precision_mean_list, regret_mean_list, precision_ci_list, regret_ci_list, line_name_list = calculate_k(self, self.plot_name_list, self.plot_name_list, FQE_saving_step_list, self.initial_state,
                        self.k, self.num_runs)

        plot_mean_list = [precision_mean_list, regret_mean_list]
        plot_ci_list = [precision_ci_list, regret_ci_list]

        plot_folder = os.path.join(Result_k_save_path, "Figure_6_L_plot")
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        y_axis_names = ["k precision", "k regret"]
        colors = generate_unique_colors(len(plot_mean_list[0]))
        line_name = ["hopper-medium-expert-v0", "hopper-medium-expert-v0"]
        print("line name list : ", line_name_list)
        # print("plot mean list : ",plot_mean_list)
        # print("ci lsit : ",plot_ci_list)
        plot_subplots(data=plot_mean_list, save_path=plot_folder, y_axis_names=y_axis_names,
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
        performance_dict = load_dict_from_txt(performance_total_path)
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
        FQE_dictionary = load_from_pkl(FQE_path)
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
            FQE_name = load_from_pkl(FQE_name_path)[0]
            FQE_performance = load_FQE_performance(FQE_name)
            FQE_performance_list.append(FQE_performance)
        FQE_rank_list = rank_elements_larger_higher(FQE_performance_list)
        return FQE_rank_list
    def calculate_k(self,data_address_lists,plot_name_list,FQE_saving_step_list,initial_state,k,num_runs):
        """

        :param data_address: 想要计算的方法的文件夹名字(list)，放在Bvft_saving_place下面
        :return: precision, regret 的平均值和 confidence interval,还有要画图的吗名称
        """
        FQE_name_list_new = []
        # whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
        env = self.env
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        Ranking_list = []
        Policy_name_list = []
        for i in range(len(data_address_lists)):
            Ranking_list.append([])
        for runs in range(num_runs):
            policy_name_list, policy_list = pick_policy(15,device)
            for data_address_index in range(len(data_address_lists)):
                Ranking_list[data_address_index].append(get_ranking(data_address_lists[data_address_index],policy_name_list,FQE_saving_step_list))
            Policy_name_list.append(policy_name_list)
        Precision_list = []
        Regret_list = []
        for index in range(len(data_address_lists)):
            Precision_list.append([])
            Regret_list.append([])
        for i in range(num_runs):
            for num_index in range(len(data_address_lists)):
                Precision_list[num_index].append(calculate_top_k_precision(initial_state,env,Policy_name_list[num_index],Ranking_list[num_index][i]))
                Regret_list[num_index].append(calculate_top_k_normalized_regret(Ranking_list[num_index][i],Policy_name_list[i],env,k))

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
                current_precision_mean, current_precision_ci = calculate_statistics(Precision_list[i][j])
                current_regret_mean, current_regret_ci = calculate_statistics(Regret_list[i][j])
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
        Bvft_k_save_path = os.path.join(Bvft_saving_place, Bvft_k)
        if not os.path.exists(Bvft_k_save_path):
            os.makedirs(Bvft_k_save_path)

        for ind in range(len(Ranking_list)):
            saving_path = os.path.join(Bvft_k_save_path,data_address_lists[ind])
            plot_name = "plots"
            if not os.path.exists(saving_path):
                os.makedirs(saving_path)
            k_precision_name = str(k)+"_mean_precision_"+str(num_runs)
            k_regret_name = str(k)+"_mean_regret"+str(num_runs)
            precision_ci_name = str(k)+"_CI_precision"+str(num_runs)
            regret_ci_name = str(k)+"_CI_regret"+str(num_runs)


            k_precision_mean_saving_path = os.path.join(saving_path,k_precision_name)
            k_regret_mean_saving_path = os.path.join(saving_path,k_regret_name)
            k_precision_ci_saving_path = os.path.join(saving_path,precision_ci_name)
            k_regret_ci_saving_path = os.path.join(saving_path,regret_ci_name)
            plot_name_saving_path = os.path.join(saving_path,plot_name)

            save_as_pkl(k_precision_mean_saving_path,precision_mean_list)
            save_as_pkl(k_regret_mean_saving_path,regret_mean_list)
            save_as_pkl(k_precision_ci_saving_path,precision_ci_list)
            save_as_pkl(k_regret_ci_saving_path,regret_ci_list)
            save_as_pkl(plot_name_saving_path,plot_name_list)

            save_as_txt(k_precision_mean_saving_path,precision_mean_list)
            save_as_txt(k_regret_mean_saving_path,regret_mean_list)
            save_as_txt(k_precision_ci_saving_path,precision_ci_list)
            save_as_txt(k_regret_ci_saving_path,regret_ci_list)
            save_as_txt(plot_name_saving_path,plot_name_list)


        return precision_mean_list,regret_mean_list,precision_ci_list,regret_ci_list,plot_name_list
class Bvft_poli(policy_select):
    def select_Q(self):
        device = self.device
        print("begin save best Q, current device : ", device)
        whole_dataset = self.whole_dataset
        env = self.env
        train_episodes = whole_dataset.episodes[0:2000]
        test_episodes = whole_dataset.episodes[2000:2276]
        Bvft_batch_dim = get_mean_length(test_episodes)
        trajectory_num = len(test_episodes)
        print("mean length : ", Bvft_batch_dim)
        buffer_one = FIFOBuffer(limit=500000)
        replay_buffer_test = ReplayBuffer(buffer=buffer_one, episodes=test_episodes)
        buffer = FIFOBuffer(limit=500000)
        replay_buffer = ReplayBuffer(buffer=buffer, episodes=train_episodes)

        gamma = 0.99
        rmax, rmin = env.reward_range[0], env.reward_range[1]
        data_size = get_data_size(test_episodes)
        print("data size : ", get_data_size(whole_dataset.episodes))
        test_data = CustomDataLoader(replay_buffer_test, batch_size=Bvft_batch_dim)

        Bvft_saving_folder = "Policy_ranking_saving_place"
        Bvft_Q_saving_folder = "Bvft_ranking"
        self.data_saving_path.append(Bvft_Q_saving_folder)
        Bvft_Q_saving_path = os.path.join(Bvft_saving_folder, Bvft_Q_saving_folder)
        if not os.path.exists(Bvft_Q_saving_path):
            os.makedirs(Bvft_Q_saving_path)
        # Bvft_resolution_losses_saving_folder = "Bvft_resolution_loss_saving_place"
        # Bvft_resolution_losses_saving_path = os.path.join(Bvft_saving_folder, Bvft_resolution_losses_saving_folder)
        # if not os.path.exists(Bvft_resolution_losses_saving_path):
        #     os.makedirs(Bvft_resolution_losses_saving_path)
        # if not os.path.exists(Bvft_Q_saving_path):
        #     os.makedirs(Bvft_Q_saving_path)
        policy_name_list, policy_list = self.load_policy(device)

        Q_FQE, Q_name_list, FQE_step_Q_list = self.load_FQE(policy_name_list, self.FQE_saving_step_list, replay_buffer,
                                                       device)  # 1d: how many policy #2d: how many step #3d: 4
        FQE_lr_list = [1e-4, 2e-5]
        FQE_hl_list = [[128, 256], [128, 1024]]
        resolution_list = np.array([0.1, 0.2, 0.5, 0.7, 1.0]) * 100
        # print("input resolution list for Bvft : ", resolution_list)
        Bvft_folder = "Bvft_Records"
        if not os.path.exists(Bvft_folder):
            os.makedirs(Bvft_folder)

        line_name_list = []
        for i in range(len(FQE_saving_step_list)):
            for j in range(len(FQE_lr_list)):
                for k in range(len(FQE_hl_list)):
                    line_name_list.append('FQE_' + str(FQE_lr_list[j]) + '_' + str(FQE_hl_list[k]) + '_' + str(
                        FQE_saving_step_list[i]) + "step")
        for i in range(len(Q_FQE)):
            save_folder_name = Q_name_list[i]
            # Bvft_resolution_loss_policy_saving_path = os.path.join(Bvft_resolution_losses_saving_path, save_folder_name)
            Bvft_Q_result_saving_path = os.path.join(Bvft_Q_saving_path, save_folder_name)

            q_functions = []
            q_name_functions = []
            for j in range(len(Q_FQE[0])):
                for h in range(len(Q_FQE[0][0])):
                    q_functions.append(Q_FQE[i][j][h])
                    q_name_functions.append(FQE_step_Q_list[i][j][h])
            Bvft_losses = []
            # Bvft_final_resolution_loss = []
            # for i in range(len(FQE_saving_step_list) * 4):
            #     current_list = []
            #     Bvft_final_resolution_loss.append(current_list)
            group_list = []
            for resolution in resolution_list:
                record = BvftRecord()
                bvft_instance = BVFT(q_functions, test_data, gamma, rmax, rmin, policy_name_list[i], record,
                                     "torch_actor_critic_cont", verbose=True, data_size=data_size,
                                     trajectory_num=trajectory_num)
                # print("resolution : ",resolution)
                bvft_instance.run(resolution=resolution)

                group_list.append(record.group_counts[0])
                # for i in range(len(record.losses[0])):
                #     Bvft_final_resolution_loss[i].append(record.losses[0][i])

                Bvft_losses.append(record.losses[0])
            # print('Bvft losses : ',Bvft_losses)
            min_loss_list = self.get_min_loss(Bvft_losses)
            # print("min loss list : ",min_loss_list)
            ranking_list = rank_elements_lower_higher(min_loss_list)
            # print(" ranking list : ",ranking_list)

            best_ranking_index = np.argmin(ranking_list)
            # print("best ranking index: ",best_ranking_index)
            # sys.exit()
            save_list = [q_name_functions[best_ranking_index]]
            # save_as_pkl(Bvft_resolution_loss_policy_saving_path, Bvft_final_resolution_loss)
            # save_as_txt(Bvft_resolution_loss_policy_saving_path, Bvft_final_resolution_loss)
            save_as_txt(Bvft_Q_result_saving_path, save_list)
            save_as_pkl(Bvft_Q_result_saving_path, save_list)
            delete_files_in_folder(Bvft_folder)
            # draw_Bvft_resolution_loss_graph(Bvft_final_resolution_loss, FQE_saving_step_list, resolution_list,
            #                                 save_folder_name, line_name_list, group_list)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
k = 4
num_runs = 10
FQE_saving_step_list = [2000000]
initial_state = 12345
data_saving_path = ["Bvft_ranking"]
bvft_obj = Bvft_poli(device, data_saving_path, whole_dataset,env,k,num_runs,FQE_saving_step_list,initial_state)

bvft_obj.select_Q()
bvft_obj.calculate_k(data_saving_path,self.data_saving_path,self.FQE_saving_step_list,self.initial_state,self.k,self.num_runs)
