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
        self.initial_state = initial_state

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
        print("object start run")
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
        if os.path.exists(precision_path):
            print("load saved data")
            precision_mean_list = load_from_pkl(k_precision_mean_saving_path)
            regret_mean_list = load_from_pkl(k_regret_mean_saving_path)
            precision_ci_list = load_from_pkl(k_precision_ci_saving_path)
            regret_ci_list = load_from_pkl(k_regret_ci_saving_path)
            line_name_list = load_from_pkl(plot_name_saving_path)
        else:
            precision_mean_list, regret_mean_list, precision_ci_list, regret_ci_list, line_name_list = self.calculate_k(self.data_saving_path,self.data_saving_path,self.FQE_saving_step_list,self.initial_state,self.k,self.num_runs)

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
        device = self.device
        Ranking_list = []
        Policy_name_list = []
        data_address_lists = remove_duplicates(data_address_lists)
        for i in range(len(data_address_lists)):
            Ranking_list.append([])
        for runs in range(num_runs):
            policy_name_list, policy_list = self.pick_policy(15,device)
            for data_address_index in range(len(data_address_lists)):
                Ranking_list[data_address_index].append(self.get_ranking(data_address_lists[data_address_index],policy_name_list,FQE_saving_step_list))
            Policy_name_list.append(policy_name_list)
        Precision_list = []
        Regret_list = []
        for index in range(len(data_address_lists)):
            Precision_list.append([])
            Regret_list.append([])
        for i in range(num_runs):
            for num_index in range(len(data_address_lists)):
                Precision_list[num_index].append(calculate_top_k_precision(initial_state,env,Policy_name_list[num_index],Ranking_list[num_index][i],k))
                Regret_list[num_index].append(calculate_top_k_normalized_regret(Ranking_list[num_index][i],Policy_name_list[i],env,k))

        Precision_k_list = []
        Regret_k_list = []
        for i in range(len(Ranking_list)):
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
                current_precision_mean, current_precision_ci = calculate_statistics(Precision_k_list[i][j])
                current_regret_mean, current_regret_ci = calculate_statistics(Regret_k_list[i][j])
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
class BVFT_(ABC):
    def __init__(self, q_functions, data, gamma, rmax, rmin,file_name_pre, record: BvftRecord = BvftRecord(), q_type='torch_actor_critic_cont',
                 verbose=False, bins=None, data_size=5000,trajectory_num=200):
        self.data = data                                                        #Data D
        self.gamma = gamma                                                      #gamma
        self.res = 0                                                            #\epsilon k (discretization parameter set)
        self.q_sa_discrete = []                                                 #discreate q function
        self.q_to_data_map = []                                                 # to do
        self.q_size = len(q_functions)                                          #how many (s,a) pairs (q function length)
        self.verbose = verbose                                                  #if true, print log
        if bins is None:
            bins = [2, 3, 4, 8, 16, 100, 1e10]
        self.bins = bins                                                        #used for discretizing Q-values
        self.q_sa = []                                                          #all trajectory q s a
        self.r_plus_vfsp = []                                                   #reward
        self.q_functions = q_functions                                          #all q functions
        self.record = record
        self.file_name = file_name_pre

        if q_type == 'tabular':
            self.n = len(data)
            rewards = np.array([t[2] for t in self.data])
            actions = [int(t[1]) for t in self.data]
            states = np.array([t[0] for t in self.data])
            for Q in q_functions:
                self.q_sa.append(np.array([Q[states[i], actions[i]] for i in range(self.n)]))
                vfsp = np.array([0.0 if t[3] is None else np.max(Q[t[3]]) for t in self.data])
                self.r_plus_vfsp.append(rewards + self.gamma * vfsp)   #value after one bellman iteration

        elif q_type == 'keras_standard':
            self.n = len(data)
            rewards = np.array([t[2] for t in self.data])
            actions = [int(t[1]) for t in self.data]
            next_states = np.array([t[3][0] for t in self.data])
            states = np.array([t[0][0] for t in self.data])
            for Q in q_functions:
                qs = Q.predict(states)
                self.q_sa.append(np.array([qs[i][actions[i]] for i in range(self.n)]))
                vfsp = np.max(Q.predict(next_states), axis=1)
                self.r_plus_vfsp.append(rewards + self.gamma * vfsp)

        elif q_type == 'torch_actor_critic_cont':              #minimum batch size
            # batch_size = 1000
            # self.data.batch_size = batch_dim                                #batch size
            self.q_sa = [np.zeros(data_size) for _ in q_functions]             #q_functions corresponding 0
            self.r_plus_vfsp = [np.zeros(data_size) for _ in q_functions]      #initialization 0
            ptr = 0
            while ptr < trajectory_num:                                             #for everything in data size
                length = self.data.get_iter_length(ptr)
                state, action, next_state, reward, done = self.data.sample(ptr)
                # print("state : ",state)
                # print("reward : ", reward)
                # print("next state : ",next_state)
                for i in range(len(q_functions)):
                    actor= q_functions[i]
                    critic= q_functions[i]
                    # self.q_sa[i][ptr:ptr + length] = critic.predict_value(state, action).cpu().detach().numpy().flatten()[
                    #                                  :length]
                    self.q_sa[i][ptr:ptr + length] = critic.predict_value(state, action).flatten()[
                                                     :length]
                    # print("self qa : ",self.q_sa[i][ptr:ptr + 20])
                    # print("done : ",done)
                    # print("reward : ",reward)
                    # print("type state : ",type(state))
                    # print("type next state : ",type(next_state))
                    # print("action : ",actor.predict(next_state))
                    # print("predicted qa value : ",critic.predict_value(next_state, actor.predict(next_state)))
                    vfsp = (reward + critic.predict_value(next_state, actor.predict(next_state)) * done * self.gamma)

                    # self.r_plus_vfsp[i][ptr:ptr + length] = vfsp.cpu().detach().numpy().flatten()[:length]
                    self.r_plus_vfsp[i][ptr:ptr + length] = vfsp.flatten()[:length]
                    # print("self r plus vfsp : ",self.r_plus_vfsp[i][ptr:ptr + 20])
                ptr += 1
            self.n = data_size  #total number of data points

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

    @abstractmethod
    def compute_loss(self, q1, groups):                                 #
        pass

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
class Bvft_zero(policy_select):
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
        Bvft_Q_saving_folder = "Bvft_res_0"
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
        resolution_list = np.array([0.00001])
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
class BVFT_abs(BVFT_):
    def compute_loss(self, q1, groups):
        Tf = self.r_plus_vfsp[q1].copy()
        for group in groups:
            Tf[group] = np.mean(Tf[group])
        diff = self.q_sa[q1] - Tf
        return np.sqrt(np.mean(np.abs(diff )))  #square loss function
class Bvft_abs(policy_select):
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
        Bvft_Q_saving_folder = "Bvft_abs"
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
                bvft_instance = BVFT_abs(q_functions, test_data, gamma, rmax, rmin, policy_name_list[i], record,
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
class Bvft_FQE(policy_select):
    def select_Q(self):
        device = self.device
        print("begin save best Q, current device : ", device)
        whole_dataset = self.whole_dataset
        env = self.env

        Bvft_saving_folder = "Policy_ranking_saving_place"
        Bvft_Q_saving_folder = "Bvft_abs"
        self.data_saving_path.append(Bvft_Q_saving_folder)
        Bvft_Q_saving_path = os.path.join(Bvft_saving_folder, Bvft_Q_saving_folder)
        if not os.path.exists(Bvft_Q_saving_path):
            os.makedirs(Bvft_Q_saving_path)
        policy_name_list, policy_list = self.load_policy(device)

        Q_FQE, Q_name_list, FQE_step_Q_list = self.load_FQE(policy_name_list, self.FQE_saving_step_list, replay_buffer,
                                                       device)  # 1d: how many policy #2d: how many step #3d: 4
        FQE_lr_list = [1e-4, 2e-5]
        FQE_hl_list = [[128, 256], [128, 1024]]
        resolution_list = np.array([0.1, 0.2, 0.5, 0.7, 1.0]) * 100
        # print("input resolution list for Bvft : ", resolution_list)
        Bvft_folder = "FQE_"
        if not os.path.exists(Bvft_folder):
            os.makedirs(Bvft_folder)

        line_name_list = []
        for i in range(len(FQE_saving_step_list)):
            for j in range(len(FQE_lr_list)):
                for k in range(len(FQE_hl_list)):
                    line_name_list.append('FQE_' + str(FQE_lr_list[j]) + '_' + str(FQE_hl_list[k]) + '_' + str(
                        FQE_saving_step_list[i]) + "step")
        print("Q_fqe: ", Q_FQE)
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
            print(q_name_functions)
            sys.exit()
            ranking_list = rank_elements_lower_higher(min_loss_list)
            best_ranking_index = np.argmin(ranking_list)
            save_list = [q_name_functions[best_ranking_index]]
            save_as_txt(Bvft_Q_result_saving_path, save_list)
            save_as_pkl(Bvft_Q_result_saving_path, save_list)
            delete_files_in_folder(Bvft_folder)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
k = 5
num_runs = 300
FQE_saving_step_list = [2000000]
initial_state = 12345
# data_saving_path = ["Bvft_ranking","Bvft_res_0","Bvft_abs"]
data_saving_path = ["Bvft_ranking","Bvft_abs"]
bvft_obj = Bvft_poli(device, data_saving_path, whole_dataset,env,k,num_runs,FQE_saving_step_list,initial_state)
bvft_res_0 = Bvft_zero(device, data_saving_path, whole_dataset,env,k,num_runs,FQE_saving_step_list,initial_state)
bvft_FQE = Bvft_FQE(device, data_saving_path, whole_dataset,env,k,num_runs,FQE_saving_step_list,initial_state)
bvft_FQE.select_Q()
# bvft_abs_0 = Bvft_abs(device, data_saving_path, whole_dataset,env,k,num_runs,FQE_saving_step_list,initial_state)
# bvft_obj.select_Q()
# bvft_res_0.select_Q()
# bvft_abs_0.select_Q()
# bvft_obj.calculate_k(data_saving_path,self.data_saving_path,self.FQE_saving_step_list,self.initial_state,self.k,self.num_runs)
bvft_res_0.run()

