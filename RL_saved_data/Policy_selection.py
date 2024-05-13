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
    def __init__(self,device, env):
        self.device = device
        self.env = env
    def select_Q(selfs):
        @abstractmethod
        pass
    def run(self):
        Bvft
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
        # print("k : ",k)
        k_precision_name = str(k)+"_mean_precision_"+str(num_runs)
        k_regret_name = str(k)+"_mean_regret"+str(num_runs)
        precision_ci_name = str(k)+"_CI_precision"+str(num_runs)
        regret_ci_name = str(k)+"_CI_regret"+str(num_runs)


        k_precision_mean_saving_path = os.path.join(Bvft_k_save_path,k_precision_name)
        k_regret_mean_saving_path = os.path.join(Bvft_k_save_path,k_regret_name)
        k_precision_ci_saving_path = os.path.join(Bvft_k_save_path,precision_ci_name)
        k_regret_ci_saving_path = os.path.join(Bvft_k_save_path,regret_ci_name)
        plot_name_saving_path = os.path.join(Bvft_k_save_path,plot_name)

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
