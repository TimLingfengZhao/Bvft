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
# def load_rankings(num_runs):
#     Bvft_saving_folder = "Bvft_saving_place"
#     Bvft_ranking_folder = "Bvft_FQE_avgQ_ranking"
#     InitialQ_ranking_folder = "InitialQ_ranking"
#     Random_ranking_folder = "Random_ranking"
#     Bvft_ranking_path = os.path.join(Bvft_saving_folder, Bvft_ranking_folder)
#     InitialQ_ranking_path = os.path.join(Bvft_saving_folder, InitialQ_ranking_folder)
#     Random_ranking_path = os.path.join(Bvft_saving_folder, Random_ranking_folder)
#     Bvft_ranking_list = []
#     InitialQ_ranking_list = []
#     Random_ranking_list = []
#
#     Bvft_policy_name_list = []
#     InitialQ_policy_name_list = []
#     Random_policy_name_list = []
#     for run in range(num_runs):
#         run_folder = str(run) + "run"
#         Bvft_ranking_run_folder = os.path.join(Bvft_ranking_path, run_folder)
#         InitialQ_ranking_run_folder = os.path.join(InitialQ_ranking_path, run_folder)
#         Random_ranking_run_folder = os.path.join(Random_ranking_path, run_folder)
#
#         Bvft_ranking_run_path = os.path.join(Bvft_ranking_run_folder, "rankings")
#         InitialQ_ranking_run_path = os.path.join(InitialQ_ranking_run_folder, "rankings")
#         Random_ranking_run_path = os.path.join(Random_ranking_run_folder, "rankings")
#
#         Bvft_policy_name_path = os.path.join(Bvft_ranking_run_folder, "policy_names")
#         InitialQ_policy_name_path = os.path.join(InitialQ_ranking_run_folder, "policy_names")
#         Random_policy_name_path = os.path.join(Random_ranking_run_folder, "policy_names")
#
#         Bvft_ranking_run_list = load_from_pkl(Bvft_ranking_run_path)
#         InitialQ_ranking_run_list = load_from_pkl(InitialQ_ranking_run_path)
#         Random_ranking_run_list = load_from_pkl(Random_ranking_run_path)
#
#         Bvft_policy_name_run_list = load_from_pkl(Bvft_policy_name_path)
#         InitialQ_policy_name_run_list = load_from_pkl(InitialQ_policy_name_path)
#         Random_policy_name_run_list = load_from_pkl(Random_policy_name_path)
#
#         Bvft_ranking_list.append(Bvft_ranking_run_list)
#         InitialQ_ranking_list.append(InitialQ_ranking_run_list)
#         Random_ranking_list.append(Random_ranking_run_list)
#
#         Bvft_policy_name_list.append(Bvft_policy_name_run_list)
#         InitialQ_policy_name_list.append(InitialQ_policy_name_run_list)
#         Random_policy_name_list.append(Random_policy_name_run_list)
#
#     return Bvft_ranking_list, InitialQ_ranking_list, Random_ranking_list, Bvft_policy_name_list, InitialQ_policy_name_list, Random_policy_name_list

def pick_policy(m,device):
    policy_folder = 'policy_trained'
    if not os.path.exists(policy_folder ):
        os.makedirs(policy_folder )
    policy_files = sample_files(policy_folder, m)
    policy_name_list = []
    policy_list = []
    for policy_file_name in policy_files:          #policy we want to evaluate
        policy_path = os.path.join(policy_folder, policy_file_name)
        policy = d3rlpy.load_learnable(policy_path, device=device)

        policy_name_list.append(policy_file_name[:-3])
        policy_list.append(policy)
    return policy_name_list, policy_list
# def pick_FQE(policy_name_list,device,number_FQE,FQE_total_step,FQE_episode_step,replay_buffer):
#     policy_folder = 'policy_trained'
#     FQE_lr_list = [1e-4,2e-5]
#     FQE_hl_list = [[128,256],[128,1024]]
#     Q_FQE = []
#     for policy_file_name in policy_name_list:
#         policy_path = os.path.join(policy_folder, policy_file_name)
#         policy = d3rlpy.load_learnable(policy_path + ".d3", device=device)
#         for FQE_learning_rate in FQE_lr_list:
#             for FQE_hidden_layer in FQE_hl_list:
#                 Q_list = []
#                 FQE_directory = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer)
#                 if not os.path.exists(FQE_directory):
#                     os.makedirs(FQE_directory)
#                 for i in range(number_FQE):
#                     FQE_step = int(FQE_total_step / number_FQE)
#                     FQE_model_pre = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer) + '_' + str(
#                         FQE_step * (i + 1)) + "step" + "_"
#                     FQE_model_name = FQE_model_pre + policy_file_name
#                     FQE_model_name = FQE_model_name+ ".pt"
#                     FQE_file_path = os.path.join(FQE_directory, FQE_model_name)
#                     fqeconfig = d3rlpy.ope.FQEConfig(
#                         learning_rate=FQE_learning_rate,
#                         encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=FQE_hidden_layer)
#                     )
#                     fqe = FQE(algo=policy, config=fqeconfig, device=device)
#                     fqe.build_with_dataset(replay_buffer)
#                     fqe.load_model(FQE_file_path)
#                     Q_list.append(fqe)
#         Q_FQE.append(Q_list)
#     return Q_FQE

def FQE_ranking(policy_name_list,FQE_saving_step_list,env):

    FQE_learning_rate_list = [0.0001,2e-5]
    FQE_hidden_layer_list = [[128, 256],[128, 1024]]
    performance_list = []
    FQE_name_list = []
    for steps in FQE_saving_step_list:
        for FQE_learning_rate in FQE_learning_rate_list:
            for FQE_hidden_layer in FQE_hidden_layer_list:
                FQE_name = "FQE_"+str(FQE_learning_rate)+"_"+str(FQE_hidden_layer)+"_"+str(steps)+"step"
                FQE_name_list.append(FQE_name)
                current_performance_list = []
                for i in range(len(policy_name_list)):
                    policy_name = policy_name_list[i]
                    current_performance_list.append(load_FQE_performance_specific(FQE_learning_rate,FQE_hidden_layer,steps,policy_name))
                performance_list.append(rank_elements_larger_higher(current_performance_list))
    return performance_list,FQE_name_list



def Bvft_ranking(policy_name_list,FQE_saving_step_list,env):
    Bvft_saving_place = "Bvft_saving_place"
    Bvft_Q_saving_place = "Bvft_Q_saving_place"
    Bvft_Q_saving_path = os.path.join(Bvft_saving_place,Bvft_Q_saving_place)
    if not os.path.exists(Bvft_Q_saving_path):
        os.makedirs(Bvft_Q_saving_path)
    FQE_performance_list = []
    for i in range(len(policy_name_list)):
        policy_file_name = policy_name_list[i]
        folder_name = policy_file_name + "_" + str(FQE_saving_step_list)
        FQE_name_path = os.path.join(Bvft_Q_saving_path,folder_name)
        FQE_name = load_from_pkl(FQE_name_path)[0]
        FQE_performance = load_FQE_performance(FQE_name)
        FQE_performance_list.append(FQE_performance)
    FQE_rank_list = rank_elements_larger_higher(FQE_performance_list)
    return FQE_rank_list


def Initial_Q_ranking(policy_list,policy_name_list,env):
    performance_folder = "policy_returned_result"
    total_name = "policy_returned_total.txt"
    performance_total_path = os.path.join(performance_folder,total_name)
    performance_dict = load_dict_from_txt(performance_total_path)

    performance_list = []

    policy_folder = 'policy_trained'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for policy_name in policy_name_list:
        included = False
        if policy_name in performance_dict:
            performance_list.append(performance_dict[policy_name])
            included = True
        if not included:
            policy_path = os.path.join(policy_folder, policy_name)
            policy = d3rlpy.load_learnable(policy_path+".d3", device=device)
            performance_list.append(calculate_policy_value(env, policy, gamma=0.99,num_run=100))
    initialQ_rankings = rank_elements(performance_list)
    return initialQ_rankings
    # Bvft_saving_folder = "Bvft_saving_place"
    # Initial_Q_saving_folder = "InitialQ_ranking"
    # Initial_Q_ranking_saving_path_before = os.path.join(Bvft_saving_folder,Initial_Q_saving_folder)
    # if not os.path.exists(Initial_Q_ranking_saving_path_before):
    #     os.makedirs(Initial_Q_ranking_saving_path_before  )
    # runs_folder = str(num_runs)+"run"
    # Initial_Q_ranking_saving_path = os.path.join(Initial_Q_ranking_saving_path_before,runs_folder)
    # if not os.path.exists(Initial_Q_ranking_saving_path):
    #     os.makedirs(Initial_Q_ranking_saving_path )
    # policy_name_folder = "policy_names"
    # Initial_Q_name_folder = "rankings"
    # Policy_name_saving_path = os.path.join(Initial_Q_ranking_saving_path,policy_name_folder)
    # Ranking_saving_path = os.path.join(Initial_Q_ranking_saving_path,Initial_Q_name_folder)
    # save_as_txt(Policy_name_saving_path,policy_name_list)
    # save_as_pkl(Policy_name_saving_path,policy_name_list)
    # save_as_txt(Ranking_saving_path,initialQ_rankings)
    # save_as_pkl(Ranking_saving_path,initialQ_rankings)


def random_ranking(policy_name_list,num_runs):
    r_ranking = list(range(1,len(policy_name_list)+1))
    random.shuffle(r_ranking)
    return r_ranking
    # Bvft_saving_folder = "Bvft_saving_place"
    # random_ranking_saving_folder = "Random_ranking"
    # Random_ranking_saving_path_before = os.path.join(Bvft_saving_folder,random_ranking_saving_folder)
    # if not os.path.exists(Random_ranking_saving_path_before):
    #     os.makedirs(Random_ranking_saving_path_before  )
    # runs_folder = str(num_runs)+"run"
    # Random_ranking_saving_path = os.path.join(Random_ranking_saving_path_before,runs_folder)
    # if not os.path.exists(Random_ranking_saving_path):
    #     os.makedirs(Random_ranking_saving_path )
    # policy_name_folder = "policy_names"
    # ranking_name_folder = "rankings"
    # Policy_name_saving_path = os.path.join(Random_ranking_saving_path,policy_name_folder)
    # Ranking_saving_path = os.path.join(Random_ranking_saving_path,ranking_name_folder)
    # save_as_txt(Policy_name_saving_path,policy_name_list)
    # save_as_pkl(Policy_name_saving_path,policy_name_list)
    # save_as_txt(Ranking_saving_path,r_ranking)
    # save_as_pkl(Ranking_saving_path,r_ranking)
def calculate_k(FQE_saving_step_list, initial_state ,k, num_runs):
    whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    FQE_ranking_list = [] #1d: 多少个FQE #2d num_runs
    for i in range(len(FQE_saving_step_list)):
        for j in range(4):
            FQE_ranking_list.append([])
    Policy_name_list = []
    Bvft_ranking_list = []
    # Initial_Q_ranking_list = []
    Random_ranking_list = []
    for runs in range(num_runs):
        policy_name_list, policy_list = pick_policy(15,device)
        random_rank = random_ranking(policy_name_list,num_runs)
        Bvft_rank = Bvft_ranking(policy_name_list,FQE_saving_step_list,env)
        performance_list,FQE_name_list = FQE_ranking(policy_name_list, FQE_saving_step_list, env)
        for i in range(len(FQE_name_list)):
            FQE_ranking_list[i].append(performance_list[i])
        # Initial_Q_rank = Initial_Q_ranking(policy_list,policy_name_list,env)
        Bvft_ranking_list.append(Bvft_rank)
        # Initial_Q_ranking_list.append(Initial_Q_rank)
        Random_ranking_list.append(random_rank)
        Policy_name_list.append(policy_name_list)

    Bvft_k_precision_list = []
    # InitialQ_k_precision_list = []
    Random_k_precision_list = []
    FQE_k_precision_list = []

    FQE_k_regret_list = []
    Bvft_k_regret_list = []
    # InitialQ_k_regret_list = []
    Random_k_regret_list = []
    for j in range(len(FQE_ranking_list)):
        FQE_k_regret_list.append([])
        FQE_k_precision_list.append([])
    for i in range(num_runs):
        Bvft_k_precision_list.append(calculate_top_k_precision(initial_state, env,
                                                               Policy_name_list[i],
                                                               Bvft_ranking_list[i],
                                                               k))
        Bvft_k_regret_list.append(calculate_top_k_normalized_regret(Bvft_ranking_list[i],
                                                                    Policy_name_list[i],
                                                                    env,
                                                                    k))
        for j in range(len(FQE_ranking_list)):
            FQE_k_precision_list[j].append(calculate_top_k_precision(initial_state,env,Policy_name_list[i],FQE_ranking_list[j][i],k))
            FQE_k_regret_list[j].append(calculate_top_k_normalized_regret(FQE_ranking_list[j][i],Policy_name_list[i],env,k))


        # InitialQ_k_precision_list.append(calculate_top_k_precision(initial_state, env,
        #                                                            Policy_name_list[i],
        #                                                            Initial_Q_ranking_list[i],
        #                                                            k))
        # InitialQ_k_regret_list.append(calculate_top_k_normalized_regret(Initial_Q_ranking_list[i],
        #                                                                 Policy_name_list[i],
        #                                                                 env,
        #                                                                 k))
        Random_k_precision_list.append(calculate_top_k_precision(initial_state, env,
                                                                 Policy_name_list[i],
                                                                 Random_ranking_list[i],
                                                                 k))
        Random_k_regret_list.append(calculate_top_k_normalized_regret(Random_ranking_list[i],
                                                                      Policy_name_list[i],
                                                                      env,
                                                                      k))
    Bvft_k_precision_result = []
    # InitialQ_k_precision_result = []
    Random_k_precision_result = []
    FQE_k_precision_result = []

    Bvft_k_regret_result = []
    # InitialQ_k_regret_result = []
    Random_k_regret_result = []
    FQE_k_regret_result = []
    for i in range(len(FQE_ranking_list)):
        FQE_k_regret_result.append([])
        FQE_k_precision_result.append([])
    for i in range(k):
        Bvft_precision = []
        # InitialQ_precision = []
        Random_precision = []
        FQE_precision = []
        Bvft_regret = []
        # InitialQ_regret = []
        Random_regret = []
        FQE_regret = []
        for num in range(len(FQE_ranking_list)):
            FQE_regret.append([])
            FQE_precision.append([])
        for j in range(num_runs):
            Bvft_precision.append(Bvft_k_precision_list[j][i])
            # InitialQ_precision.append(InitialQ_k_precision_list[j][i])
            Random_precision.append(Random_k_precision_list[j][i])
            Bvft_regret.append(Bvft_k_regret_list[j][i])
            # InitialQ_regret.append(Bvft_k_regret_list[j][i])
            Random_regret.append(Random_k_regret_list[j][i])
            for k in range(len(FQE_ranking_list)):
                FQE_precision[k].append(FQE_k_precision_list[k][j][i])
                FQE_regret[k].append(FQE_k_regret_list[k][j][i])
        Bvft_k_precision_result.append(Bvft_precision)
        # InitialQ_k_precision_result.append(InitialQ_precision)
        Random_k_precision_result.append(Random_precision)
        Bvft_k_regret_result.append(Bvft_regret)
        # InitialQ_k_regret_result.append(InitialQ_regret)
        Random_k_regret_result.append(Random_regret)
        for k in range(len(FQE_ranking_list)):
            FQE_k_regret_result[k].append(FQE_regret[k])
            FQE_k_precision_result.append(FQE_precision[k])

    k_precision_list = [Bvft_k_precision_result,  Random_k_precision_result]
    k_regret_list = [Bvft_k_regret_result,  Random_k_regret_result]
    for k in range(len(FQE_ranking_list)):
        k_precision_list.append(FQE_k_precision_result[k])
        k_regret_list.append(FQE_k_regret_result[k])

    precision_mean_list = []
    regret_mean_list = []
    precision_ci_list = []
    regret_ci_list = []

    for i in range(len(k_precision_list)):
        current_precision_mean_list = []
        current_regret_mean_list = []
        current_precision_ci_list = []
        current_regret_ci_list = []
        for j in range(k):
            print("i : ",i)
            print("k : ",k)
            print("len : ",len(k_precision_list))
            print("len 0 : ",len(k_precision_list[0]))
            current_precision_mean, current_precision_ci = calculate_statistics(k_precision_list[i][j])
            current_regret_mean, current_regret_ci = calculate_statistics(k_regret_list[i][j])
            current_precision_mean_list.append(current_precision_mean)
            current_precision_ci_list.append(current_precision_ci)
            current_regret_mean_list.append(current_regret_mean)
            current_regret_ci_list.append(current_regret_ci)
        precision_mean_list.append(current_precision_mean_list)
        regret_mean_list.append(current_regret_mean_list)
        precision_ci_list.append(current_precision_ci_list)
        regret_ci_list.append(current_regret_ci_list)
    plot_name_list = ["Bvft-multiFQE-avgQ", "InitialQ", "Random"]
    for ele in FQE_name_list:
        plot_name_list.append(ele)

    Bvft_saving_place = 'Bvft_saving_place'
    Bvft_k = 'Bvft_k_results'
    Bvft_k_save_path = os.path.join(Bvft_saving_place, Bvft_k)
    if not os.path.exists(Bvft_k_save_path):
        os.makedirs(Bvft_k_save_path)
    print("k : ",k)
    k_precision_name = str(k)+"_mean_precision_"+str(num_runs)
    k_regret_name = str(k)+"_mean_regret"+str(num_runs)
    precision_ci_name = str(k)+"_CI_precision"+str(num_runs)
    regret_ci_name = str(k)+"_CI_regret"+str(num_runs)

    plot_name = "line_names"

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


def plot_normalized_k(FQE_saving_step_list, initial_state ,k, num_runs):
    #1d: different baselines
    #2d: precision, regret
    #3d: points
    print("start normalize plot k : ",k)
    whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
    Bvft_saving_place = 'Bvft_saving_place'
    Bvft_k = 'Bvft_k_results'
    Bvft_k_save_path = os.path.join(Bvft_saving_place, Bvft_k)
    if not os.path.exists(Bvft_k_save_path):
        os.makedirs(Bvft_k_save_path)
    k_precision_name = str(k)+"_mean_precision_"+str(num_runs)
    k_regret_name = str(k)+"_mean_regret"+str(num_runs)
    precision_ci_name = str(k)+"_CI_precision"+str(num_runs)
    regret_ci_name = str(k)+"_CI_regret"+str(num_runs)
    plot_name = "line_names"

    k_precision_mean_saving_path = os.path.join(Bvft_k_save_path,k_precision_name)
    k_regret_mean_saving_path = os.path.join(Bvft_k_save_path,k_regret_name)
    k_precision_ci_saving_path = os.path.join(Bvft_k_save_path,precision_ci_name)
    k_regret_ci_saving_path = os.path.join(Bvft_k_save_path,regret_ci_name)
    plot_name_saving_path = os.path.join(Bvft_k_save_path,plot_name)
    precision_path = os.path.join(Bvft_k_save_path,k_precision_name)
    if os.path.exists(precision_path):
        print("load saved data")
        precision_mean_list = load_from_pkl(k_precision_mean_saving_path)
        regret_mean_list = load_from_pkl(k_regret_mean_saving_path)
        precision_ci_list = load_from_pkl(k_precision_ci_saving_path)
        regret_ci_list = load_from_pkl(k_regret_ci_saving_path)
        line_name_list = load_from_pkl(plot_name_saving_path)
    else :
        precision_mean_list, regret_mean_list, precision_ci_list, regret_ci_list, line_name_list=calculate_k(FQE_saving_step_list, initial_state, k, num_runs)

    plot_mean_list = [precision_mean_list,regret_mean_list]
    plot_ci_list = [precision_ci_list,regret_ci_list]

    Bvft_plot_folder = os.path.join(Bvft_saving_place, "Bvft_plot")
    if not os.path.exists(Bvft_plot_folder):
        os.makedirs(Bvft_plot_folder)
    y_axis_names = ["k precision", "k regret"]
    colors = generate_unique_colors(len(plot_mean_list[0]))
    line_name = ["hopper-medium-expert-v0", "hopper-medium-expert-v0"]
    print("plot mean list : ",plot_mean_list)
    print("ci lsit : ",plot_ci_list)
    plot_subplots(data=plot_mean_list, save_path=Bvft_plot_folder, y_axis_names=y_axis_names,
                  line_names=line_name_list, colors=colors, ci=plot_ci_list)
    print("plot finished")
def main():
    parser = argparse.ArgumentParser(description="Plot k precision and k regret plot for 3 different rankings")
    parser.add_argument("--FQE_saving_step_list", type=int, nargs='+', default=[500000, 1000000, 1500000, 2000000], help="Number of steps in each episode of FQE")
    parser.add_argument("--initial_state", type=int, default=12345, help="Initial state in real environment")
    parser.add_argument("--k", type=int, default=5, help="number k")
    parser.add_argument("--num_runs", type=int, default=300,
                        help="Number of sample random policy in ranking")

    args = parser.parse_args()
    plot_normalized_k(args.FQE_saving_step_list, args.initial_state, args.k, args.num_runs)
#--initial_state 12345 --k 2 --num_runs 5
if __name__ == "__main__":
    main()