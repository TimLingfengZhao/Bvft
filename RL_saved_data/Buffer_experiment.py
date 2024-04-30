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
def load_rankings(num_runs):
    Bvft_saving_folder = "Bvft_saving_place"
    Bvft_ranking_folder = "Bvft_FQE_avgQ_ranking"
    InitialQ_ranking_folder = "InitialQ_ranking"
    Random_ranking_folder = "Random_ranking"
    Bvft_ranking_path = os.path.join(Bvft_saving_folder, Bvft_ranking_folder)
    InitialQ_ranking_path = os.path.join(Bvft_saving_folder, InitialQ_ranking_folder)
    Random_ranking_path = os.path.join(Bvft_saving_folder, Random_ranking_folder)
    Bvft_ranking_list = []
    InitialQ_ranking_list = []
    Random_ranking_list = []

    Bvft_policy_name_list = []
    InitialQ_policy_name_list = []
    Random_policy_name_list = []
    for run in range(num_runs):
        run_folder = str(run) + "run"
        Bvft_ranking_run_folder = os.path.join(Bvft_ranking_path, run_folder)
        InitialQ_ranking_run_folder = os.path.join(InitialQ_ranking_path, run_folder)
        Random_ranking_run_folder = os.path.join(Random_ranking_path, run_folder)

        Bvft_ranking_run_path = os.path.join(Bvft_ranking_run_folder, "rankings")
        InitialQ_ranking_run_path = os.path.join(InitialQ_ranking_run_folder, "rankings")
        Random_ranking_run_path = os.path.join(Random_ranking_run_folder, "rankings")

        Bvft_policy_name_path = os.path.join(Bvft_ranking_run_folder, "policy_names")
        InitialQ_policy_name_path = os.path.join(InitialQ_ranking_run_folder, "policy_names")
        Random_policy_name_path = os.path.join(Random_ranking_run_folder, "policy_names")

        Bvft_ranking_run_list = load_from_pkl(Bvft_ranking_run_path)
        InitialQ_ranking_run_list = load_from_pkl(InitialQ_ranking_run_path)
        Random_ranking_run_list = load_from_pkl(Random_ranking_run_path)

        Bvft_policy_name_run_list = load_from_pkl(Bvft_policy_name_path)
        InitialQ_policy_name_run_list = load_from_pkl(InitialQ_policy_name_path)
        Random_policy_name_run_list = load_from_pkl(Random_policy_name_path)

        Bvft_ranking_list.append(Bvft_ranking_run_list)
        InitialQ_ranking_list.append(InitialQ_ranking_run_list)
        Random_ranking_list.append(Random_ranking_run_list)

        Bvft_policy_name_list.append(Bvft_policy_name_run_list)
        InitialQ_policy_name_list.append(InitialQ_policy_name_run_list)
        Random_policy_name_list.append(Random_policy_name_run_list)

    return Bvft_ranking_list, InitialQ_ranking_list, Random_ranking_list, Bvft_policy_name_list, InitialQ_policy_name_list, Random_policy_name_list


def calculate_k(num_runs, initial_state, env, k):
    Bvft_ranking_list, InitialQ_ranking_list, Random_ranking_list, Bvft_policy_name_list, InitialQ_policy_name_list, Random_policy_name_list = load_rankings(
        num_runs)
    Bvft_k_precision_list = []
    InitialQ_k_precision_list = []
    Random_k_precision_list = []

    Bvft_k_regret_list = []
    InitialQ_k_regret_list = []
    Random_k_regret_list = []

    for i in range(num_runs):
        Bvft_k_precision_list.append(calculate_top_k_precision(initial_state, env,
                                                               Bvft_policy_name_list[i],
                                                               Bvft_ranking_list[i],
                                                               k))
        Bvft_k_regret_list.append(calculate_top_k_normalized_regret(Bvft_ranking_list[i],
                                                                    Bvft_policy_name_list[i],
                                                                    env,
                                                                    k))
        InitialQ_k_precision_list.append(calculate_top_k_precision(initial_state, env,
                                                                   InitialQ_policy_name_list[i],
                                                                   InitialQ_ranking_list[i],
                                                                   k))
        InitialQ_k_regret_list.append(calculate_top_k_normalized_regret(InitialQ_ranking_list[i],
                                                                        InitialQ_policy_name_list[i],
                                                                        env,
                                                                        k))
        Random_k_precision_list.append(calculate_top_k_precision(initial_state, env,
                                                                 Random_policy_name_list[i],
                                                                 Random_ranking_list[i],
                                                                 k))
        Random_k_regret_list.append(calculate_top_k_normalized_regret(Random_ranking_list[i],
                                                                      Random_policy_name_list[i],
                                                                      env,
                                                                      k))
    Bvft_k_precision_result = []
    InitialQ_k_precision_result = []
    Random_k_precision_result = []

    Bvft_k_regret_result = []
    InitialQ_k_regret_result = []
    Random_k_regret_result = []
    for i in range(k):
        Bvft_precision = []
        InitialQ_precision = []
        Random_precision = []
        Bvft_regret = []
        InitialQ_regret = []
        Random_regret = []
        for j in range(num_runs):
            Bvft_precision.append(Bvft_k_precision_list[j][i])
            InitialQ_precision.append(InitialQ_k_precision_list[j][i])
            Random_precision.append(Random_k_precision_list[j][i])
            Bvft_regret.append(Bvft_k_regret_list[j][i])
            InitialQ_regret.append(Bvft_k_regret_list[j][i])
            Random_regret.append(Random_k_regret_list[j][i])
        Bvft_k_precision_result.append(Bvft_precision)
        InitialQ_k_precision_result.append(InitialQ_precision)
        Random_k_precision_result.append(Random_precision)
        Bvft_k_regret_result.append(Bvft_regret)
        InitialQ_k_regret_result.append(InitialQ_regret)
        Random_k_regret_result.append(Random_regret)

    k_precision_list = [Bvft_k_precision_result, InitialQ_k_precision_result, Random_k_precision_result]
    k_regret_list = [Bvft_k_regret_result, InitialQ_k_regret_result, Random_k_regret_result]

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


def plot_normalized_k( initial_state ,k, num_runs):
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
    k_precision_name = str(k)+"_mean_precision_"+str(num_runs)+".pkl"
    k_regret_name = str(k)+"_mean_regret"+str(num_runs)+".pkl"
    precision_ci_name = str(k)+"_CI_precision"+str(num_runs)+".pkl"
    regret_ci_name = str(k)+"_CI_regret"+str(num_runs)+".pkl"
    plot_name = "line_names"+".pkl"

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
        precision_mean_list, regret_mean_list, precision_ci_list, regret_ci_list, line_name_list=calculate_k(num_runs, initial_state, env, k)

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
    parser.add_argument("--initial_state", type=int, default=12345, help="Initial state in real environment")
    parser.add_argument("--k", type=int, default=2, help="number k")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of sample random policy in ranking")

    args = parser.parse_args()

    plot_normalized_k( args.initial_state ,args.k, args.num_runs)
#--initial_state 12345 --k 2 --num_runs 5
if __name__ == "__main__":
    main()