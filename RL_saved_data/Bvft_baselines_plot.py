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
    for run in num_runs:
        run_folder = run + "run"
        Bvft_ranking_run_folder = os.path.join(Bvft_ranking_path, run_folder)
        InitialQ_ranking_run_folder = os.path.join(InitialQ_ranking_path, run_folder)
        Random_ranking_run_folder = os.path.join(Random_ranking_path, run_folder)

        Bvft_ranking_run_path = os.path.join(Bvft_ranking_run_folder, "rankings.pkl")
        InitialQ_ranking_run_path = os.path.join(InitialQ_ranking_run_folder, "rankings.pkl")
        Random_ranking_run_path = os.path.join(Random_ranking_run_folder, "rankings.pkl")

        Bvft_policy_name_path = os.path.join(Bvft_ranking_run_folder, "policy_names.pkl")
        InitialQ_policy_name_path = os.path.join(InitialQ_ranking_run_folder, "policy_names.pkl")
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
    return Bvft_k_precision_result, InitialQ_k_precision_result, Random_k_precision_result, Bvft_k_regret_result, InitialQ_k_regret_result, Random_k_regret_result


def plot_normalized_k(num_interval, FQE_number_epoch, FQE_episode_step, initial_state, m, k, num_runs, env):
    Bvft_k_precision_result, InitialQ_k_precision_result, Random_k_precision_result, Bvft_k_regret_result, InitialQ_k_regret_result, Random_k_regret_result = calculate_k(
        num_runs, initial_state, env, k)

    Bvft_saving_place = 'Bvft_saving_place'
    Bvft_k = 'Bvft_k_results'
    Bvft_k_save_path = os.path.join(Bvft_saving_place, Bvft_k)
    if not os.path.exists(Bvft_k_save_path):
        os.makedirs(Bvft_k_save_path)
    plot_precision_list = []
    plot_regret_list = []
    ci_precision_list = []
    ci_regret_list = []
    for k_i in range(k):
        k_precision = 0
        k_regret = 0
        k_precision_name = str((k_i + 1)) + "_precision"
        k_regret_name = str((k_i + 1)) + "_regret"
        k_precision_ci_name = str(k_i + 1) + "_precision_ci"
        k_regret_ci_name = str(k_i + 1) + "_regret_ci"
        precision_path = os.path.join(Bvft_k_save_path, k_precision_name)
        regret_path = os.path.join(Bvft_k_save_path, k_regret_name)
        precision_ci_path = os.path.join(Bvft_k_save_path, k_precision_ci_name)
        regret_ci_path = os.path.join(Bvft_k_save_path, k_regret_ci_name)

        if (os.path.exists(regret_ci_path + ".pkl")):
            k_precision = load_from_pkl(precision_path)
            k_regret = load_from_pkl(regret_path)
            k_precision_ci = load_from_pkl(precision_ci_path)
            k_regret_ci = load_from_pkl(regret_ci_path)
            ci_precision_list.append(k_precision_ci[0])
            ci_regret_list.append(k_regret_ci[0])
            plot_precision_list.append(k_precision[0])
            plot_regret_list.append(k_regret[0])
        else:
            k_precision_list = []
            k_regret_list = []
            for i in range(num_runs):
                print("iteration : ", i)
                k_pre, k_reg = run_bvft(num_interval, FQE_number_epoch, FQE_episode_step, initial_state, m, (k_i + 1))
                k_precision_list.append(k_pre)
                k_regret_list.append(k_reg)
            k_precision = sum(k_precision_list) / len(k_precision_list)
            k_regret = sum(k_regret_list) / len(k_regret_list)
            k_precision_ci = calculate_statistics(k_precision_list)
            k_regret_ci = calculate_statistics(k_regret_list)

            k_pre_saving_path = os.path.join(Bvft_k_save_path, k_precision_name)
            k_reg_saving_path = os.path.join(Bvft_k_save_path, k_regret_name)
            k_pre_ci_saving_path = os.path.join(Bvft_k_save_path, k_precision_ci_name)
            k_reg_ci_saving_path = os.path.join(Bvft_k_save_path, k_regret_ci_name)

            save_as_pkl(k_pre_saving_path, [k_precision])
            save_as_pkl(k_reg_saving_path, [k_regret])
            save_as_pkl(k_pre_ci_saving_path, [k_precision_ci])
            save_as_pkl(k_reg_ci_saving_path, [k_regret_ci])

            save_as_txt(k_pre_saving_path, [k_precision])
            save_as_txt(k_reg_saving_path, [k_regret])
            save_as_txt(k_pre_ci_saving_path, [k_precision_ci])
            save_as_txt(k_reg_ci_saving_path, [k_regret_ci])
            ci_precision_list.append(k_precision_ci)
            ci_regret_list.append(k_regret_ci)
            plot_precision_list.append(k_precision)
            plot_regret_list.append(k_regret)
    total_precision_name = "total_" + str(k) + "_precision"
    total_reg_name = "total" + str(k) + "_regret"
    total_precision_ci = "total_" + str(k) + "_precision_ci"
    total_reg_ci = "total" + str(k) + "_regret_ci"
    total_k_pre_saving_path = os.path.join(Bvft_k_save_path, total_precision_name)
    total_k_reg_saving_path = os.path.join(Bvft_k_save_path, total_reg_name)
    total_k_pre_ci_saving_path = os.path.join(Bvft_k_save_path, total_precision_ci)
    total_k_reg_ci_saving_path = os.path.join(Bvft_k_save_path, total_reg_ci)

    save_as_pkl(total_k_pre_saving_path, plot_precision_list)
    save_as_pkl(total_k_reg_saving_path, plot_regret_list)
    save_as_pkl(total_k_pre_ci_saving_path, ci_precision_list)
    save_as_pkl(total_k_reg_ci_saving_path, ci_regret_list)

    save_as_txt(total_k_pre_saving_path, plot_precision_list)
    save_as_txt(total_k_reg_saving_path, plot_regret_list)
    save_as_txt(total_k_pre_ci_saving_path, ci_precision_list)
    save_as_txt(total_k_reg_ci_saving_path, ci_regret_list)

    plot_list = []
    ci_list = []
    precision_list = []
    precision_list.append(plot_precision_list)
    regre_list = []
    regre_list.append(plot_regret_list)

    precision_ci_list = []
    precision_ci_list.append(ci_precision_list)
    regret_ci_list = []
    regret_ci_list.append(ci_regret_list)

    plot_list.append(precision_list)
    plot_list.append(regre_list)

    ci_list.append(precision_ci_list)
    ci_list.append(regret_ci_list)
    Bvft_plot_folder = os.path.join(Bvft_saving_place, "Bvft_plot")
    if not os.path.exists(Bvft_plot_folder):
        os.makedirs(Bvft_plot_folder)
    y_axis_names = ["k precision", "k regret"]
    colors = generate_unique_colors(len(plot_list[0]))
    line_name = ["hopper-medium-expert-v0", "hopper-medium-expert-v0"]

    plot_subplots(data=plot_list, save_path=Bvft_plot_folder, y_axis_names=y_axis_names,
                  line_names=line_name, colors=colors, ci=ci_list)