import numpy as np
import sys
import os
import pickle
from typing import Sequence
from d3rlpy.datasets import get_d4rl
import gym
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

from d3rlpy.dataset import Episode
import numpy as np
from BvftUtil import *
import pickle
import d3rlpy
from d3rlpy.models.q_functions import IQNQFunctionFactory
from d3rlpy.ope import FQE, FQEConfig
from d3rlpy.models.encoders import VectorEncoderFactory
import torch



def run_policy_evaluation(input_gamma, num_runs, device):
    print(f"Running evaluation with gamma={input_gamma}, num_runs={num_runs}, on device={device}")
    outlier_max = 400
    outlier_min = 0
    input_gamma = input_gamma
    num_run = num_runs

    whole_dataset, env = get_d4rl('hopper-medium-expert-v0')

    policy_returned_result_folder = "policy_returned_result"
    if not os.path.exists(policy_returned_result_folder):
        os.makedirs(policy_returned_result_folder)

    policy_normal_result_folder = "policy_returned_normal"
    policy_outlier_result_folder = "policy_returned_outlier"
    policy_total_result_folder = "policy_returned_total"

    policy_normal_path = os.path.join(policy_returned_result_folder, policy_normal_result_folder)
    policy_outlier_path = os.path.join(policy_returned_result_folder, policy_outlier_result_folder)
    policy_total_path = os.path.join(policy_returned_result_folder, policy_total_result_folder)

    policy_folder = 'policy_trained'
    policy_plot_folder = 'policy_returned_result'

    policy_normal_path = os.path.join(policy_returned_result_folder, policy_normal_result_folder )

    policy_total_name_list = []
    policy_total_reward_list = []

    plot_name_list = []
    plot_reward_list = []

    outlier_name_list = []
    outlier_reward_list = []
    while(True):
        for policy_file_name in os.listdir(policy_folder):
            policy_path = os.path.join(policy_folder, policy_file_name)
            print("sucess : ", policy_path)
            policy = d3rlpy.load_learnable(policy_path, device=device)

            total_reward = calculate_policy_value(env, policy, gamma=input_gamma, num_run=num_run)
            if (total_reward > outlier_max):
                outlier_name_list.append(policy_file_name[:-3])
                outlier_reward_list.append(total_reward)
                plot_name_list.append(policy_file_name[:3])
                plot_reward_list.append(outlier_max)
            elif  (total_reward < outlier_min):
                outlier_name_list.append(policy_file_name[:-3])
                outlier_reward_list.append(total_reward)
                plot_name_list.append(policy_file_name[:3])
                plot_reward_list.append(outlier_min)
            else:
                plot_name_list.append(policy_file_name[:-3])
                plot_reward_list.append(total_reward)
            policy_total_name_list.append(policy_file_name[:-3])
            policy_total_reward_list.append(total_reward)

        plot_and_save_bar_graph_with_labels(plot_reward_list, plot_name_list, policy_plot_folder)

        normal_dict = list_to_dict(plot_name_list, plot_reward_list)
        outlier_dict = list_to_dict(outlier_name_list, outlier_reward_list)
        total_dict = list_to_dict(policy_total_name_list, policy_total_reward_list)

        save_as_pkl(policy_normal_path, normal_dict)
        save_as_pkl(policy_outlier_path, outlier_dict)
        save_as_pkl(policy_total_path, total_dict)

        save_dict_as_txt(policy_normal_path, normal_dict)
        save_dict_as_txt(policy_outlier_path, outlier_dict)
        save_dict_as_txt(policy_total_path, total_dict)
        print("sleep noe")
        time.sleep(600)

if __name__ == "__main__":
    # tf.disable_v2_behavior()
    parser = argparse.ArgumentParser(description="Run policy evaluation and processing.")
    parser.add_argument("--input_gamma", type=float, default=0.99, help="Gamma value for evaluation.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs for evaluation.")
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run_policy_evaluation(args.input_gamma, args.num_runs, device)