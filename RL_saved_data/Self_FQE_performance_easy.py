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
# import tensorflow.compat.v1 as tf
from FQE_util import *

def run_FQE_evaluation(device,FQE_learning_rate,FQE_hidden_layer,FQE_total_step,FQE_episode_step):
    print(f"Running evaluation with learning rate={FQE_learning_rate}, hidden layer={FQE_hidden_layer}, on device={device}")
    outlier_max = 400
    outlier_min = 0
    FQE_number_epoch = int(FQE_total_step / FQE_episode_step)
    num_intervel = 1

    whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
    train_episodes = whole_dataset.episodes[0:2000]
    test_episodes = whole_dataset.episodes[2000:2276]
    buffer = FIFOBuffer(limit=1500000)
    replay_buffer = ReplayBuffer(buffer=buffer, episodes=train_episodes)

    policy_returned_result_folder = "policy_returned_result"
    if not os.path.exists(policy_returned_result_folder):
        os.makedirs(policy_returned_result_folder)

    Self_defined_FQE = "Self_defined_FQE"

    FQE_returned_folder = os.path.join(Self_defined_FQE,"FQE_returned_result")
    if not os.path.exists(FQE_returned_folder):
        os.makedirs(FQE_returned_folder)
    FQE_directory = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer)
    FQE_folder = os.path.join(FQE_returned_folder,FQE_directory)
    if not os.path.exists(FQE_folder):
        os.makedirs(FQE_folder)

    FQE_normal_result_folder = "FQE_returned_normal"
    FQE_outlier_result_folder = "FQE_returned_outlier"
    FQE_total_result_folder = "FQE_returned_total"

    FQE_normal_path = os.path.join(FQE_folder, FQE_normal_result_folder)
    FQE_outlier_path = os.path.join(FQE_folder,FQE_outlier_result_folder)
    FQE_total_path = os.path.join(FQE_folder, FQE_total_result_folder)





    policy_folder = 'policy_trained'
    policy_plot_folder = 'plot_policy_trained'

    FQE_total_name_list = []
    FQE_total_reward_list = []

    normal_name_list = []
    normal_reward_list = []

    outlier_name_list = []
    outlier_reward_list = []

    FQE_normal_path = os.path.join(FQE_folder, FQE_normal_result_folder)
    state_dim = 11
    action_dim = 3
    while(True):
        print("come while loop")
        for policy_file_name in os.listdir(policy_folder):
            for i in range(FQE_total_step):
                FQE_model_pre = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer) + '_'+ str((i + 1)) + "iteration"+"_"
                FQE_model_name = FQE_model_pre + policy_file_name[:-3]
                plot = False
                FQE_normal_dictionary = {}
                if (os.path.exists(FQE_normal_path)):
                    print("exist")
                    FQE_normal_dictionary = load_from_pkl(FQE_normal_path)
                    saved = True
                else:
                    saved = False
                if ((i + 1) % FQE_episode_step == 0):
                    FQE_model_name = FQE_model_name
                    plot = True
                    print("plot :  True ")
                if (plot):
                    Self_FQE_folder = os.path.join(Self_defined_FQE, FQE_directory)
                    FQE_file_path = os.path.join(Self_FQE_folder, FQE_model_name)
                    if not saved:
                        if (os.path.exists(FQE_file_path)) :
                            policy_path = os.path.join(policy_folder, policy_file_name)
                            policy = d3rlpy.load_learnable(policy_path, device=device)
                            fqe = continuous_FQE(state_dim, action_dim, hidden_layer_list=FQE_hidden_layer, device=device,
                                                 target_update_frequency=FQE_learning_rate)

                            fqe.load(FQE_file_path)

                            observation, info = env.reset(seed=12345)
                            action = policy.predict(
                                np.array([observation]))
                            observation_tensor = torch.tensor(np.array([observation]), dtype=torch.float32, device=device)
                            action_tensor = torch.tensor(action, dtype=torch.float32, device=device)
                            total_reward = fqe.Q(observation_tensor[0], action_tensor[0]).cpu().detach().numpy()[0]

                            if ((total_reward > outlier_max)):
                                outlier_name_list.append(FQE_model_name)
                                outlier_reward_list.append(total_reward)
                                normal_name_list.append(FQE_model_name)
                                normal_reward_list.append(outlier_max)
                            elif ((total_reward < 0)):
                                outlier_name_list.append(FQE_model_name)
                                outlier_reward_list.append(total_reward)
                                normal_name_list.append(FQE_model_name)
                                normal_reward_list.append(outlier_min)
                            else:
                                normal_name_list.append(FQE_model_name)
                                normal_reward_list.append(total_reward)
                            FQE_total_name_list.append(FQE_model_name)

                            FQE_total_reward_list.append(total_reward)
                            plot_and_save_bar_graph_with_labels_FQE(normal_reward_list, normal_name_list, FQE_folder)
                            # plot_and_save_bar_graph_with_labels_FQE(FQE_total_reward_list, FQE_total_name_list, FQE_folder)

                            normal_dict = list_to_dict(normal_name_list, normal_reward_list)
                            outlier_dict = list_to_dict(outlier_name_list, outlier_reward_list)
                            total_dict = list_to_dict(FQE_total_name_list, FQE_total_reward_list)

                            save_as_pkl(FQE_normal_path, normal_dict)
                            save_as_pkl(FQE_outlier_path, outlier_dict)
                            save_as_pkl(FQE_total_path, total_dict)

                            save_dict_as_txt(FQE_normal_path, normal_dict)
                            save_dict_as_txt(FQE_outlier_path, outlier_dict)
                            save_dict_as_txt(FQE_total_path, total_dict)
                            saved=True
                    else:
                        print("enter else")
                        if (os.path.exists(FQE_file_path) and not (
                        is_key_in_dict(FQE_model_name[:-3], FQE_normal_dictionary))):
                            policy_path = os.path.join(policy_folder, policy_file_name)
                            print(policy_path)
                            policy = d3rlpy.load_learnable(policy_path, device=device)
                            fqe = continuous_FQE(state_dim, action_dim, hidden_layer_list=FQE_hidden_layer,
                                                 device=device,
                                                 target_update_frequency=FQE_learning_rate)

                            fqe.load(FQE_file_path)

                            observation, info = env.reset(seed=12345)
                            action = policy.predict(
                                np.array([observation]))
                            observation_tensor = torch.tensor(np.array([observation]), dtype=torch.float32, device=device)
                            action_tensor = torch.tensor(action, dtype=torch.float32, device=device)
                            total_reward = fqe.Q(observation_tensor[0], action_tensor[0])[0]


                            if ((total_reward > outlier_max)):
                                outlier_name_list.append(FQE_model_name)
                                outlier_reward_list.append(total_reward)
                                normal_name_list.append(FQE_model_name)
                                normal_reward_list.append(outlier_max)
                            elif ((total_reward < 0)):
                                outlier_name_list.append(FQE_model_name)
                                outlier_reward_list.append(total_reward)
                                normal_name_list.append(FQE_model_name)
                                normal_reward_list.append(outlier_min)
                            else:
                                normal_name_list.append(FQE_model_name)
                                normal_reward_list.append(total_reward)
                            FQE_total_name_list.append(FQE_model_name)

                            FQE_total_reward_list.append(total_reward)
                            plot_and_save_bar_graph_with_labels_FQE(normal_reward_list, normal_name_list, FQE_folder)
                            # plot_and_save_bar_graph_with_labels_FQE(FQE_total_reward_list, FQE_total_name_list,
                            #                                         FQE_folder)

                            normal_dict = list_to_dict(normal_name_list, normal_reward_list)
                            outlier_dict = list_to_dict(outlier_name_list, outlier_reward_list)
                            total_dict = list_to_dict(FQE_total_name_list, FQE_total_reward_list)

                            save_as_pkl(FQE_normal_path, normal_dict)
                            save_as_pkl(FQE_outlier_path, outlier_dict)
                            save_as_pkl(FQE_total_path, total_dict)

                            save_dict_as_txt(FQE_normal_path, normal_dict)
                            save_dict_as_txt(FQE_outlier_path, outlier_dict)
                            save_dict_as_txt(FQE_total_path, total_dict)

        # time.sleep(600)
        print("dont while loop")
        break



def run_FQE_1(FQE_total_step,FQE_episode_step):
    FQE_learning_rate = 1e-4
    FQE_hidden_layer = [128, 256]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run_FQE_evaluation(device, FQE_learning_rate, FQE_hidden_layer,FQE_total_step, FQE_episode_step)
def run_FQE_2(FQE_total_step,FQE_episode_step):
    FQE_learning_rate = 1e-4
    FQE_hidden_layer = [128, 1024]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run_FQE_evaluation(device, FQE_learning_rate, FQE_hidden_layer,FQE_total_step, FQE_episode_step)
def run_FQE_3(FQE_total_step,FQE_episode_step):
    FQE_learning_rate = 2e-5
    FQE_hidden_layer = [128, 256]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run_FQE_evaluation(device, FQE_learning_rate, FQE_hidden_layer,FQE_total_step, FQE_episode_step)
def run_FQE_4(FQE_total_step,FQE_episode_step):
    FQE_learning_rate = 2e-5
    FQE_hidden_layer = [128, 1024]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run_FQE_evaluation(device, FQE_learning_rate, FQE_hidden_layer,FQE_total_step, FQE_episode_step)
function_map = {
    "FQE_1": run_FQE_1,
    "FQE_2": run_FQE_2,
    "FQE_3": run_FQE_3,
    "FQE_4": run_FQE_4,
}
def main():
    # tf.disable_v2_behavior()
    parser = argparse.ArgumentParser(description="Plot specific FQE function prediction plot based on learning rate and combination.")
#    parser.add_argument("FQE", choices=["FQE_1", "FQE_2", "FQE_3", "FQE_4"], help="Identifier of the function to run")
    parser.add_argument("--FQE_total_iter", type=int, default=2000, help="Total iterations for FQE training")
    parser.add_argument("--FQE_episode_iter", type=int, default=400, help="Number of iterations to save")
    args = parser.parse_args()
    # function_to_run = function_map[args.FQE]
    # function_to_run(args.FQE_total_step,args.FQE_episode_step)
    run_FQE_1(args.FQE_total_iter,args.FQE_episode_iter)
    run_FQE_2(args.FQE_total_iter,args.FQE_episode_iter)
    run_FQE_3(args.FQE_total_iter,args.FQE_episode_iter)
    run_FQE_4(args.FQE_total_iter,args.FQE_episode_iter)

if __name__ == "__main__":
    main()