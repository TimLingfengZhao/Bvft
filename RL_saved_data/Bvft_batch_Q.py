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

def load_policy(device):
    policy_folder = 'policy_trained'
    if not os.path.exists(policy_folder ):
        os.makedirs(policy_folder )

    policy_name_list = []
    policy_list = []
    for policy_file_name in os.listdir(policy_folder):          #policy we want to evaluate
        policy_path = os.path.join(policy_folder, policy_file_name)
        policy = d3rlpy.load_learnable(policy_path, device=device)

        policy_name_list.append(policy_file_name[:-3])
        policy_list.append(policy)
    return policy_name_list, policy_list
def load_FQE(policy_name_list,FQE_step_list,replay_buffer,device):
    policy_folder = 'policy_trained'
    FQE_lr_list = [1e-4,2e-5]
    FQE_hl_list = [[128,256],[128,1024]]
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
                        FQE_step ) + "step" + "_"
                    FQE_model_name = FQE_model_pre + policy_file_name
                    FQE_policy_name.append(FQE_model_name)

                    FQE_model_name = FQE_model_name+ ".pt"
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
    return Q_FQE,Q_name_list,FQE_step_Q_list

def get_min_loss(loss_list): #input 2d list, return 1d list
    # print("loss list : ",loss_list)
    if (len(loss_list)==1):
        return loss_list[0]
    min_loss = []
    for i in range(len(loss_list[0])):
        current_loss = []
        for j in range(len(loss_list)):
            current_loss.append(loss_list[j][i])
        # print("current loss : ",current_loss)
        # print(len(loss_list))
        # print(len(loss_list[0]))
        # print(len(loss_list[0][0]))
        min_loss.append(min(current_loss))
    return min_loss


def Calculate_best_Q(FQE_saving_step_list,resolution_list):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("begin save best Q, current device : ",device)
    whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
    train_episodes = whole_dataset.episodes[0:2000]
    test_episodes = whole_dataset.episodes[2000:2276]

    print("min length test epsodes : ",find_min_length(test_episodes))
    print("min length train episodes : ",find_min_length(train_episodes))
    Bvft_batch_dim = find_min_length(test_episodes)
    # print("first train episodes : ",train_episodes[0])
    # print("one test episodes : ",test_episodes[0])
    # print("environment reward range : ",env.reward_range)    # print("environment reward range : ",env.reward_range)

    # print("test episodes : ",test_episodes)
    buffer = FIFOBuffer(limit=50000)
    replay_buffer_test = ReplayBuffer(buffer=buffer, episodes=test_episodes)
    buffer = FIFOBuffer(limit=50000)
    replay_buffer = ReplayBuffer(buffer=buffer, episodes=train_episodes)

    gamma = 0.99
    rmax, rmin = env.reward_range[0], env.reward_range[1]
    data_size = get_data_size(test_episodes)
    print("data size : ",data_size)
    test_data = CustomDataLoader(replay_buffer_test, batch_size=Bvft_batch_dim)

    Bvft_saving_folder = "Bvft_saving_place"
    Bvft_Q_saving_folder = "Bvft_Q_saving_place"

    Bvft_Q_saving_path = os.path.join(Bvft_saving_folder,Bvft_Q_saving_folder)

    Bvft_resolution_losses_saving_folder = "Bvft_resolution_loss_saving_place"
    Bvft_resolution_losses_saving_path = os.path.join(Bvft_saving_folder,Bvft_resolution_losses_saving_folder)
    if not os.path.exists(Bvft_resolution_losses_saving_path):
        os.makedirs(Bvft_resolution_losses_saving_path)
    if not os.path.exists(Bvft_Q_saving_path):
        os.makedirs(Bvft_Q_saving_path)
    policy_name_list, policy_list = load_policy(device)

    Q_FQE,Q_name_list,FQE_step_Q_list = load_FQE(policy_name_list,FQE_saving_step_list,replay_buffer,device) #1d: how many policy #2d: how many step #3d: 4
    FQE_lr_list = [1e-4,2e-5]
    FQE_hl_list = [[128,256],[128,1024]]
    # resolution_list = np.array([0.1, 0.2, 0.5, 0.7, 1.0]) * 100
    print("input resolution list for Bvft : ", resolution_list)
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
        Bvft_resolution_loss_policy_saving_path = os.path.join(Bvft_resolution_losses_saving_path, save_folder_name)
        # Bvft_resolution_loss_saving_name = "Bvft_mean_loss_"+str(resolution_list)
        # Bvft_resolution_loss_policy_saving_path = os.path.join(Bvft_resolution_loss_policy_saving_folder,Bvft_resolution_loss_saving_name)
        Bvft_Q_result_saving_path = os.path.join(Bvft_Q_saving_path, save_folder_name)

        q_functions = []
        q_name_functions = []
        for j in range(len(Q_FQE[0])):
            for h in range(len(Q_FQE[0][0])):
                q_functions.append(Q_FQE[i][j][h])
                q_name_functions.append(FQE_step_Q_list[i][j][h])
        Bvft_losses = []
        Bvft_final_resolution_loss = []
        for i in range(len(FQE_saving_step_list) * 4):
            current_list = []
            Bvft_final_resolution_loss.append(current_list)
        group_list = []
        for resolution in resolution_list:
            record = BvftRecord()
            bvft_instance = BVFT(q_functions, test_data, gamma, rmax, rmin, policy_name_list[i], record,
                                 "torch_actor_critic_cont", verbose=True, data_size=data_size,batch_dim=Bvft_batch_dim)
            # print("resolution : ",resolution)
            bvft_instance.run(resolution=resolution)
            group_list.append(record.group_counts[0])
            for i in range(len(record.losses[0])):
                Bvft_final_resolution_loss[i].append(record.losses[0][i])

            Bvft_losses.append(record.losses[0])
        # print('Bvft losses : ',Bvft_losses)
        min_loss_list = get_min_loss(Bvft_losses)
        # print("min loss list : ",min_loss_list)
        ranking_list = rank_elements(min_loss_list)

        best_ranking_index = np.argmin(ranking_list)
        save_list = [q_name_functions[best_ranking_index]]
        save_as_pkl(Bvft_resolution_loss_policy_saving_path,Bvft_final_resolution_loss)
        save_as_txt(Bvft_resolution_loss_policy_saving_path,Bvft_final_resolution_loss)
        save_as_txt(Bvft_Q_result_saving_path, save_list)
        save_as_pkl(Bvft_Q_result_saving_path, save_list)
        delete_files_in_folder(Bvft_folder)
        draw_Bvft_resolution_loss_graph(Bvft_final_resolution_loss,FQE_saving_step_list,resolution_list,save_folder_name,line_name_list,group_list)







def main():
    parser = argparse.ArgumentParser(description="Run specific Bvft based on learning rate and combination.")
    parser.add_argument("--FQE_saving_step_list", type=int, nargs='+', default=[500000, 1000000, 1500000, 2000000], help="Number of steps in each episode of FQE")
    # parser.add_argument("--resolution_list", type=float, nargs='+', default=[10., 20., 50., 70., 100.], help="Resolution list parameter for Bvft")
    parser.add_argument("--resolution_list", type=float, nargs='+', default=[0.0000002,0.0001, 0.01, 3., 10., 20.,10000.], help="Resolution list parameter for Bvft")
    args = parser.parse_args()

    Calculate_best_Q(FQE_saving_step_list = args.FQE_saving_step_list,resolution_list = args.resolution_list)
#--num_interval 5 --FQE_number_epoch 45 --FQE_episode_step 20000 --initial_state 12345 --m 10 --k 1 --num_runs 4
#--FQE_episode_step 100000 --FQE_Pickup_number 1 --FQE_total_step 900000 --m 2 --num_runs 10
#--FQE_saving_step_list 900000
if __name__ == "__main__":
    main()