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
import argparse
from scope_rl.ope import CreateOPEInput
import d3rlpy

from scope_rl.utils import check_array
import copy
import numpy as np
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from FQE_util import *
def load_checkpoint_FQE(model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        return True
    return False
def save_list(list,path):
    with open(path,'wb') as file:
        pickle.dump(list,file)
def read_list(list,path):
    with open(path,'rb') as file:
        list = pickle.load(file)
    return list
def FQE_train(choice_number, save_iter):
    learning_rate = 1e-4
    hidden_layer = [128, 256]
    if choice_number == 1 :
        hidden_layer = [128, 1024]
    elif choice_number == 2:
        learning_rate = 2e-5
    elif choice_number == 3:
        learning_rate = 2e-5
        hidden_layer = [128, 1024]
    whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
    train_episodes = whole_dataset.episodes[0:2000]
    test_episodes = whole_dataset.episodes[2000:2276]

    buffer = FIFOBuffer(limit=1500000)

    replay_buffer = ReplayBuffer(buffer=buffer, episodes=train_episodes)
    policy_folder = 'policy_trained'
    # policy_name = "bcq_300000_1e-05_2_64_200000step.d3"
    self_defined_FQE = "Self_defined_FQE"
    if not os.path.exists(self_defined_FQE):
        os.makedirs(self_defined_FQE)
    state_dim = 11
    action_dim = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    FQE_directory = 'FQE_' + str(learning_rate) + '_' + str(hidden_layer)
    FQE_directory = os.path.join(self_defined_FQE, FQE_directory)
    if not os.path.exists(FQE_directory):
        os.makedirs(FQE_directory)

    FQE_model_pre = 'FQE_' + str(learning_rate) + '_' + str(hidden_layer) + '_'

    FQE_checkpoint_directory = "FQE_checkpoints"
    FQE_checkpoint_directory = os.path.join(self_defined_FQE, FQE_checkpoint_directory)

    if not os.path.exists(FQE_checkpoint_directory):
        os.makedirs(FQE_checkpoint_directory)
    for policy_file_name in os.listdir(policy_folder):

        policy_path = os.path.join(policy_folder, policy_file_name)
        policy = d3rlpy.load_learnable(policy_path, device=device)
        FQE_total_file_name = FQE_directory + "_" + str(2000) + "step" + "_" + policy_file_name

        FQE_checkpoint_list_path = os.path.join(FQE_checkpoint_directory,
                                                FQE_total_file_name[:-2] + '_' + 'checkpoint_list.pkl')
        FQE_checkpoint_path = os.path.join(FQE_checkpoint_directory, FQE_total_file_name[:-2] + '_' + 'checkpoint')

        fqe = continuous_FQE(state_dim, action_dim, hidden_layer_list=hidden_layer, device=device,
                             target_update_frequency=learning_rate)
        test_data = CustomDataLoader(replay_buffer, batch_size=1000)
        check_point_list = []
        if not load_checkpoint_FQE(fqe, FQE_checkpoint_path):
            for i in range(2000):
                fqe.train(test_data, policy, i)
                fqe.save(FQE_checkpoint_path)
                check_point_list.append(i)
                save_list(check_point_list, FQE_checkpoint_list_path)
                if (i+1) % save_iter == 0 :
                    FQE_ep_name = FQE_model_pre + str(i) + "iteration_" + policy_file_name
                    FQE_ep_name = FQE_ep_name[:-2]
                    FQE_save_path = os.path.join(FQE_directory, FQE_ep_name)
                    fqe.save(FQE_save_path)


            if os.path.exists(FQE_checkpoint_list_path):
                os.remove(FQE_checkpoint_list_path)
            if os.path.exists(FQE_checkpoint_path):
                os.remove(FQE_checkpoint_path)
        else:
            fqe.load(FQE_checkpoint_path)
            check_point_list = read_list(check_point_list, FQE_checkpoint_list_path)
            for i in range(check_point_list[-1] + 1,2000):
                fqe.train(test_data, policy, i)
                fqe.save(FQE_checkpoint_path)
                check_point_list.append(i)
                save_list(check_point_list, FQE_checkpoint_list_path)
                if (i+1) % save_iter == 0 :
                    FQE_ep_name = FQE_model_pre + str(i) + "iteration_" + policy_file_name
                    FQE_ep_name = FQE_ep_name[:-2]
                    FQE_save_path = os.path.join(FQE_directory, FQE_ep_name)
                    fqe.save(FQE_save_path)
def main():
    parser = argparse.ArgumentParser(description="Run specific FQE function based on learning rate and combination.")
    parser.add_argument("FQE", choices=["0", "1", "2", "3"], help="0：1e-4,256 1: 1e-4, 1024 2: 2e-5,256 3: 2e-5: 1024")
    parser.add_argument("--save_iter", type=int, default=400,
                        help="Iteration basic number to save")
    args = parser.parse_args()
    FQE_train(args.FQE, args.save_iter)

if __name__ == "__main__":
    main()