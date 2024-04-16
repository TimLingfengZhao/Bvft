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



def run_bvft(num_interval,FQE_number_epoch,FQE_episode_step,initial_state,m,k):
    print("begin bvft")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
    train_episodes = whole_dataset.episodes[0:2000]
    test_episodes = whole_dataset.episodes[2000:2276]

    buffer = FIFOBuffer(limit=50000)
    replay_buffer_test = ReplayBuffer(buffer=buffer, episodes=test_episodes)
    buffer = FIFOBuffer(limit=50000)
    replay_buffer = ReplayBuffer(buffer=buffer, episodes=train_episodes)

    gamma = 0.99
    rmax, rmin = 1.0, 0.0
    record = BvftRecord()
    batch_dim = 1000
    test_data = CustomDataLoader(replay_buffer_test,    batch_size=batch_dim)

    policy_steps = 1000
    policy_folder = 'policy_trained'
    if not os.path.exists(policy_folder ):
        os.makedirs(policy_folder )
    FQE_lr_list = [1e-4,2e-5]
    FQE_hl_list = [[128,256],[128,1024]]

    policy_name_list = []
    policy_list = []
    Q_FQE = []
    policy_files = sample_files(policy_folder, m)
    for policy_file_name in policy_files:          #policy we want to evaluate
        policy_path = os.path.join(policy_folder, policy_file_name)
        policy = d3rlpy.load_learnable(policy_path, device=device)

        Q_list = []
        policy_name_list.append(policy_file_name[:-3])
        policy_list.append(policy)
        for FQE_learning_rate in FQE_lr_list:
            for FQE_hidden_layer in FQE_hl_list:
                FQE_directory = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer)
                if not os.path.exists(FQE_directory):
                    os.makedirs(FQE_directory)
                for i in range(FQE_number_epoch):
                    FQE_model_pre = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer) + '_'+ str(FQE_episode_step * (i + 1)) + "step"+"_"
                    FQE_model_name = FQE_model_pre + policy_file_name
                    if((i+1)%num_interval == 0):
                        FQE_model_name = FQE_model_name[:-3] + ".pt"
                        FQE_file_path = os.path.join(FQE_directory, FQE_model_name)

                        fqeconfig = d3rlpy.ope.FQEConfig(
                            learning_rate=FQE_learning_rate,
                            encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=FQE_hidden_layer)
                        )
                        fqe = FQE(algo=policy, config=fqeconfig, device=device)
                        fqe.build_with_dataset(replay_buffer)
                        fqe.load_model(FQE_file_path)
                        Q_list.append(fqe)
        Q_FQE.append(Q_list)
    for i in range(len(Q_FQE)):
        q_functions = Q_FQE[i]
        bvft_instance = BVFT(q_functions, test_data, gamma, rmax, rmin, policy_name_list[i],record, "torch_actor_critic_cont", verbose=True,batch_dim=1000)
        bvft_instance.run()
    Bvft_folder = "Bvft_Records"
    ranking_list = []
    avg_q_list = []
    for bvft_result in os.listdir(Bvft_folder):
        Bvft_path = os.path.join(Bvft_folder, bvft_result)
        bvft_result = BvftRecord.load(Bvft_path)
        ranking_list.append(bvft_result.ranking.tolist())
        avg_q_list.append(bvft_result.avg_q)
    best_ranking_index = []
    for ele in ranking_list:
        best_ranking_index.append(np.argmin(ele))
    best_avg_q = []
    for i in range(len(best_ranking_index)):
        best_avg_q.append(avg_q_list[i][best_ranking_index[i]])
    bvft_ranking = rank_elements(best_avg_q)
    k_precision = calculate_top_k_precision(initial_state,env,policy_list,bvft_ranking,k)
    k_regret = calculate_top_k_normalized_regret(bvft_ranking,policy_list,env,k)
    return k_precision, k_regret





if __name__ == '__main__':
    num_interval = 5
    FQE_number_epoch = 45
    FQE_episode_step = 20000
    initial_state = 12345
    m = 5
    num_runs = 30
    k = 1
    k_precision_list = []
    k_regret_list = []
    for i in range(num_runs):
        print("iteration : ", i)
        k_pre,k_reg = run_bvft(num_interval, FQE_number_epoch, FQE_episode_step, initial_state, m,k)
        k_precision_list.append(k_pre)
        k_regret_list.append(k_reg)
    k_precision = sum(k_precision_list)/len(k_precision_list)
    k_regret = sum(k_regret_list)/len(k_regret_list)
    Bvft_saving_place = 'Bvft_saving_place'
    Bvft_k = 'Bvft_k_results'
    Bvft_k_save_path = os.path.join(Bvft_saving_place,Bvft_k)
    if not os.path.exists(Bvft_k_save_path ):
        os.makedirs(Bvft_k_save_path )
    k_precision_name = str(k)+"_precision"
    k_regret_name = str(k)+"_regret"
    k_pre_saving_path = os.path.join(Bvft_k_save_path,k_precision_name)
    k_regre_saving_path = os.path.join(Bvft_k_save_path,k_regret_name)
    save_as_pkl(k_pre_saving_path, [k_precision])
    save_as_pkl(k_pre_saving_path,[k_regret])
    save_as_txt(k_pre_saving_path, [k_precision])
    save_as_txt(k_pre_saving_path,[k_regret])