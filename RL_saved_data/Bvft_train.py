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


class BVFT(object):
    def __init__(self, q_functions, data, gamma, rmax, rmin,file_name_pre, record: BvftRecord = BvftRecord(), q_type='torch_actor_critic_cont',
                 verbose=False, bins=None, data_size=5000,batch_dim = 1000):
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

        elif q_type == 'torch_actor_critic_cont':
            batch_size = min(1024,  data_size)                  #minimum batch size
            # batch_size = 1000
            self.data.batch_size = batch_size                                  #batch size
            self.q_sa = [np.zeros(data_size) for _ in q_functions]             #q_functions corresponding 0
            self.r_plus_vfsp = [np.zeros(data_size) for _ in q_functions]      #initialization 0
            ptr = 0
            while ptr < data_size:                                             #for everything in data size
                length = min(batch_size, data_size - ptr)
                state, action, next_state, reward, done = self.data.sample(length)
                print(type(state))
                print(type(action))
                for i in range(len(q_functions)):
                    actor= q_functions[i]
                    critic= q_functions[i]
                    # self.q_sa[i][ptr:ptr + length] = critic.predict_value(state, action).cpu().detach().numpy().flatten()[
                    #                                  :length]
                    self.q_sa[i][ptr:ptr + length] = critic.predict_value(state, action).flatten()[
                                                     :length]
                    # print(self.q_sa[i][ptr:ptr + length])
                    vfsp = (reward + critic.predict_value(next_state, actor.predict(next_state)) * done * self.gamma)

                    # self.r_plus_vfsp[i][ptr:ptr + length] = vfsp.cpu().detach().numpy().flatten()[:length]
                    self.r_plus_vfsp[i][ptr:ptr + length] = vfsp.flatten()[:length]
                    # print(self.r_plus_vfsp[i][ptr:ptr + length])
                ptr += batch_size
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

    def compute_loss(self, q1, groups):                                 #
        Tf = self.r_plus_vfsp[q1].copy()
        for group in groups:
            Tf[group] = np.mean(Tf[group])
        diff = self.q_sa[q1] - Tf
        return np.sqrt(np.mean(diff ** 2))  #square loss function

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


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    outlier_max = 400
    outlier_min = 0
    num_intervel = 1


    FQE_number_epoch = 20
    FQE_episode_step = 20000


    whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
    train_episodes = whole_dataset.episodes[0:2000]
    test_episodes = whole_dataset.episodes[2000:2276]

    buffer = FIFOBuffer(limit=50000)
    replay_buffer_test = ReplayBuffer(buffer=buffer, episodes=test_episodes)
    # uiuc = replay_buffer_test.sample_trajectory(1001)
    # print(uiuc.observations)
    # # print(uiuc.actions)
    # # print(uiuc.rewards)
    # # print(uiuc.terminals)
    # print(len(uiuc.observations[0:1000]))
    # print(len(uiuc.observations[0]))
    # print(type(uiuc.observations))
    # sys.exit()
    buffer = FIFOBuffer(limit=50000)
    replay_buffer = ReplayBuffer(buffer=buffer, episodes=train_episodes)

    policy_list = []
    policy_name_list= []
    # (0) Setup environment
    env = gym.make("Hopper-v4")
    policy_steps = 1000

    number_policy = 2
    num_comb = 4
    num_interval = 5

    Bvft_saving_place = 'Bvft_saving_place'
    Bvft_para = 'Bvft_parameter'
    Bvft_para_save_path = os.path.join(Bvft_saving_place,Bvft_para)
    if not os.path.exists(Bvft_para_save_path ):
        os.makedirs(Bvft_para_save_path )

    policy_folder = 'policy_trained'
    if not os.path.exists(policy_folder ):
        os.makedirs(policy_folder )
    FQE_lr_list = [1e-4,2e-5]
    FQE_hl_list = [[128,256],[128,1024]]
    policy_name_list = []
    Q_FQE = []
    for policy_file_name in os.listdir(policy_folder):
        Q_list = []
        policy_name_list.append(policy_file_name[:-3])
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

                        policy_path = os.path.join(policy_folder, policy_file_name)
                        policy = d3rlpy.load_learnable(policy_path, device=device)

                        fqeconfig = d3rlpy.ope.FQEConfig(
                            learning_rate=FQE_learning_rate,
                            encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=FQE_hidden_layer)
                        )
                        fqe = FQE(algo=policy, config=fqeconfig, device=device)
                        fqe.build_with_dataset(replay_buffer)
                        fqe.load_model(FQE_file_path)
                        Q_list.append(fqe)
        Q_FQE.append(Q_list)
    policy_name_save_path = os.path.join(Bvft_para_save_path,"policy_name_list")
    q_function_save_path = os.path.join(Bvft_para_save_path, "q_function_list")
    save_as_txt(policy_name_save_path,policy_name_list)
    save_as_pkl(policy_name_save_path,policy_name_list)
    save_as_txt(q_function_save_path,Q_FQE)
    save_as_pkl(q_function_save_path, Q_FQE)




    gamma = 0.99
    rmax, rmin = 1.0, 0.0
    record = BvftRecord()
    batch_dim = 1000
    test_data = CustomDataLoader(replay_buffer_test,    batch_size=batch_dim)

    for i in range(len(Q_FQE)):
        q_functions = Q_FQE[i]
        bvft_instance = BVFT(q_functions, test_data, gamma, rmax, rmin, policy_name_list[i],record, "torch_actor_critic_cont", verbose=True,batch_dim=1000)
        bvft_instance.run()