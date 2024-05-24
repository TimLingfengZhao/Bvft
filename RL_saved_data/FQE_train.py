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
# tf.disable_v2_behavior()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
def load_checkpoint_policy(model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        model = d3rlpy.load_learnable(checkpoint_path,device=device)
        return True
    return False

def load_checkpoint_FQE(model, checkpoint_path,dataset):
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

#训练的时候直接保存
#disable autosave
#train完成 sleep 等以下没train wan de policy
#daytime = datetime.now()                   #流程train表格 对照
                                           #换d4rl dataset (try)
                                           #try exception (不需要)


def run_FQE(FQE_learning_rate,FQE_hidden_layer,FQE_total_step, FQE_episode_step,num_interval):
    FQE_number_epoch = FQE_total_step / FQE_episode_step
    whole_dataset, env = get_d4rl('hopper-medium-expert-v0')

    train_episodes = whole_dataset.episodes[0:2000]
    test_episodes = whole_dataset.episodes[2000:2276]

    buffer = FIFOBuffer(limit=1500000)

    replay_buffer = ReplayBuffer(buffer=buffer, episodes=train_episodes)

    FQE_reward_list = []
    FQE_name_list = []

    # policy_folder = 'policy_saving_space'
    policy_folder = 'policy_trained'
    FQE_directory = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer)
    FQE_checkpoint_directory = "FQE_checkpoints"
    if not os.path.exists(policy_folder ):
        os.makedirs(policy_folder )
    if not os.path.exists(FQE_checkpoint_directory):
        os.makedirs(FQE_checkpoint_directory)
    if not os.path.exists(FQE_directory):
        os.makedirs(FQE_directory)
    while(True):
        for policy_file_name in os.listdir(policy_folder):
            FQE_model_pre = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer) + '_'

            policy_path = os.path.join(policy_folder, policy_file_name)

            if os.path.isfile(policy_path):
                FQE_total_file_name = FQE_directory+"_"+str(FQE_total_step)+"step"+"_"+policy_file_name
                print("start to train " + FQE_total_file_name)
                FQE_total_file_name = FQE_total_file_name[:-2]+"pt"
                FQE_total_path = os.path.join(FQE_directory,FQE_total_file_name)
                if not os.path.exists(FQE_total_path):
                    policy = d3rlpy.load_learnable(policy_path, device=device)
                    fqeconfig = d3rlpy.ope.FQEConfig(
                        learning_rate=FQE_learning_rate,
                        encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=FQE_hidden_layer)
                    )
                    fqe = FQE(algo=policy, config=fqeconfig, device=device)

                    num_epoch = FQE_number_epoch
                    check_point_list = []

                    FQE_checkpoint_path = os.path.join(FQE_checkpoint_directory, FQE_total_file_name[:-2]+'_'+'checkpoint.pt')
                    FQE_checkpoint_list_path = os.path.join(FQE_checkpoint_directory, FQE_total_file_name[:-2]+'_'+'checkpoint_list.pkl')
                    if not load_checkpoint_FQE(fqe, FQE_checkpoint_path, replay_buffer):
                        for epoch in range(int(num_epoch)):
                            fqe.fit(dataset=replay_buffer,
                                    n_steps=FQE_episode_step,
                                    with_timestamp=False,
                                    )
                            fqe.save_model(FQE_checkpoint_path)
                            check_point_list.append(epoch)
                            save_list(check_point_list, FQE_checkpoint_list_path)
                            if ((epoch + 1) % num_interval == 0):
                                FQE_ep_name = FQE_model_pre + str((epoch + 1) * FQE_episode_step) + "step_" + policy_file_name
                                FQE_ep_name = FQE_ep_name[:-2] + "pt"
                                FQE_save_path = os.path.join(FQE_directory,FQE_ep_name)
                                fqe.save_model(FQE_save_path )
                            # FQE_name_list.append(
                            #    FQE_file_path[:-3] + "_" + str((epoch + 1) * FQE_episode_step) + "step")

                        if os.path.exists(FQE_checkpoint_list_path):
                            os.remove(FQE_checkpoint_list_path)
                        if os.path.exists(FQE_checkpoint_path):
                            os.remove(FQE_checkpoint_path)
                    else:
                        fqe.build_with_dataset(replay_buffer)
                        fqe.load_model(FQE_checkpoint_path)
                        check_point_list = read_list(check_point_list, FQE_checkpoint_list_path)
                        for epoch in range(check_point_list[-1] + 1, int(num_epoch)):
                            fqe.fit(dataset=replay_buffer,
                                    n_steps=FQE_episode_step,
                                    with_timestamp=False,
                                    )
                            fqe.save_model(FQE_checkpoint_path)
                            check_point_list.append(epoch)
                            save_list(check_point_list, FQE_checkpoint_list_path)
                            if ((epoch + 1) % num_interval == 0):
                                FQE_ep_name = FQE_model_pre + str((epoch + 1) * FQE_episode_step) + policy_file_name
                                FQE_ep_name = FQE_ep_name[:-2] + "pt"
                                FQE_save_path = os.path.join(FQE_directory, FQE_ep_name)
                                fqe.save_model(FQE_save_path)
                            # FQE_name_list.append(
                            #    FQE_file_path[:-3] + "_" + str((epoch + 1) * FQE_episode_step) + "step")
                        if os.path.exists(FQE_checkpoint_list_path):
                            os.remove(FQE_checkpoint_list_path)
                        if os.path.exists(FQE_checkpoint_path):
                            os.remove(FQE_checkpoint_path)
        # time.sleep(600)  #睡觉600s
        # FQE_plot_fold = "FQE_plot"
        # FQE_plot_folder = "plot_" + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer)
        # FQE_plot_path = os.path.join(FQE_plot_fold,FQE_plot_folder)
        # plot_and_save_bar_graph_with_labels(FQE_reward_list, FQE_name_list, FQE_plot_path)
        print("sleep now")
        time.sleep(600)
def run_FQE_1(FQE_total_step, FQE_episode_step,num_interval):
    FQE_learning_rate = 1e-4
    FQE_hidden_layer = [128, 256]
    run_FQE(FQE_learning_rate,FQE_hidden_layer,FQE_total_step, FQE_episode_step,num_interval)
def run_FQE_2(FQE_total_step, FQE_episode_step,num_interval):
    FQE_learning_rate = 1e-4
    FQE_hidden_layer = [128, 1024]
    run_FQE(FQE_learning_rate,FQE_hidden_layer,FQE_total_step, FQE_episode_step,num_interval)
def run_FQE_3(FQE_total_step, FQE_episode_step,num_interval):
    FQE_learning_rate = 2e-5
    FQE_hidden_layer = [128, 256]
    run_FQE(FQE_learning_rate,FQE_hidden_layer,FQE_total_step, FQE_episode_step,num_interval)
def run_FQE_4(FQE_total_step, FQE_episode_step,num_interval):
    FQE_learning_rate = 2e-5
    FQE_hidden_layer = [128, 1024]
    run_FQE(FQE_learning_rate,FQE_hidden_layer,FQE_total_step, FQE_episode_step,num_interval)

def fqe_train(fqe_choice, FQE_total_step, FQE_episode_step,num_interval):
    if(fqe_choice =="FQE_1" ):
        run_FQE_1(FQE_total_step, FQE_episode_step,num_interval)
    elif (fqe_choice == "FQE_2"):
        run_FQE_2(FQE_total_step, FQE_episode_step,num_interval)
    elif (fqe_choice == "FQE_3"):
        run_FQE_3(FQE_total_step, FQE_episode_step,num_interval)
    else:
        run_FQE_4(FQE_total_step, FQE_episode_step,num_interval)



def main():
    parser = argparse.ArgumentParser(description="Run specific FQE function based on learning rate and combination.")
    parser.add_argument("FQE", choices=["FQE_1", "FQE_2", "FQE_3", "FQE_4"], help="Identifier of the function to run")
    parser.add_argument("--FQE_total_step", type=int, default=2000000, help="Total number of steps for FQE training")
    parser.add_argument("--FQE_episode_step", type=int, default=400000, help="Number of steps in one episode for FQE training")
    parser.add_argument("--num_interval", type=int, default=5,
                        help="how many episode to save file")
    args = parser.parse_args()
    function_to_run = fqe_train(args.FQE,args.FQE_total_step,args.FQE_episode_step,args.num_interval)
    function_to_run()

if __name__ == "__main__":
    main()