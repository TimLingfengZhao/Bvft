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
import tensorflow.compat.v1 as tf

def run_FQE_evaluation(device,FQE_learning_rate,FQE_hidden_layer,FQE_saving_step_list):
    print(f"Plot FQE MSE with learning rate ={FQE_learning_rate}, hidden layer={FQE_hidden_layer}, on device={device}")

    whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
    train_episodes = whole_dataset.episodes[0:2000]
    test_episodes = whole_dataset.episodes[2000:2276]
    buffer = FIFOBuffer(limit=500000)
    replay_buffer = ReplayBuffer(buffer=buffer, episodes=train_episodes)

    policy_returned_result_folder = "policy_returned_result"
    if not os.path.exists(policy_returned_result_folder):
        os.makedirs(policy_returned_result_folder)

    FQE_returned_folder = "FQE_returned_result"
    FQE_directory = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer)
    FQE_folder = os.path.join(FQE_returned_folder,FQE_directory)
    if not os.path.exists(FQE_folder):
        os.makedirs(FQE_folder)

    FQE_normal_result_folder = "FQE_returned_normal"
    FQE_normal_path = os.path.join(FQE_folder, FQE_normal_result_folder)

    FQE_normal_dictionary = load_from_pkl(FQE_normal_path)

    policy_normal_model = 'policy_returned_normal'
    policy_normal_path = os.path.join(policy_returned_result_folder,policy_normal_model)
    policy_normal_dictionary = load_from_pkl(policy_normal_path)


    true_list = []
    prediction_list = []
    for policy_key in policy_normal_dictionary:
        policy_file_name = policy_key
        for FQE_step in FQE_saving_step_list:
            FQE_model_pre = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer) + '_'+str(FQE_step) + "step"+"_"
            FQE_model_name = FQE_model_pre + policy_file_name
            true_list.append(policy_normal_dictionary[policy_file_name])
            prediction_list.append(FQE_normal_dictionary[FQE_model_name])
    NMSE,standard_error = normalized_mean_suqare_error(true_list,prediction_list)

    return NMSE, standard_error

def get_Bvft_FQE_name(saved_name):
    Bvft_saving_folder = "Bvft_saving_place"
    Bvft_Q_saving_folder = "Bvft_Q_saving_place"
    Bvft_Q_saving_path = os.path.join(Bvft_saving_folder,Bvft_Q_saving_folder)
    if not os.path.exists(Bvft_Q_saving_path):
        os.makedirs(Bvft_Q_saving_path)
    Bvft_Q_result_saving_path = os.path.join(Bvft_Q_saving_path, saved_name)
    result = load_from_pkl(Bvft_Q_result_saving_path)
    return result[0]


def run_Bvft_evaluation(FQE_saving_step_list):
    print("PlotBvft MSE")

    whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
    train_episodes = whole_dataset.episodes[0:2000]
    test_episodes = whole_dataset.episodes[2000:2276]
    buffer = FIFOBuffer(limit=500000)
    replay_buffer = ReplayBuffer(buffer=buffer, episodes=train_episodes)

    policy_returned_result_folder = "policy_returned_result"
    if not os.path.exists(policy_returned_result_folder):
        os.makedirs(policy_returned_result_folder)

    FQE_returned_folder = "FQE_returned_result"
    FQE_directory = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer)
    FQE_folder = os.path.join(FQE_returned_folder,FQE_directory)
    if not os.path.exists(FQE_folder):
        os.makedirs(FQE_folder)

    FQE_normal_result_folder = "FQE_returned_normal"
    FQE_normal_path = os.path.join(FQE_folder, FQE_normal_result_folder)

    FQE_normal_dictionary = load_from_pkl(FQE_normal_path)

    policy_normal_model = 'policy_returned_normal'
    policy_normal_path = os.path.join(policy_returned_result_folder,policy_normal_model)
    policy_normal_dictionary = load_from_pkl(policy_normal_path)


    true_list = []
    prediction_list = []
    for policy_key in policy_normal_dictionary:
        policy_file_name = policy_key
        for FQE_step in FQE_saving_step_list:
            FQE_model_name = get_Bvft(policy_file_name+"_"+str(FQE_step))
            true_list.append(policy_normal_dictionary[policy_file_name])
            prediction_list.append(FQE_normal_dictionary[FQE_model_name])
    NMSE,standard_error = normalized_mean_suqare_error(true_list,prediction_list)

    return NMSE, standard_error



def run_FQE_1(FQE_saving_step_list):
    FQE_learning_rate = 1e-4
    FQE_hidden_layer = [128, 256]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return run_FQE_evaluation(device, FQE_learning_rate, FQE_hidden_layer,FQE_saving_step_list)
def run_FQE_2(FQE_saving_step_list):
    FQE_learning_rate = 1e-4
    FQE_hidden_layer = [128, 1024]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return run_FQE_evaluation(device, FQE_learning_rate, FQE_hidden_layer,FQE_saving_step_list)
def run_FQE_3(FQE_saving_step_list):
    FQE_learning_rate = 2e-5
    FQE_hidden_layer = [128, 256]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return run_FQE_evaluation(device, FQE_learning_rate, FQE_hidden_layer,FQE_saving_step_list)
def run_FQE_4(FQE_saving_step_list):
    FQE_learning_rate = 2e-5
    FQE_hidden_layer = [128, 1024]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return run_FQE_evaluation(device, FQE_learning_rate, FQE_hidden_layer,FQE_saving_step_list)

def Draw_MSE_graph(FQE_saving_step_list):
    # while(True):
    FQE_1_MSE,FQE_1_SE= run_FQE_1(FQE_saving_step_list)
    FQE_2_MSE,FQE_2_SE = run_FQE_2(FQE_saving_step_list)
    FQE_3_MSE,FQE_3_SE = run_FQE_3(FQE_saving_step_list)
    FQE_4_MSE,FQE_4_SE = run_FQE_4(FQE_saving_step_list)
    Bvft_MSE,Bvft_SE = run_Bvft_evaluation(FQE_saving_step_list)
    name_list = ["hopper-medium-expert-v0"]
    labels = ["FQE_1e-4_" + "[128,256]_" + str(FQE_total_step),
              "FQE_1e-4_" + "[128,1024]_" + str(FQE_total_step),
              "FQE_2e-5_" + "[128,256]_" + str(FQE_total_step),
              "FQE_2e-5_" + "[128,1024]_" + str(FQE_total_step),
              "Bvft-PE-InitialQ"]

    means = [FQE_1_MSE, FQE_2_MSE, FQE_3_MSE, FQE_4_MSE,Bvft_MSE]
    SE = [FQE_1_SE, FQE_2_SE, FQE_3_SE, FQE_4_SE,Bvft_SE]

    # means = [FQE_1_MSE]
    # SE= [FQE_1_SE]
    # colors = ['blue']
    # print("means : ",means)
    FQE_returned_folder = "Bvft_saving_place"
    Bvft_plot = "Bvft_plot"
    Figure_saving_path = os.path.join(FQE_returned_folder,Bvft_plot)
    #
    colors = ['blue', 'orange', 'green', 'purple',"red"]
    draw_mse_graph(combinations=name_list, means=means,  colors=colors, standard_errors = SE,
                   labels=labels, folder_path=Figure_saving_path, filename="Figure6R NMSE graph")
        # time.sleep(60)
def main():
    tf.disable_v2_behavior()
    parser = argparse.ArgumentParser(description="Plot specific FQE function prediction plot based on learning rate and combination.")
    parser.add_argument("--FQE_saving_step_list", type=int, nargs='+', default=[1000000], help="Number of steps in each episode of FQE")
    args = parser.parse_args()
    Draw_MSE_graph(args.FQE_saving_step_list)

if __name__ == "__main__":
    main()