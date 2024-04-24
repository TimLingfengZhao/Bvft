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

def get_Bvft_FQE_name(FQE_saving_step_list):
    Bvft_saving_folder = "Bvft_saving_place"
    Bvft_Q_saving_folder = "Bvft_Q_saving_place"
    Bvft_Q_saving_path = os.path.join(Bvft_saving_folder,Bvft_Q_saving_folder)
    if not os.path.exists(Bvft_Q_saving_path):
        os.makedirs(Bvft_Q_saving_path)
    Bvft_Q_result_saving_path = os.path.join(Bvft_Q_saving_path, str(FQE_saving_step_list))
    result = load_from_pkl(Bvft_Q_result_saving_path)
    return result[0]


def extract_substrings(s):
    parts = s.split('_')

    if len(parts) < 4:
        return None, None

    return parts[1], parts[2]
def run_FQE_evaluation(device,FQE_learning_rate,FQE_hidden_layer,FQE_saving_step_list,Bvft=False):
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

    policy_total_model = 'policy_returned_total'
    policy_total_path = os.path.join(policy_returned_result_folder, policy_total_model)
    policy_total_dictionary = load_from_pkl(policy_total_path)

    true_list = []
    prediction_list = []
    max_step = max(FQE_saving_step_list)
    policy_folder = "policy_trained"
    for policy_file_name in os.listdir("policy_trained"):
        if not Bvft:
            FQE_directory = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer)
            FQE_folder = os.path.join(FQE_returned_folder, FQE_directory)
            if not os.path.exists(FQE_folder):
                os.makedirs(FQE_folder)

            FQE_total_result_folder = "FQE_returned_total"
            FQE_total_path = os.path.join(FQE_folder, FQE_total_result_folder)

            FQE_total_dictionary = load_from_pkl(FQE_total_path)



            FQE_model_pre = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer) + '_'+str(max_step) + "step"+"_"
            FQE_model_name = FQE_model_pre + policy_file_name
            true_list.append(policy_total_dictionary[policy_file_name])
            prediction_list.append(FQE_total_dictionary[FQE_model_name])
        else:
            print("Bvft ")

            FQE_model_name = get_Bvft_FQE_name(policy_file_name + "_" + str(FQE_saving_step_list))

            Bvft_FQE_learning_rate, Bvft_FQE_hidden_layer = extract_substrings(FQE_model_name)

            FQE_directory = 'FQE_' + str(Bvft_FQE_learning_rate) + '_' + str(Bvft_FQE_hidden_layer)
            FQE_folder = os.path.join(FQE_returned_folder, FQE_directory)
            if not os.path.exists(FQE_folder):
                os.makedirs(FQE_folder)

            FQE_total_result_folder = "FQE_returned_total"
            FQE_total_path = os.path.join(FQE_folder, FQE_total_result_folder)

            FQE_total_dictionary = load_from_pkl(FQE_total_path)

            if policy_file_name in policy_total_dictionary:
                true_list.append(policy_total_dictionary[policy_file_name])
            else:
                policy_path = os.path.join(policy_folder, policy_file_name)
                d3rlpy.load_learnable(policy_path, device=device)
                true_list.append(calculate_policy_value(env,policy))
            prediction_list.append(FQE_total_dictionary[FQE_model_name])
    NMSE,standard_error = normalized_mean_square_error_with_error_bar(true_list,prediction_list)

    return NMSE, standard_error



def run_FQE_1(device,FQE_saving_step_list):
    FQE_learning_rate = 1e-4
    FQE_hidden_layer = [128, 256]

    return run_FQE_evaluation(device, FQE_learning_rate, FQE_hidden_layer,FQE_saving_step_list,Bvft=False)
def run_FQE_2(device,FQE_saving_step_list):
    FQE_learning_rate = 1e-4
    FQE_hidden_layer = [128, 1024]

    return run_FQE_evaluation(device, FQE_learning_rate, FQE_hidden_layer,FQE_saving_step_list,Bvft=False)
def run_FQE_3(device,FQE_saving_step_list):
    FQE_learning_rate = 2e-5
    FQE_hidden_layer = [128, 256]

    return run_FQE_evaluation(device, FQE_learning_rate, FQE_hidden_layer,FQE_saving_step_list,Bvft=False)
def run_FQE_4(device,FQE_saving_step_list):
    FQE_learning_rate = 2e-5
    FQE_hidden_layer = [128, 1024]

    return run_FQE_evaluation(device, FQE_learning_rate, FQE_hidden_layer,FQE_saving_step_list,Bvft=False)

def run_Bvft(device,FQE_saving_step_list):
    FQE_learning_rate = 2e-5
    FQE_hidden_layer = [128, 1024]

    return run_FQE_evaluation(device, FQE_learning_rate, FQE_hidden_layer,FQE_saving_step_list,Bvft=True)
def Draw_MSE_graph(FQE_saving_step_list):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # while(True):
    FQE_1_MSE,FQE_1_SE= run_FQE_1(device,FQE_saving_step_list)
    FQE_2_MSE,FQE_2_SE = run_FQE_2(device,FQE_saving_step_list)
    FQE_3_MSE,FQE_3_SE = run_FQE_3(device,FQE_saving_step_list)
    FQE_4_MSE,FQE_4_SE = run_FQE_4(device,FQE_saving_step_list)
    Bvft_MSE,Bvft_SE = run_Bvft(device,FQE_saving_step_list)
    name_list = ["hopper-medium-expert-v0"]
    labels = ["FQE_1e-4_" + "[128,256]_" ,
              "FQE_1e-4_" + "[128,1024]_" ,
              "FQE_2e-5_" + "[128,256]_" ,
              "FQE_2e-5_" + "[128,1024]_" ,
              "Bvft-PE-InitialQ"]

    means = [FQE_1_MSE, FQE_2_MSE, FQE_3_MSE, FQE_4_MSE,Bvft_MSE]
    SE = [FQE_1_SE, FQE_2_SE, FQE_3_SE, FQE_4_SE,Bvft_SE]

    # means = [FQE_1_MSE]
    # SE= [FQE_1_SE]
    # colors = ['blue']git p
    # print("means : ",means)
    FQE_returned_folder = "Bvft_saving_place"
    Bvft_plot = "Bvft_plot"
    Figure_saving_path = os.path.join(FQE_returned_folder,Bvft_plot)
    #
    colors = ['blue', 'orange', 'green', 'purple',"red"]
    draw_mse_graph(combinations=name_list, means=means,  colors=colors, standard_errors = SE,
                   labels=labels, folder_path=Figure_saving_path, filename="Figure6R_NMSE_graph")
        # time.sleep(60)
def main():
    # tf.disable_v2_behavior()
    parser = argparse.ArgumentParser(description="Plot specific FQE function prediction plot based on learning rate and combination.")
    parser.add_argument("--FQE_saving_step_list", type=int, nargs='+', default=[1000000], help="Number of steps in each episode of FQE")
    args = parser.parse_args()
    Draw_MSE_graph(args.FQE_saving_step_list)
#python Bvft_figure_6R_draw.py --FQE_saving_step_list 900000
if __name__ == "__main__":
    main()