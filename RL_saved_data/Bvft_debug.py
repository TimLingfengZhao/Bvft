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
def run_Debug_graph(device,FQE_saving_step_list):
    FQE_learning_rate_list = [1e-4, 2e-5]
    FQE_hidden_layer_list = [[128, 256],[128, 1024]]
    whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
    train_episodes = whole_dataset.episodes[0:2000]
    test_episodes = whole_dataset.episodes[2000:2276]
    buffer = FIFOBuffer(limit=500000)
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
    name_list = []
    max_step = max(FQE_saving_step_list)
    for policy_file_name in os.listdir("policy_trained"):
        Performance_list = []
        policy_name = policy_file_name[:-3]
        Performance_list.append(policy_total_dictionary[policy_name])
        name_list.append(policy_name)
        for FQE_learning_rate in FQE_learning_rate_list:
            for FQE_hidden_layer in FQE_hidden_layer_list:
                FQE_directory = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer)
                FQE_folder = os.path.join(FQE_returned_folder, FQE_directory)
                if not os.path.exists(FQE_folder):
                    os.makedirs(FQE_folder)

                FQE_total_result_folder = "FQE_returned_total"
                FQE_total_path = os.path.join(FQE_folder, FQE_total_result_folder)

                FQE_model_pre = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer) + '_'+str(max_step) + "step"+"_"
                FQE_model_name = FQE_model_pre + policy_name

                FQE_total_dictionary = load_from_pkl(FQE_total_path)
                Prediction_list.append(FQE_total_dictionary[FQE_model_name])
                if (FQE_learning_rate == 2e-4 and FQE_hidden_layer == [128, 1024]):
                    FQE_model_name_bvft = get_Bvft_FQE_name(policy_name + "_" + str(FQE_saving_step_list))
                    Bvft_FQE_learning_rate, Bvft_FQE_hidden_layer = extract_substrings(FQE_model_name_bvft)
                    FQE_directory = 'FQE_' + str(Bvft_FQE_learning_rate) + '_' + str(Bvft_FQE_hidden_layer)
                    Prediction_list.append(FQE_total_dictionary[FQE_model_name_bvft])

        prediction_list.append(Performance_list)
        name_list.append(Name_list)
    plot_performance_list, plot_name_list = sort_lists_by_first_dec(performance_list, name_list)


    return plot_performance_list, plot_name_list



def run_Debug_graph(device,FQE_saving_step_list):

    return run_FQE_evaluation(device, FQE_learning_rate, FQE_hidden_layer,FQE_saving_step_list,Bvft=False)
def Draw_MSE_graph(FQE_saving_step_list):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # while(True):
    plot_performance_list, plot_name_list= run_Debug_graph(device,FQE_saving_step_list)
    line_name_list = ["Policy_performance","FQE_1e-4_256","FQE_1e-4_1024","FQE_2e-5_256","FQE_2e-5_1024","Bvft"]
    draw_debug_graph(plot_performance_list,plot_name_list,line_name_list)
        # time.sleep(60)
def main():
    # tf.disable_v2_behavior()
    parser = argparse.ArgumentParser(description="Plot specific FQE function prediction plot based on learning rate and combination.")
    parser.add_argument("--FQE_saving_step_list", type=int, nargs='+', default=[2000000], help="Number of steps in each episode of FQE")
    args = parser.parse_args()
    Draw_MSE_graph(args.FQE_saving_step_list)
#python Bvft_figure_6R_draw.py --FQE_saving_step_list 900000
if __name__ == "__main__":
    main()