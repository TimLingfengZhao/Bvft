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

def run_FQE_plot_performance(FQE_total_step,FQE_episode_step,input_gamma,num_run):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    FQE_learning_rate_list = [1e-4,2e-5]
    # FQE_learning_rate_list = [1e-4]
    FQE_hidden_layer_list = [[128,256],[128,1024]]
    FQE_number_epoch = int(FQE_total_step / FQE_episode_step)
    num_intervel = 1

    policy_returned_result_folder = "policy_returned_result"
    if not os.path.exists(policy_returned_result_folder):
        os.makedirs(policy_returned_result_folder)

    FQE_returned_folder = "FQE_returned_result"
    FQE_total_result_folder = "FQE_returned_total"

    policy_folder = 'policy_trained'
    policy_total_model = 'policy_returned_total'
    policy_total_path = os.path.join(policy_returned_result_folder,policy_total_model)
    policy_total_dictionary = load_from_pkl(policy_total_path)

    policy_epoch_prediction_folder_path = 'policy_FQE_prediction_epoch'
#    policy_epoch_prediction_folder_path = os.path.join(policy_returned_result_folder,policy_epoch_prediction_folder)

    for policy_file_name in policy_total_dictionary:
        overall_predictions_list = []
        overall_predictions_name_list = []
        overall_epoch_list = []
        true_value = []

        for FQE_learning_rate in FQE_learning_rate_list:
            for FQE_hidden_layer in FQE_hidden_layer_list:
                prediction_list = []
                epoch_list = []
                for i in range(FQE_number_epoch):
                    FQE_directory = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer)

                    FQE_folder = os.path.join(FQE_returned_folder, FQE_directory)
                    if not os.path.exists(FQE_folder):
                        os.makedirs(FQE_folder)
                    FQE_total_path = os.path.join(FQE_folder, FQE_total_result_folder)
                    FQE_total_dictionary = load_from_pkl(FQE_total_path)
                    FQE_model_pre = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer) + '_'+str(FQE_episode_step * (i + 1)) + "step"+"_"
                    FQE_model_name = FQE_model_pre + policy_file_name
                    se = False
                    if ((i + 1) % num_intervel == 0):
                        FQE_model_name = FQE_model_name
                        se = True
                    if (se):
                        if (is_key_in_dict(FQE_model_name, FQE_total_dictionary)):
                            prediction_list.append(FQE_total_dictionary[FQE_model_name])
                            epoch_list.append((i+1)*FQE_episode_step)
                if len(epoch_list) > 0:
                    overall_predictions_name_list.append('FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer))
                    overall_epoch_list.append(epoch_list)
                    overall_predictions_list.append(prediction_list)
        if len(overall_predictions_name_list) > 0:
            for i in range(len(overall_predictions_list[0])):
                true_value.append(policy_total_dictionary[policy_file_name])
            overall_predictions_list.append(true_value)
            overall_epoch_list.append(overall_epoch_list[0])
            overall_predictions_name_list.append("True policy expectation")
            picture_name = policy_file_name+"_"+"FQE_prediction_picture"+".png"
            # breakpoint()
            plot_predictions(x_axis_names=overall_epoch_list,predictions=overall_predictions_list,
                         line_names = overall_predictions_name_list,
                         saved_folder_path = policy_epoch_prediction_folder_path,
                         saved_name = picture_name,
                         picture_name=policy_file_name)
def main():
    tf.disable_v2_behavior()
    parser = argparse.ArgumentParser(description="Plot specific FQE function prediction plot based on learning rate and combination.")
    parser.add_argument("--FQE_total_step", type=int, default=600000, help="Total number of steps for FQE training")
    parser.add_argument("--FQE_episode_step", type=int, default=100000, help="Number of steps in one episode for FQE training")
    parser.add_argument("--input_gamma", type=float, default=0.99, help="gamma calculating J pi")
    parser.add_argument("--num_run", type=int, default=100, help="num runs to calculate average policy expectation")
    args = parser.parse_args()
    run_FQE_plot_performance(args.FQE_total_step,args.FQE_episode_step,args.input_gamma,args.num_run)

if __name__ == "__main__":
    main()