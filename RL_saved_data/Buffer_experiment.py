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
import time
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device : ",device)
def load_checkpoint_policy(model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print("check point path: ,", checkpoint_path)
        return True
    return False

# def load_checkpoint_FQE(model, checkpoint_path,dataset):
#     if os.path.exists(checkpoint_path):
#         model.build_with_dataset(dataset)
#         model.load_model(FQE_checkpoint_path,device = device)
#         return True
#     return False

def save_list(list,path):
    with open(path,'wb') as file:
        pickle.dump(list,file)
def read_list(list,path):
    with open(path,'rb') as file:
        list = pickle.load(file)
        print("lsit : ",list)
    return list
random_state = 12345
FQE_episode_step = 10000
FQE_learning_rate = 1e-4
FQE_hidden_layer = [128,1024]
FQE_total_step = 500000
FQE_number_epoch = FQE_total_step / FQE_episode_step

policy_hidden_size = [64, 1024]
policy_hidden_layer = [2,3]
policy_learning_rate = [0.001,0.00001]
policy_learning_steps = [300000]
# policy_algorithm_name =  ["bcq","cql"]
policy_algorithm_name =  ["cql"]
policy_episode_step = 10000
daytime = datetime.now()


replay_buffer_limit = 500000
checkpoint_interval = 1

buffer = d3rlpy.dataset.FIFOBuffer(limit=replay_buffer_limit)
transition_picker = d3rlpy.dataset.BasicTransitionPicker()
trajectory_slicer = d3rlpy.dataset.BasicTrajectorySlicer()
writer_preprocessor = d3rlpy.dataset.BasicWriterPreprocess()

whole_dataset, env = get_d4rl('hopper-medium-expert-v0')

train_episodes = whole_dataset.episodes[0:2000]
test_episodes = whole_dataset.episodes[2000:2276]

buffer = FIFOBuffer(limit=500)
dataset = ReplayBuffer(buffer=buffer, episodes=train_episodes)

sampled_traj = dataset.sample_trajectory(1)
print("sampled traj : ",sample_traj)
