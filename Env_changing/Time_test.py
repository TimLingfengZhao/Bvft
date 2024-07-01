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
# from top_k_cal import *
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
# from BvftUtil import *
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

policy_hidden_size = [1024]
policy_hidden_layer = [3]
policy_learning_rate = [0.001]
policy_learning_steps = [500]
# policy_algorithm_name =  ["bcq","cql"]
policy_algorithm_name =  ["bcq"]
# policy_algorithm_name =  ["bcq"]
policy_episode_step = 100
daytime = datetime.now()


replay_buffer_limit = 1500000
checkpoint_interval = 1


# import tensorflow.compat.v1 as tf

buffer = d3rlpy.dataset.FIFOBuffer(limit=replay_buffer_limit)
transition_picker = d3rlpy.dataset.BasicTransitionPicker()
trajectory_slicer = d3rlpy.dataset.BasicTrajectorySlicer()
writer_preprocessor = d3rlpy.dataset.BasicWriterPreprocess()

whole_dataset, env = get_d4rl('hopper-medium-v2')

train_episodes = whole_dataset.episodes[0:1500]
test_episodes = whole_dataset.episodes[1500:2186]

buffer = FIFOBuffer(limit=1500000)

dataset = ReplayBuffer(buffer=buffer, episodes=train_episodes)
policy_list = []
for policy_hidden in policy_hidden_size :
   for policy_layer in policy_hidden_layer:
      for policy_rate in policy_learning_rate:
         for policy_step in policy_learning_steps:
            for policy_name in policy_algorithm_name:
                policy_hidden_list = []
                if policy_layer == 2:
                    policy_hidden_list = [policy_hidden, policy_hidden]
                else:
                    policy_hidden_list = [policy_hidden, policy_hidden, policy_hidden]
                bcq = BCQConfig(
                    actor_encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=policy_hidden_list),
                    critic_encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=policy_hidden_list),
                    actor_learning_rate=policy_rate,
                    critic_learning_rate=policy_rate,
                ).create(device=device)
                num_epoch = policy_step / policy_episode_step

                policy_list = []
                for epoch in range(int(num_epoch)):
                    bcq.fit(dataset,
                            n_steps=policy_episode_step,
                            with_timestamp=False,)
                    policy_list.append(bcq)
import random

def generate_2d_list(rows=1000, cols=11):

    return [[random.uniform(0, 100) for _ in range(cols)] for _ in range(rows)]
def random_list(n: int) -> list:
    return [[random.random() for _ in range(3)] for _ in range(n)]
one_dim = [[0.1,0.2,0.3,0.4,0.5,0.6,0.6,0.6,0.6,0.6,0.6]]

generated_list = generate_2d_list()
state = generated_list
time_list = []
time_list_one = []


def calculate_time_cost(input_number, policy):
    times = []
    batch_size = 2
    total_data = 2**4

    for i in range(input_number):
        batches = total_data // batch_size
        time_cost = 0
        for j in range(batches):
            start = time.time()
            policy.predict(np.array(generate_2d_list(rows=2**j)))
            end = time.time()
            time_cost += end-start
        times.append(time_cost)
        batch_size *= 2
    return times

policy = policy_list[0]
print(calculate_time_cost(3,policy))

def calculate_time_cost_ran(input_number):
    times = []
    batch_size = 2
    total_data = 2**4

    for i in range(input_number):
        batches = total_data // batch_size
        time_cost = 0
        for j in range(batches):
            start = time.time()
            random_list(2**j)
            end = time.time()
            time_cost += end-start
        times.append(time_cost)
        batch_size *= 2
    return times
print(calculate_time_cost_ran(3))
def calculate_time_cost(total_data: int, policy, n: int):
    # Function to calculate time cost for a given batch size
    def time_cost_for_batch_size(batch_size: int) -> float:
        batches = total_data // batch_size
        remaining = total_data % batch_size
        time_cost = 0

        for _ in range(batches):
            start = time.time()
            policy.predict(np.array(generate_2d_list(rows=batch_size)))
            end = time.time()
            time_cost += end - start

        if remaining > 0:
            start = time.time()
            policy.predict(np.array(generate_2d_list(rows=remaining)))
            end = time.time()
            time_cost += end - start

        return time_cost

    time_cost_128 = time_cost_for_batch_size(128)

    time_cost_n = time_cost_for_batch_size(n)

    return time_cost_128, time_cost_n
list_one, list_two = calculate_time_cost(10000,policy,100)
list_three, list_four = calculate_time_cost(10000,policy,150)
print(list_one)
print(list_two)
print(list_three)