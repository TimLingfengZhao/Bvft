from abc import ABC, abstractmethod
import numpy as np
import sys
import os
import pickle
from multiprocessing import Process, Pool
import heapq
from typing import Sequence
from d3rlpy.datasets import get_d4rl
import gym
import torch.multiprocessing as mp
import multiprocessing
from multiprocessing import shared_memory
import random
import copy
import concurrent.futures
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
# from top_k_cal import *
from scope_rl.ope.continuous import DirectMethod as DM
from scope_rl.policy import ContinuousEvalHead
from d3rlpy.algos import DDPGConfig
from d3rlpy.algos import SACConfig
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
import torch
import torch.nn as nn
from scope_rl.ope.estimators_base import BaseOffPolicyEstimator
from d3rlpy.dataset import Episode
from top_k_cal import *
from abc import ABC, abstractmethod
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
import gymnasium
import logging
data_folder = "Offline_data"
data_path = os.path.join(data_folder,"DDPG_Hopper-v4_gravity_[0.0, 0.0, -4.9]_magnetic_[0.0, 0.0, 0.0]_wind_[10.0, 0.0, 0.0]_1000_maxStep_200_trajectory_0")

class Bvft_:
    def __init__(self, max_timestep=1000, gamma=0.99):
        self.max_timestep = max_timestep
        self.gamma = gamma

    def run_simulation(self, state_action_policy_env):
        state, action, policy, env = state_action_policy_env
        total_rewards = 0
        num_step = 0
        discount_factor = 1
        observation = state
        env.reset()
        env.observation = observation
        ui = env.step(action)
        state = ui[0]
        reward = ui[1]
        total_rewards += reward
        done = ui[2]

        start_time = time.time()  # Start timing
        while not done and num_step < self.max_timestep:
            action = policy.predict(np.array([state]))
            ui = env.step(action[0])
            state = ui[0]
            reward = ui[1]
            done = ui[2]
            total_rewards += reward * discount_factor
            discount_factor *= self.gamma
            num_step += 1
        end_time = time.time()  # End timing

        total_time = end_time - start_time
        total_steps = num_step

        return total_rewards, total_time, total_steps

class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, states):
        # Generate random actions for each state in the batch
        return np.array([self.action_space.sample() for _ in range(len(states))])

def evaluate_policy_on_dataset(data, policy, env):
    hopper_exp = Bvft_(max_timestep=1000, gamma=0.99)
    total_times = []
    total_steps_list = []
    total_rewards_list = []

    for trajectory in data:
        for initial_state in trajectory["state"]:
            initial_action = policy.predict(np.array([initial_state]))[0]
            state_action_policy_env = (initial_state, initial_action, policy, env)
            total_rewards, total_time, total_steps = hopper_exp.run_simulation(state_action_policy_env)
            total_rewards_list.append(total_rewards)
            total_times.append(total_time)
            total_steps_list.append(total_steps)

    return total_rewards_list, total_times, total_steps_list

# Initialize the environment and policy
device = "cuda:0" if torch.cuda.is_available() else "cpu"
env = gymnasium.make("Hopper-v4")

# Load the offline data
offline_data = "Offline_data"
offlin = os.path.join(offline_data, "DDPG_Hopper-v4_gravity_[0.0, 0.0, -4.9]_magnetic_[0.0, 0.0, 0.0]_wind_[10.0, 0.0, 0.0]_100_maxStep_10_trajectory_0")
data = load_from_pkl(offlin).dataset

# Load the DDPG policy
policy_op = "Policy_operation"
policy_trained = "Policy_trained"
policy_fo = os.path.join(policy_op, policy_trained)
policy_folder = os.path.join(policy_fo, "Hopper-v4_gravity_[0.0, 0.0, -4.9]_magnetic_[0.0, 0.0, 0.0]_wind_[10.0, 0.0, 0.0]")
policy_na = os.path.join(policy_folder, "DDPG_300000_0.0001_[64, 256]_300000step.d3")
ddpg_policy = d3rlpy.load_learnable(policy_na, device=device)

# Initialize the random policy
random_policy = RandomPolicy(env.action_space)

# Evaluate the DDPG policy on the dataset
ddpg_total_rewards_list, ddpg_total_times, ddpg_total_steps_list = evaluate_policy_on_dataset(data, ddpg_policy, env)
print("DDPG Policy Evaluation:")
for i, (rewards, time_used, steps) in enumerate(zip(ddpg_total_rewards_list, ddpg_total_times, ddpg_total_steps_list)):
    print(f"Evaluation {i+1}: Total rewards = {rewards}, Total time used = {time_used:.4f} seconds, Total steps = {steps}")

# Evaluate the random policy on the dataset
random_total_rewards_list, random_total_times, random_total_steps_list = evaluate_policy_on_dataset(data, random_policy, env)
print("\nRandom Policy Evaluation:")
for i, (rewards, time_used, steps) in enumerate(zip(random_total_rewards_list, random_total_times, random_total_steps_list)):
    print(f"Evaluation {i+1}: Total rewards = {rewards}, Total time used = {time_used:.4f} seconds, Total steps = {steps}")

# Calculate the average speed difference
ddpg_total_t = np.sum(ddpg_total_times)
random_total_t = np.sum(random_total_times)


ddpg_avg_time = np.mean(ddpg_total_times)
random_avg_time = np.mean(random_total_times)
speed_difference = ddpg_avg_time / random_avg_time
print(f"\nAverage Speed Difference (DDPG vs Random): {speed_difference:.4f} times faster")
print(f"ddpg total time : {ddpg_total_t } and rnadom total time : {random_total_t}")

ddpg_avg_step = np.mean(ddpg_total_steps_list)
random_avg_step = np.mean(random_total_steps_list)
print(f"ddpg avg step : {ddpg_avg_step } and rnadom avg step: {random_avg_step}")

ddpg_total_step = np.sum(ddpg_total_steps_list)
random_total_step = np.sum(random_total_steps_list)
print(f"ddpg avg step : {ddpg_total_step } and rnadom avg step: {random_total_step}")