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
import argparse
from scope_rl.ope import CreateOPEInput
import d3rlpy

from scope_rl.utils import check_array
import copy
import numpy as np
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FC_Q(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_layer_list):
        super(FC_Q, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_layer_list[0])
        self.l2 = nn.Linear(hidden_layer_list[0], hidden_layer_list[1])
        self.l3 = nn.Linear(hidden_layer_list[1], 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

class continuous_FQE:
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_layer_list,
            device,
            discount=0.99,
            optimizer="Adam",
            optimizer_parameters={},
            polyak_target_update=False,
            target_update_frequency=8e3,
            tau=0.005,
    ):

        self.device = device

        # Create Q-network and target Q-network
        self.Q = FC_Q(state_dim, action_dim,hidden_layer_list).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.discount = discount

        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # Number of training iterations
        self.iterations = 0

    def train(self, replay_buffer,  policy, trajectory_number):
        # Sample replay buffer

        state, action, next_state, reward, done = replay_buffer.sample(trajectory_number)

        # Convert to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)

        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        done = done.unsqueeze(-1) if done.dim() == 1 else done
        # Compute the target Q value
        with torch.no_grad():
            next_action = policy.predict(next_state)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            next_action= torch.tensor(next_action, dtype=torch.float32, device=self.device)
            print("1 - dfone : ",1-done)
            print(self.Q_target(next_state, next_action))
            print((1 - done) * self.discount * self.Q_target(next_state, next_action))
            print(((1 - done) * self.discount * self.Q_target(next_state, next_action)).squeeze(-1))
            print(reward)
            print(reward + ((1 - done) * self.discount * self.Q_target(next_state, next_action)).squeeze(-1))
            print(self.Q(state, action))
            print(self.Q(state, action).squeeze(-1))
            sys.exit()
            target_Q = reward + ((1 - done) * self.discount * self.Q_target(next_state, next_action)).squeeze(-1)
        # Get current Q estimate
        current_Q = self.Q(state, action).squeeze(-1)


        Q_loss = F.mse_loss(current_Q, target_Q)
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        self.iterations += 1
        self.maybe_update_target()

        print("current loss : ", Q_loss.item())

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, filename):
        torch.save(self.Q.state_dict(), filename + "_Q")
        # torch.save(self.Q_optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.Q.load_state_dict(torch.load(filename + "_Q"))
        self.Q_target = copy.deepcopy(self.Q)
        # self.Q_optimizer.load_state_dict(torch.load(filename + "_optimizer"))
whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
# Example policy function (replace with actual policy to be evaluated)
train_episodes = whole_dataset.episodes[0:2000]
test_episodes = whole_dataset.episodes[2000:2276]

buffer = FIFOBuffer(limit=1500000)

replay_buffer = ReplayBuffer(buffer=buffer, episodes=train_episodes)
policy_folder = 'policy_trained'
policy_name = "cql_300000_0.001_2_64_250000step.d3"

# Usage Example
state_dim = 11
action_dim = 3
device = "cuda" if torch.cuda.is_available() else "cpu"
policy_path = os.path.join(policy_folder,policy_name)
policy = d3rlpy.load_learnable(policy_path, device=device)
fqe = continuous_FQE(state_dim, action_dim, [128, 256], device=device)

test_data = CustomDataLoader(replay_buffer, batch_size=1000)
# Training loop
num_epochs = 100  # Number of epochs to train
for i in range(2000):
    # Sample from replay buffer and train
    fqe.train(test_data, policy,i)
