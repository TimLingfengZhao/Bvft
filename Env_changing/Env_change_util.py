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
class Hopper_edi(ABC):

    def __init__(self,device,parameter_list,parameter_name_list,algorithm_name_list = ["DDPG","SAC"],policy_total_step=300000,
                 policy_episode_step=10000, policy_saving_number = 5,policy_learning_rate=0.0001,policy_hidden_layer=[64,256],
                 env_name = "Hopper-v4"):
        self.device = device
        self.env_name = env_name
        self.parameter_list = parameter_list
        self.parameter_name_list = parameter_name_list
        self.env_list = []
        self.algorithm_name_list = algorithm_name_list
        self.policy_total_step = policy_total_step
        self.policy_episode_step = policy_episode_step,
        self.policy_saving_number = policy_saving_number
        for parameters in self.parameter_list :
            current_env = gymnasium.make(self.env_name)
            for param_name, param_value in zip(self.parameter_name_list, parameters):
                setattr(current_env.unwrapped.model.opt, param_name, param_value)
            # print(current_env.unwrapped.model.opt)
            self.env_list.append(current_env)
    def create_folder(self,folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    def train_policy(self):
        Policy_operation_folder = "Policy_operation"
        Policy_saving_folder = os.path.join(Policy_operation_folder,"Policy_trained")
        self.create_folder(Policy_saving_folder)
        Policy_checkpoints_folder = os.path.join(Policy_operation_folder,"Policy_checkpoints")
        self.create_folder(Policy_checkpoints_folder)
        while(True):
            for env in self.env_list:
                for parameters in self.parameter_list:
                    policy_folder_name = f"{self.env_name}"
                    for i in range(len(parameters)):
                        param_name = self.parameter_name_list[i]
                        param_value = parameters[i].tolist()
                        policy_folder_name += f"_{param_name}_{str(param_value)}"
                    print(f"start training {policy_folder_name} with algorithm {str(self.algorithm_name_list)}")
                    policy_saving_path = os.path.join(Policy_saving_folder, policy_folder_name)
                    policy_checkpoints_path = os.path.join(Policy_checkpoints_folder, policy_folder_name)
                    self.create_folder(policy_saving_path)
                    self.create_folder(policy_checkpoints_path)
                    for algorithm_name in self.algorithm_name_list:














