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

    def __init__(self,device,parameter_list,parameter_name_list,policy_training_parameter_map,
                 env_name = "Hopper-v4"):
        self.device = device
        self.env_name = env_name
        self.parameter_list = parameter_list
        self.parameter_name_list = parameter_name_list
        self.env_list = []
        self.policy_total_step = policy_training_parameter_map["policy_total_step"]
        self.policy_episode_step = policy_training_parameter_map["policy_episode_step"]
        self.policy_saving_number = policy_training_parameter_map["policy_saving_number"]
        self.policy_learning_rate = policy_training_parameter_map["policy_learning_rate"]
        self.policy_hidden_layer = policy_training_parameter_map["policy_hidden_layer"]
        self.algorithm_name_list = policy_training_parameter_map["algorithm_name_list"]
        for i in range(len(self.parameter_list)):
            current_env = gymnasium.make(self.env_name)
            for param_name, param_value in zip(self.parameter_name_list, self.parameter_list[i]):
                setattr(current_env.unwrapped.model.opt, param_name, param_value)
            # print(current_env.unwrapped.model.opt)
            self.env_list.append(current_env)
        self.para_map = {index: item for index, item in enumerate(self.parameter_list)}
    def create_folder(self,folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    def save_as_pkl(self,file_path, list_to_save):
        full_path = f"{file_path}.pkl"
        with open(full_path, 'wb') as file:
            pickle.dump(list_to_save, file)

    def save_as_txt(self,file_path, list_to_save):
        full_path = f"{file_path}.txt"
        with open(full_path, 'w') as file:
            for item in list_to_save:
                file.write(f"{item}\n")

    def save_dict_as_txt(self,file_path, dict_to_save):
        full_path = f"{file_path}.txt"
        with open(full_path, 'w') as file:
            for key, value in dict_to_save.items():
                file.write(f"{key}:{value}\n")

    def load_dict_from_txt(self,file_path):
        with open(file_path, 'r') as file:
            return {line.split(':', 1)[0]: line.split(':', 1)[1].strip() for line in file}

    def list_to_dict(self,name_list, reward_list):
        return dict(zip(name_list, reward_list))

    def load_from_pkl(self,file_path):
        full_path = f"{file_path}.pkl"
        with open(full_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def whether_file_exists(self,checkpoint_path):
        if not os.path.exists(checkpoint_path):
            return False
        return True
    def generate_one_trajectory(self,env_number,max_time_step,algorithm_name):
        Policy_operation_folder = "Policy_operation"
        Policy_saving_folder = os.path.join(Policy_operation_folder,"Policy_trained")
        self.create_folder(Policy_saving_folder)
        policy_folder_name = f"{self.env_name}"
        for j in range(len(self.parameter_list[env_number])):
            param_name = self.parameter_name_list[j]
            param_value = self.parameter_list[env_number][j].tolist()
            policy_folder_name += f"_{param_name}_{str(param_value)}"
        policy_saving_path = os.path.join(Policy_saving_folder, policy_folder_name)
        policy_model_name = f"{algorithm_name}_{str(self.policy_total_step)}_{str(self.policy_learning_rate)}_{str(self.policy_hidden_layer)}_{self.policy_total_step}step.d3"
        policy_path = os.path.join(policy_saving_path, policy_model_name)
        policy = d3rlpy.load_learnable(policy_path, device=self.device)
        env = self.env_list[env_number]
        obs,info = env.reset(seed=12345)

        observations = []
        rewards = []
        dones = []
        next_steps = []
        episode_data = {}
        observations.append(obs)
        for t in range(max_time_step):

            action = policy.predict(np.array([obs]))

            state, reward, done, truncated, info = env.step(action[0])
            rewards.append(reward)
            dones.append(done)
            next_steps.append(state)
            if((t != max_time_step-1) and done == False):
                observations.append(state)

            obs = state

            if done or truncated:
                break

        episode_data["state"] = observations
        episode_data["rewards"] = rewards
        episode_data["done"] = dones
        episode_data["next_state"] = next_steps
        print(len(observations))
        print(len(rewards))
        print(len(rewards))
        print(len(next_steps))
        return episode_data

    def generate_offline_data(self,trajectory_numbers,max_time_step,algorithm_name):
        self.print_environment_parameters()
        true_env_number = int(input("Please enter the environment parameter number you choose: "))

        final_data = []
        for i in range(trajectory_numbers):
            one_episode_data = self.generate_one_trajectory(true_env_number,max_time_step,algorithm_name)
            final_data.append(one_episode_data)


    def print_environment_parameters(self):
        print(f"{self.parameter_name_list} parameters of environments in current class :")
        for key, value in self.para_map.items():
            print(f"{key}: {value}")

    def train_policy(self):
        Policy_operation_folder = "Policy_operation"
        Policy_saving_folder = os.path.join(Policy_operation_folder,"Policy_trained")
        self.create_folder(Policy_saving_folder)
        Policy_checkpoints_folder = os.path.join(Policy_operation_folder,"Policy_checkpoints")
        self.create_folder(Policy_checkpoints_folder)
        while(True):
            for i in range(len(self.parameter_list)):
                current_env = self.env_list[i]
                policy_folder_name = f"{self.env_name}"
                for j in range(len(self.parameter_list[i])):
                    param_name = self.parameter_name_list[j]
                    param_value = self.parameter_list[i][j].tolist()
                    policy_folder_name += f"_{param_name}_{str(param_value)}"
                print(f"start training {policy_folder_name} with algorithm {str(self.algorithm_name_list)}")
                policy_saving_path = os.path.join(Policy_saving_folder, policy_folder_name)
                policy_checkpoints_path = os.path.join(Policy_checkpoints_folder, policy_folder_name)
                self.create_folder(policy_saving_path)
                self.create_folder(policy_checkpoints_path)
                num_epoch = int(self.policy_total_step / self.policy_episode_step)
                buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=1000000, env=current_env)
                explorer = d3rlpy.algos.ConstantEpsilonGreedy(0.3)
                checkpoint_list = []
                for algorithm_name in self.algorithm_name_list:
                    checkpoint_path = os.path.join(policy_checkpoints_path,f"{algorithm_name}_checkpoints.d3")
                    checkpoint_list_path = os.path.join(policy_checkpoints_path, f"{algorithm_name}_checkpoints")
                    policy_model_name = f"{algorithm_name}_{str(self.policy_total_step)}_{str(self.policy_learning_rate)}_{str(self.policy_hidden_layer)}.d3"
                    policy_path = os.path.join(policy_saving_path, policy_model_name)
                    if(not self.whether_file_exists(policy_path[:-3]+"_"+str(self.policy_total_step)+"step.d3")):
                        print(f"{policy_path} not exists")
                        if (self.whether_file_exists(checkpoint_path)):
                            policy = d3rlpy.load_learnable(checkpoint_path, device=self.device)
                            checkpoint_list = self.load_from_pkl(checkpoint_list_path)
                            print(f"enter self checkpoints {checkpoint_path} with epoch {str(checkpoint_list[-1])}")
                            for epoch in range(checkpoint_list[-1] + 1, int(num_epoch)):
                                policy.fit_online(env=current_env,
                                                  buffer=buffer,
                                                  explorer=explorer,
                                                  n_steps=self.policy_episode_step,
                                                  eval_env=current_env,
                                                  with_timestamp=False,
                                                  )
                                policy.save(checkpoint_path)
                                checkpoint_list.append(epoch)
                                self.save_as_pkl(checkpoint_list_path, checkpoint_list)
                                if ((epoch + 1) % self.policy_saving_number == 0):
                                    policy.save(policy_path[:-3] + "_" + str(
                                        (epoch + 1) * self.policy_episode_step) + "step" + ".d3")
                        else:
                            self_class = getattr(d3rlpy.algos, algorithm_name + "Config")
                            policy = self_class(
                                actor_encoder_factory=d3rlpy.models.VectorEncoderFactory(
                                    hidden_units=self.policy_hidden_layer),
                                critic_encoder_factory=d3rlpy.models.VectorEncoderFactory(
                                    hidden_units=self.policy_hidden_layer),
                                actor_learning_rate=self.policy_learning_rate,
                                critic_learning_rate=self.policy_learning_rate,
                            ).create(device=self.device)
                            for epoch in range(num_epoch):
                                policy.fit_online(env=current_env,
                                                  buffer=buffer,
                                                  explorer=explorer,
                                                  n_steps=self.policy_episode_step,
                                                  eval_env=current_env,
                                                  with_timestamp=False,
                                                  )
                                policy.save(checkpoint_path)
                                checkpoint_list.append(epoch)
                                self.save_as_pkl(checkpoint_list_path, checkpoint_list)
                                if ((epoch + 1) % self.policy_saving_number == 0):
                                    policy.save(policy_path[:-3] + "_" + str(
                                        (epoch + 1) * self.policy_episode_step) + "step" + ".d3")
                        if os.path.exists(checkpoint_list_path + ".pkl"):
                            os.remove(checkpoint_list_path + ".pkl")
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                        print(f"end training {policy_folder_name} with algorithm {str(self.algorithm_name_list)}")
            print("sleep now")
            time.sleep(600)

