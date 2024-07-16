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
offlin = os.path.join(offline_data, "DDPG_Hopper-v4_gravity_[0.0, 0.0, -4.9]_magnetic_[0.0, 0.0, 0.0]_wind_[10.0, 0.0, 0.0]_1000_maxStep_24_trajectory_0")
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
print(f"ddpg total step : {ddpg_total_step } and rnadom total step: {random_total_step}")

ddpg_max_step = np.max(ddpg_total_steps_list)
random_max_step = np.max(random_total_steps_list)
print(f"ddpg max step : {ddpg_max_step } and rnadom max step: {random_max_step}")


ddpg_min_step = np.min(ddpg_total_steps_list)
random_min_step = np.min(random_total_steps_list)
print(f"ddpg min step : {ddpg_min_step } and rnadom min step: {random_min_step}")

ddpg_len_step = len(ddpg_total_steps_list)
random_len_step = len(random_total_steps_list)
print(f"ddpg shape : {ddpg_len_step } and rnadom shape: {random_len_step}")
# def run_simulation(self, state_action_policy_env_batch):
#     states, actions, policy, envs = state_action_policy_env_batch
#
#     # Initialize environments with states
#     for env, state in zip(envs, states):
#         env.reset()
#         env.observation = state
#
#     total_rewards = np.zeros(len(states))
#     num_steps = np.zeros(len(states))
#     discount_factors = np.ones(len(states))
#     done_flags = np.zeros(len(states), dtype=bool)
#
#     while not np.all(done_flags) and np.any(num_steps < self.max_timestep):
#         actions_batch = policy.predict(np.array(states))
#         for idx, (env, action) in enumerate(zip(envs, actions_batch)):
#             if not done_flags[idx]:
#                 next_state, reward, done, _, _ = env.step(action)
#                 total_rewards[idx] += reward * discount_factors[idx]
#                 discount_factors[idx] *= self.gamma
#                 num_steps[idx] += 1
#                 states[idx] = next_state
#                 done_flags[idx] = done
#     gc.collect()
#     return total_rewards
#
#
# def get_qa(self, policy_number, env_copy_list, states, actions, batch_size=8):
#     policy = self.policy_list[policy_number]
#     results = []
#
#     total_len = len(states)
#     for i in range(0, total_len, batch_size):
#         actual_batch_size = min(batch_size, total_len - i)
#         state_action_policy_env_pairs = (
#             states[i:i + actual_batch_size], actions[i:i + actual_batch_size], policy,
#             env_copy_list[:actual_batch_size])
#         batch_results = self.run_simulation(state_action_policy_env_pairs)
#         results.extend(batch_results)
#     gc.collect()
#     return results
#
#
# def create_deep_copies(self, env, batch_size):
#     return [copy.deepcopy(env) for _ in range(batch_size)]
#
#
# @profile
# def process_env(self, env_index, policy_index, fin_list_shm_name):
#     start_time = time.time()
#     logging.info(f"Starting processing for env_index {env_index}, policy_index {policy_index}")
#
#     with SharedMemoryManager() as smm:
#         fin_list_shm = shared_memory.SharedMemory(name=fin_list_shm_name)
#         fin_list = dill.loads(fin_list_shm.buf)
#
#         env_copy_shm = shared_memory.SharedMemory(name=fin_list[0])
#         env_copy_lists = dill.loads(env_copy_shm.buf[:fin_list[1]])
#
#         policy_list_shm = shared_memory.SharedMemory(name=fin_list[2])
#         policy_list = dill.loads(policy_list_shm.buf[:fin_list[3]])
#
#         data_shm = shared_memory.SharedMemory(name=fin_list[4])
#         data = dill.loads(data_shm.buf[:fin_list[5]])
#
#         env_list_shm = shared_memory.SharedMemory(name=fin_list[6])
#         env_list = dill.loads(env_list_shm.buf[:fin_list[7]])
#
#         q_sa_shm = shared_memory.SharedMemory(name=fin_list[8])
#         q_sa = dill.loads(q_sa_shm.buf[:fin_list[9]])
#
#         r_plus_vfsp_shm = shared_memory.SharedMemory(name=fin_list[10])
#         r_plus_vfsp = dill.loads(r_plus_vfsp_shm.buf[:fin_list[11]])
#
#         env = env_list[env_index]
#         env_copy_list = env_copy_lists[env_index][policy_index]
#         policy = policy_list[policy_index]
#         ptr = 0
#         trajectory_length = 0
#
#         while ptr < data.length:
#             length = data.get_iter_length(ptr)
#             state, action, next_state, reward, done = data.sample(ptr)
#
#             q_sa[(env_index + 1) * len(policy_list) + (policy_index + 1) - 1][
#             trajectory_length:trajectory_length + length] = self.get_qa(policy_index, env_copy_list, state, action,
#                                                                         self.batch_size)
#             vfsp = (reward + self.get_qa(policy_index, env_copy_list, next_state, policy.predict(next_state),
#                                          self.batch_size) * (1 - np.array(done)) * self.gamma)
#             r_plus_vfsp[(env_index + 1) * (policy_index + 1) - 1][
#             trajectory_length:trajectory_length + length] = vfsp.flatten()[:length]
#             trajectory_length += length
#             ptr += 1
#
#         self.data_size = trajectory_length
#
#         end_time = time.time()
#         logging.info(
#             f"Finished processing for env_index {env_index}, policy_index {policy_index} in {end_time - start_time} seconds")
#
#
# # def process_env(self, env_index, policy_index, env_copy_lists_name, policy_list_name, data_name, env_list_name, q_sa_name, r_plus_vfsp_name, q_sa_shape, r_plus_vfsp_shape):
# #     start_time = time.time()
# #     logging.info(f"Starting processing for env_index {env_index}, policy_index {policy_index}")
# #
# #     # Attach to existing shared memory blocks
# #     shm_env_copy_lists = shared_memory.SharedMemory(name=env_copy_lists_name)
# #     env_copy_lists = dill.loads(bytes(shm_env_copy_lists.buf))
# #
# #     shm_policy_list = shared_memory.SharedMemory(name=policy_list_name)
# #     policy_list = dill.loads(bytes(shm_policy_list.buf))
# #
# #     shm_data = shared_memory.SharedMemory(name=data_name)
# #     data = dill.loads(bytes(shm_data.buf))
# #
# #     shm_env_list = shared_memory.SharedMemory(name=env_list_name)
# #     env_list = dill.loads(bytes(shm_env_list.buf))
# #
# #     # Attach to shared memory for q_sa and r_plus_vfsp
# #     shm_q_sa = shared_memory.SharedMemory(name=q_sa_name)
# #     q_sa = np.ndarray(q_sa_shape, dtype=np.float64, buffer=shm_q_sa.buf)
# #
# #     shm_r_plus_vfsp = shared_memory.SharedMemory(name=r_plus_vfsp_name)
# #     r_plus_vfsp = np.ndarray(r_plus_vfsp_shape, dtype=np.float64, buffer=shm_r_plus_vfsp.buf)
# #
# #     env = env_list[env_index]
# #     env_copy_list = env_copy_lists[env_index][policy_index]
# #     policy = policy_list[policy_index]
# #     ptr = 0
# #     trajectory_length = 0
# #
# #     # Log initial memory usage
# #     logging.info(f"Initial memory usage: {memory_usage()[0]} MB")
# #
# #     while ptr < data.length:
# #         length = data.get_iter_length(ptr)
# #         state, action, next_state, reward, done = data.sample(ptr)
# #
# #         # Log memory usage before get_qa
# #         logging.info(f"Memory usage before get_qa: {memory_usage()[0]} MB")
# #         q_sa[(env_index + 1) * len(policy_list) + (policy_index + 1) - 1][
# #             trajectory_length:trajectory_length + length] = self.get_qa(policy_index, env_copy_list, state, action,
# #                                                                         self.batch_size)
# #         # Log memory usage after get_qa
# #         logging.info(f"Memory usage after get_qa: {memory_usage()[0]} MB")
# #
# #         vfsp = (reward + self.get_qa(policy_index, env_copy_list, next_state,
# #                                      policy.predict(next_state), self.batch_size) * (
# #                         1 - np.array(done)) * self.gamma)
# #         r_plus_vfsp[(env_index + 1) * (policy_index + 1) - 1][
# #             trajectory_length:trajectory_length + length] = vfsp.flatten()[:length]
# #         trajectory_length += length
# #         ptr += 1
# #
# #         # Log memory usage after processing each chunk
# #         logging.info(f"Memory usage after processing chunk: {memory_usage()[0]} MB")
# #
# #         # Manually trigger garbage collection
# #         gc.collect()
# #
# #     self.data_size = trajectory_length
# #
# #     end_time = time.time()
# #     logging.info(
# #         f"Finished processing for env_index {env_index}, policy_index {policy_index} in {end_time - start_time} seconds")
# #
# #     # Log final memory usage
# #     logging.info(f"Final memory usage: {memory_usage()[0]} MB")
# #
# #     # Cleanup shared memory in worker
# #     shm_env_copy_lists.close()
# #     shm_policy_list.close()
# #     shm_data.close()
# #     shm_env_list.close()
# #     shm_q_sa.close()
# #     shm_r_plus_vfsp.close()
# #
# #     # Force garbage collection
# #     gc.collect()
#
# def get_whole_qa(self, algorithm_index):
#     Offline_data_folder = "Offline_data"
#     self.create_folder(Offline_data_folder)
#     data_folder_name = f"{self.algorithm_name_list[algorithm_index]}_{self.env_name}"
#     for j in range(len(self.parameter_list[self.true_env_num])):
#         param_name = self.parameter_name_list[j]
#         param_value = self.parameter_list[self.true_env_num][j].tolist()
#         data_folder_name += f"_{param_name}_{str(param_value)}"
#     data_folder_name += f"_{self.max_timestep}_maxStep_{self.trajectory_num}_trajectory_{self.true_env_num}"
#     data_q_name = data_folder_name + "_q"
#     data_q_path = os.path.join(Offline_data_folder, data_q_name)
#     data_r_name = data_folder_name + "_r"
#     data_r_path = os.path.join(Offline_data_folder, data_r_name)
#     data_size_name = data_folder_name + "_size"
#     data_size_path = os.path.join(Offline_data_folder, data_size_name)
#
#     if not self.whether_file_exists(data_q_path + ".pkl"):
#         logging.info("Enter get qa calculate loop")
#         start_time = time.time()
#
#         env_copy_start_time = time.time()
#         env_copy_lists = [[self.create_deep_copies(env, self.batch_size) for env in self.env_list] for _ in
#                           self.policy_list]
#
#         with SharedMemoryManager() as smm:
#             env_copy_serialized = dill.dumps(env_copy_lists)
#             env_copy_shm = smm.SharedMemory(size=len(env_copy_serialized))
#             env_copy_shm.buf[:len(env_copy_serialized)] = env_copy_serialized
#
#             policy_list_serialized = dill.dumps(self.policy_list)
#             policy_list_shm = smm.SharedMemory(size=len(policy_list_serialized))
#             policy_list_shm.buf[:len(policy_list_serialized)] = policy_list_serialized
#
#             data_serialized = dill.dumps(self.data)
#             data_shm = smm.SharedMemory(size=len(data_serialized))
#             data_shm.buf[:len(data_serialized)] = data_serialized
#
#             env_list_serialized = dill.dumps(self.env_list)
#             env_list_shm = smm.SharedMemory(size=len(env_list_serialized))
#             env_list_shm.buf[:len(env_list_serialized)] = env_list_serialized
#
#             q_sa_serialized = dill.dumps(self.q_sa)
#             q_sa_shm = smm.SharedMemory(size=len(q_sa_serialized))
#             q_sa_shm.buf[:len(q_sa_serialized)] = q_sa_serialized
#
#             r_plus_vfsp_serialized = dill.dumps(self.r_plus_vfsp)
#             r_plus_vfsp_shm = smm.SharedMemory(size=len(r_plus_vfsp_serialized))
#             r_plus_vfsp_shm.buf[:len(r_plus_vfsp_serialized)] = r_plus_vfsp_serialized
#
#             # Save the shared memory names and sizes to a list
#             fin_list = [
#                 env_copy_shm.name, len(env_copy_serialized),
#                 policy_list_shm.name, len(policy_list_serialized),
#                 data_shm.name, len(data_serialized),
#                 env_list_shm.name, len(env_list_serialized),
#                 q_sa_shm.name, len(q_sa_serialized),
#                 r_plus_vfsp_shm.name, len(r_plus_vfsp_serialized)
#             ]
#             fin_list_serialized = dill.dumps(fin_list)
#             fin_list_shm = smm.SharedMemory(size=len(fin_list_serialized))
#             fin_list_shm.buf[:len(fin_list_serialized)] = fin_list_serialized
#
#             threading_start_time = time.time()
#             with Pool(processes=1) as pool:
#                 results = [pool.apply_async(self.process_env, args=(i, j, fin_list_shm.name)) for i in
#                            range(len(self.env_list)) for j in range(len(self.policy_list))]
#
#                 for result in results:
#                     result.get()
#
#                 pool.close()
#                 pool.join()
#
#             threading_end_time = time.time()
#             logging.info(f"Threading (env-policy) time: {threading_end_time - threading_start_time} seconds")
#
#             end_time = time.time()
#             logging.info(f"Total running time get_qa: {end_time - start_time} seconds")
#
#             self.q_sa = dill.loads(q_sa_shm.buf)
#             self.r_plus_vfsp = dill.loads(r_plus_vfsp_shm.buf)
#
#             self.save_as_pkl(data_q_path, self.q_sa)
#             self.save_as_pkl(data_r_path, self.r_plus_vfsp)
#             self.save_as_pkl(data_size_path, self.data_size)
#
#             gc.collect()
#     else:
#         self.q_sa = self.load_from_pkl(data_q_path)
#         self.r_plus_vfsp = self.load_from_pkl(data_r_path)
#         self.data_size = self.load_from_pkl(data_size_path)
#
#
# def get_memory_usage(self):
#     # 获取当前进程的内存使用情况（以GB为单位）
#     process = psutil.Process()
#     mem_info = process.memory_info()
#     return mem_info.rss / (1024 ** 3)  # 转换为GB


# def run_simulation(self, state_action_policy_env_batch):
#     states, actions, policy, envs = state_action_policy_env_batch
#
#     # Initialize environments with states
#     for env, state in zip(envs, states):
#         env.reset()
#         env.observation = state
#
#     total_rewards = np.zeros(len(states))
#     num_steps = np.zeros(len(states))
#     discount_factors = np.ones(len(states))
#     done_flags = np.zeros(len(states), dtype=bool)
#
#     while not np.all(done_flags) and np.any(num_steps < self.max_timestep):
#         actions_batch = policy.predict(np.array(states))
#         for idx, (env, action) in enumerate(zip(envs, actions_batch)):
#             if not done_flags[idx]:
#                 next_state, reward, done, _, _ = env.step(action)
#                 total_rewards[idx] += reward * discount_factors[idx]
#                 discount_factors[idx] *= self.gamma
#                 num_steps[idx] += 1
#                 states[idx] = next_state
#                 done_flags[idx] = done
#
#     return total_rewards
#
#
# def get_qa(self, policy_number, env_copy_list, states, actions, batch_size=8):
#     policy = self.policy_list[policy_number]
#     results = []
#
#     total_len = len(states)
#     for i in range(0, total_len, batch_size):
#         actual_batch_size = min(batch_size, total_len - i)
#         state_action_policy_env_pairs = (
#             states[i:i + actual_batch_size], actions[i:i + actual_batch_size], policy,
#             env_copy_list[:actual_batch_size])
#         batch_results = self.run_simulation(state_action_policy_env_pairs)
#         results.extend(batch_results)
#
#     return np.array(results)
#
#
# def create_deep_copies(self, env, batch_size):
#     return [copy.deepcopy(env) for _ in range(batch_size)]
#
#
# def process_env(self, env_index, policy_list, env_list, data, gamma, batch_size, trajectory_num, data_size):
#     logging.info(f"Starting processing for env_index {env_index}")
#     start_time = time.time()
#
#     num_policies = len(policy_list)
#     q_sa = [np.zeros(data_size) for _ in range(len(env_list) * num_policies)]
#     r_plus_vfsp = [np.zeros(data_size) for _ in range(len(env_list) * num_policies)]
#     local_data_size = 0
#
#     env = env_list[env_index]
#     env_copy_list = [copy.deepcopy(env) for _ in range(batch_size)]
#     logging.info(f"Memory usage after creating environment copies: {memory_usage()} MB")
#     for policy_index in range(num_policies):
#         ptr = 0
#         trajectory_length = 0
#         while ptr < trajectory_num:
#             length = data.get_iter_length(ptr)
#             state, action, next_state, reward, done = data.sample(ptr)
#
#             q_values = self.get_qa(policy_index, env_copy_list, state, action, batch_size)
#             if q_values.shape[0] != length:
#                 raise ValueError(f"Shape mismatch: q_values.shape[0]={q_values.shape[0]}, length={length}")
#
#             q_sa_idx = env_index * num_policies + policy_index
#             q_sa[q_sa_idx][trajectory_length:trajectory_length + length] = q_values
#
#             vfsp_values = (reward + self.get_qa(policy_index, env_copy_list, next_state,
#                                                 policy_list[policy_index].predict(next_state), batch_size) *
#                            (1 - np.array(done)) * gamma)
#             if vfsp_values.shape[0] != length:
#                 raise ValueError(f"Shape mismatch: vfsp_values.shape[0]={vfsp_values.shape[0]}, length={length}")
#
#             r_plus_vfsp[q_sa_idx][trajectory_length:trajectory_length + length] = vfsp_values.flatten()[:length]
#             trajectory_length += length
#             ptr += 1
#
#         local_data_size += trajectory_length
#
#     end_time = time.time()
#     logging.info(f"Finished processing for env_index {env_index} in {end_time - start_time} seconds")
#     logging.info(f"Memory usage after processing env_index {env_index}: {memory_usage()} MB")
#
#     return q_sa, r_plus_vfsp, local_data_size
#
#
# def get_whole_qa(self, algorithm_index):
#     Offline_data_folder = "Offline_data"
#     self.create_folder(Offline_data_folder)
#     data_folder_name = f"{self.algorithm_name_list[algorithm_index]}_{self.env_name}"
#     for j in range(len(self.parameter_list[self.true_env_num])):
#         param_name = self.parameter_name_list[j]
#         param_value = self.parameter_list[self.true_env_num][j].tolist()
#         data_folder_name += f"_{param_name}_{str(param_value)}"
#     data_folder_name += f"_{self.max_timestep}_maxStep_{self.trajectory_num}_trajectory_{self.true_env_num}"
#     data_q_name = data_folder_name + "_q"
#     data_q_path = os.path.join(Offline_data_folder, data_q_name)
#     data_r_name = data_folder_name + "_r"
#     data_r_path = os.path.join(Offline_data_folder, data_r_name)
#     data_size_name = data_folder_name + "_size"
#     data_size_path = os.path.join(Offline_data_folder, data_size_name)
#
#     if not self.whether_file_exists(data_q_path + ".pkl"):
#         logging.info("Enter get qa calculate loop")
#         start_time = time.time()
#
#         threading_start_time = time.time()
#         with Pool(pathos.multiprocessing.cpu_count()) as pool:
#             results = [pool.apply_async(self.process_env, args=(
#                 i, self.policy_list, self.env_list, self.data, self.gamma, self.batch_size, self.trajectory_num,
#                 self.data.size)) for i in range(len(self.env_list))]
#
#             q_sa_aggregated = [np.zeros(self.data.size) for _ in range(len(self.env_list) * len(self.policy_list))]
#             r_plus_vfsp_aggregated = [np.zeros(self.data.size) for _ in
#                                       range(len(self.env_list) * len(self.policy_list))]
#             data_size_aggregated = 0
#
#             for result in results:
#                 q_sa_partial, r_plus_vfsp_partial, data_size_partial = result.get()
#                 for idx in range(len(q_sa_aggregated)):
#                     q_sa_aggregated[idx] += q_sa_partial[idx]
#                     r_plus_vfsp_aggregated[idx] += r_plus_vfsp_partial[idx]
#                 data_size_aggregated += data_size_partial
#
#         self.q_sa = q_sa_aggregated
#         self.r_plus_vfsp = r_plus_vfsp_aggregated
#         self.data_size = data_size_aggregated
#
#         threading_end_time = time.time()
#         logging.info(f"Threading (env only) time: {threading_end_time - threading_start_time} seconds")
#         logging.info(f"Memory usage after processing all environments: {memory_usage()} MB")
#
#         end_time = time.time()
#         logging.info(f"Total running time get_qa: {end_time - start_time} seconds")
#
#         self.save_as_pkl(data_q_path, self.q_sa)
#         self.save_as_pkl(data_r_path, self.r_plus_vfsp)
#         self.save_as_pkl(data_size_path, self.data_size)
#     else:
#         self.q_sa = self.load_from_pkl(data_q_path)
#         self.r_plus_vfsp = self.load_from_pkl(data_r_path)
#         self.data_size = self.load_from_pkl(data_size_path)
#
#         logging.info(f"Memory usage after loading from pickle: {memory_usage()} MB")


# def generate_one_trajectory(self, env_number, max_time_step, algorithm_name, unique_seed):
#     Policy_operation_folder = "Policy_operation"
#     Policy_saving_folder = os.path.join(Policy_operation_folder, "Policy_trained")
#     self.create_folder(Policy_saving_folder)
#     policy_folder_name = f"{self.env_name}"
#     for j in range(len(self.parameter_list[env_number])):
#         param_name = self.parameter_name_list[j]
#         param_value = self.parameter_list[env_number][j].tolist()
#         policy_folder_name += f"_{param_name}_{str(param_value)}"
#     policy_saving_path = os.path.join(Policy_saving_folder, policy_folder_name)
#     policy_model_name = f"{algorithm_name}_{str(self.policy_total_step)}_{str(self.policy_learning_rate)}_{str(self.policy_hidden_layer)}_{self.policy_total_step}step.d3"
#     policy_path = os.path.join(policy_saving_path, policy_model_name)
#     policy = d3rlpy.load_learnable(policy_path, device=self.device)
#     env = self.env_list[env_number]
#     obs, info = env.reset(seed=unique_seed)
#
#     observations = []
#     rewards = []
#     actions = []
#     dones = []
#     next_steps = []
#     episode_data = {}
#     observations.append(obs)
#     # print("initial obs : ",obs)
#     for t in range(max_time_step):
#
#         action = policy.predict(np.array([obs]))
#         # print("action after prediction : ",action)
#         state, reward, done, truncated, info = env.step(action[0])
#         actions.append(action[0])
#         rewards.append(reward)
#         dones.append(done)
#         next_steps.append(state)
#         if ((t != max_time_step - 1) and done == False):
#             observations.append(state)
#
#         obs = state
#         # print("state in env step : ",state)
#
#         if done or truncated:
#             break
#     episode_data["action"] = actions
#     episode_data["state"] = observations
#     episode_data["rewards"] = rewards
#     episode_data["done"] = dones
#     episode_data["next_state"] = next_steps
#     return episode_data
#
#
# def load_offline_data(self, max_time_step, algorithm_name, true_env_number):
#     self.print_environment_parameters()
#     self.true_env_num = true_env_number
#     Offine_data_folder = "Offline_data"
#     self.create_folder(Offine_data_folder)
#     data_folder_name = f"{algorithm_name}_{self.env_name}"
#     for j in range(len(self.parameter_list[true_env_number])):
#         param_name = self.parameter_name_list[j]
#         param_value = self.parameter_list[true_env_number][j].tolist()
#         data_folder_name += f"_{param_name}_{str(param_value)}"
#     data_folder_name += f"_{max_time_step}_maxStep_{self.trajectory_num}_trajectory_{self.true_env_num}"
#     data_path = os.path.join(Offine_data_folder, data_folder_name)
#     data_seeds_name = data_folder_name + "_seeds"
#     data_seeds_path = os.path.join(Offine_data_folder, data_seeds_name)
#     if os.path.exists(data_path):
#         self.data = self.load_from_pkl(data_path)
#         self.data_size = self.data.size
#         self.unique_numbers = self.load_from_pkl(data_seeds_path)
#         self.q_sa = [np.zeros(self.data.size) for _ in range(len(self.env_list) * len(self.env_list))]
#         self.r_plus_vfsp = [np.zeros(self.data.size) for _ in range(len(self.env_list) * len(self.env_list))]
#     else:
#         self.generate_offline_data(max_time_step, algorithm_name, true_env_number)
#
#
# def generate_offline_data(self, max_time_step, algorithm_name, true_env_number):
#     self.print_environment_parameters()
#     self.true_env_num = true_env_number
#     unique_numbers = self.generate_unique_numbers(self.trajectory_num, 1, 12345)
#     self.unique_numbers = unique_numbers
#     final_data = []
#     for i in range(self.trajectory_num):
#         one_episode_data = self.generate_one_trajectory(true_env_number, max_time_step, algorithm_name,
#                                                         unique_numbers[i])
#         final_data.append(one_episode_data)
#
#     Offine_data_folder = "Offline_data"
#     self.create_folder(Offine_data_folder)
#     data_folder_name = f"{algorithm_name}_{self.env_name}"
#     for j in range(len(self.parameter_list[true_env_number])):
#         param_name = self.parameter_name_list[j]
#         param_value = self.parameter_list[true_env_number][j].tolist()
#         data_folder_name += f"_{param_name}_{str(param_value)}"
#     data_folder_name += f"_{max_time_step}_maxStep_{self.trajectory_num}_trajectory_{self.true_env_num}"
#     data_path = os.path.join(Offine_data_folder, data_folder_name)
#     data_seeds_name = data_folder_name + "_seeds"
#     data_seeds_path = os.path.join(Offine_data_folder, data_seeds_name)
#     self.data = CustomDataLoader(final_data)
#     self.data_size = self.data.size
#     self.q_sa = [np.zeros(self.data.size) for _ in range(len(self.env_list) * len(self.env_list))]
#     self.r_plus_vfsp = [np.zeros(self.data.size) for _ in range(len(self.env_list) * len(self.env_list))]
#     self.save_as_pkl(data_path, self.data)
#     self.save_as_pkl(data_seeds_path, self.unique_numbers)