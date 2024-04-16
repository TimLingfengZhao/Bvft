import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import gym
from d3rlpy.algos import  RandomPolicyConfig
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
from d3rlpy.dataset import MDPDataset, Episode
from scope_rl.dataset import SyntheticDataset
from scope_rl.policy import GaussianHead
from scope_rl.ope import OffPolicyEvaluation as OPE
from scope_rl.ope.continuous import DirectMethod as DM
from scope_rl.policy import ContinuousEvalHead
from d3rlpy.algos import DDPGConfig
from d3rlpy.dataset import create_fifo_replay_buffer
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQLConfig
from d3rlpy.algos import BCQConfig
from d3rlpy.algos import BCConfig
from d3rlpy.algos import SACConfig
from d3rlpy.algos import TD3Config

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


def top_k_indices(lst, k):

    if k > len(lst):
        raise ValueError("k cannot be greater than the list length")

    top_k_values = heapq.nlargest(k, lst)

    top_k_indices = []
    for value in top_k_values:
        top_k_indices.extend([i for i, x in enumerate(lst) if x == value])

    return top_k_indices[:k]
def plot_value(k_list):
    directory = "k_figure"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, "normalized_regret_plot.png")
    # Plotting the list
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, marker='o',linestyle='-')
    plt.title('Bvft-PE-avgQ')
    plt.xticks(range(len(k_list)), range(1, len(k_list) + 1))
    plt.xlabel('k')
    plt.ylabel('Normalized Regret')
    plt.grid(True)
    plt.savefig(file_path)
    plt.show()
def plot_value_precision(k_list):
    directory = "k_figure"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, "precision_plot.png")
    # Plotting the list
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, marker='o',linestyle='-')
    plt.title('Bvft-PE-avgQ')
    plt.xticks(range(len(k_list)), range(1, len(k_list) + 1))
    plt.xlabel('k')
    plt.ylabel('precision ')
    plt.grid(True)
    plt.savefig(file_path)
    plt.show()

def rank_elements(lst):
    sorted_pairs = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
    ranks = [0] * len(lst)
    for rank, (original_index, _) in enumerate(sorted_pairs, start=1):
        ranks[original_index] = rank
    return ranks


def rank_elements(lst):
    enumerated_list = list(enumerate(lst))
    sorted_enumerated_list = sorted(enumerated_list, key=lambda x: x[1])

    rank_dict = {}
    current_rank = 0

    for i, (index, value) in enumerate(sorted_enumerated_list):
        if i == 0 or value != sorted_enumerated_list[i - 1][1]:
            rank_dict[value] = current_rank
            current_rank += 1
        else:
            rank_dict[value] = current_rank - 1

    rank_list = [0] * len(lst)
    for index, value in enumerated_list:
        rank_list[index] = rank_dict[value]

    return rank_list


def plot_histogram(policy_names, total_rewards,save_path, file_name="total_rewards_histogram.png"):
    os.makedirs(save_path, exist_ok=True)

    # Determine the policy with the maximum reward
    max_reward = max(total_rewards)
    max_index = total_rewards.index(max_reward)

    # Create the histogram
    fig, ax = plt.subplots(figsize=(34, 17))
    bars = ax.bar(policy_names, total_rewards, color='blue')
    bars[max_index].set_color('red')

    for bar in bars:
        height = bar.get_height()
        label = f'{height:,.5f}'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    ax.set_xlabel('Policy Names')
    ax.set_ylabel('Total Rewards')
    ax.set_title('Total Rewards of Different Policies')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    full_path = os.path.join(save_path, file_name)
    plt.savefig(full_path)
    plt.close()
    print(f"Plot saved to {full_path}")

def calculate_dataset_reward(dataset,gamma):
    done = dataset["done"]
    state = dataset["state"]
    reward = dataset["reward"]

    total_reward = 0
    for i in range(500):
        current_gamma = 1
        current_reward = 0
        for j in range(1000):
            current_reward = current_reward + reward[1000*i+j] * current_gamma
            current_gamma = current_gamma * gamma
        total_reward = total_reward + current_reward
    total_reward = total_reward / 500

    return total_reward
def calculate_dataset_reward_revise(dataset,gamma):
    done = dataset["done"]
    state = dataset["state"]
    reward = dataset["reward"]
    term = dataset["terminal"]
    env = gym.make("Hopper-v4")
    cul_reward = 0
    current_gamma = 1
    ind = []
    for i in range(5):
        ind.append(i*1000)
    for i in ind:
        while (term[i] == 0):
            cul_reward = reward[i] * current_gamma + cul_reward
            current_gamma = current_gamma * gamma
            i = i + 1
    average_reward = cul_reward / 5
    return cul_reward,average_reward
def calculate_policy_value(env, policy, gamma=0.99,num_run=100):
    total_rewards = 0
    max_iteration = 1000
    for i in range(num_run):
        num_step = 0
        discount_factor = 1
        observation, info = env.reset(seed=12345)
        action = policy.predict(np.array([observation]))
        ui = env.step(action[0])
        state = ui[0]
        reward = ui[1]
        done = ui[2]
        while ((not done) and (num_step < 1000)):
            action = policy.predict(np.array([state]))
            ui = env.step(action[0])
            state = ui[0]
            reward = ui[1]
            done = ui[2]
            total_rewards += reward * discount_factor
            discount_factor *= gamma
            num_step +=1
    total_rewards = total_rewards / num_run
    return total_rewards

def plot_bar_graph(name_list,normalized_value,environment_name):
    save_dir = "k_figure"
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(name_list)))

    bars = ax.bar(name_list, normalized_value, color=colors)

    ax.set_ylabel('Normalized Value')
    ax.set_title('Normalized MSE of FQE with Different Hyperparameters')
    ax.set_xticks([1.5])
    ax.set_xticklabels([environment_name])

    ax.legend(bars, name_list, loc="upper right")

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'normalized_MSE_FQE_histogram.png')
    plt.savefig(save_path)
    plt.close()

def cal_ranking_index(ranking_list):
    index_list = []
    for i in range(len(ranking_list)):
        for j in range(len(ranking_list[0])):
            if(ranking_list[i][j]==0):
                index_list.append(i*4+j)
    return index_list

def k_plot(list_one,figure_name):
    plt.plot(list_one, marker='o', linestyle='-', color='blue')
    plt.title('Q predictiona t initial state 12345')
    plt.xlabel('Iteration')
    plt.ylabel('Prediction_value')
    save_dir = "k_figure"
    plt.grid(True)
    save_path = os.path.join(save_dir, figure_name)
    plt.savefig(save_path)
    plt.close()

def k_plot_two(list_one,figure_name,save_dir):
    plt.plot(list_one, marker='o', linestyle='-', color='blue')
    plt.title('Q predictiona t initial state 12345')
    plt.xlabel('Iteration')
    plt.ylabel('Prediction_value')
    plt.grid(True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, figure_name)
    plt.savefig(save_path)
    plt.close()


def plot_and_save_bar_graph_with_labels(y_values, x_labels, folder_path, filename="bar_graph_with_labels.png"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    base_width = 1
    figure_width = max(len(x_labels) * base_width, 8)
    figure_height = 15
    plt.figure(figsize=(figure_width, figure_height))

    plt.bar(x_labels, y_values, width=1, color='skyblue', edgecolor='grey')
    plt.xticks(rotation=45, ha='right')
    plt.title("Policy Performance Bar Graph")

    plt.xlabel("policy name")
    plt.ylabel("policy expected value")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, filename))

    plt.close()


def plot_and_save_bar_graph_with_labels_FQE(y_values, x_labels, folder_path, filename="bar_graph_with_labels.png"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    base_width = 1
    figure_width = max(len(x_labels) * base_width, 8)
    figure_height = 15
    plt.figure(figsize=(figure_width, figure_height))

    plt.bar(x_labels, y_values, width=1, color='skyblue', edgecolor='grey')
    plt.xticks(rotation=45, ha='right')
    plt.title("FQE Performance Bar Graph")

    plt.xlabel("FQE name")
    plt.ylabel("FQE expected value")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, filename))

    plt.close()
def save_as_pkl(file_path, list_to_save):
    full_path = f"{file_path}.pkl"
    with open(full_path, 'wb') as file:
        pickle.dump(list_to_save, file)


def save_as_txt(file_path, list_to_save):
    full_path = f"{file_path}.txt"
    with open(full_path, 'w') as file:
        for item in list_to_save:
            file.write(f"{item}\n")


def save_dict_as_txt(file_path, dict_to_save):
    full_path = f"{file_path}.txt"
    with open(full_path, 'w') as file:
        for key, value in dict_to_save.items():
            file.write(f"{key}:{value}\n")


def load_dict_from_txt(file_path):
    with open(file_path, 'r') as file:
        return {line.split(':', 1)[0]: line.split(':', 1)[1].strip() for line in file}


def list_to_dict(name_list, reward_list):
    return dict(zip(name_list, reward_list))

def load_from_pkl(file_path):
    full_path = f"{file_path}.pkl"
    with open(full_path, 'rb') as file:
        data = pickle.load(file)
    return data


def normalized_mean_suqare_error(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("The length of actual and predicted values must be the same.")
    print("actual value list : ",actual)
    print("predicted value list : ",predicted)
    mse = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)
    print("MSE : ", mse)
    range_squared = (max(actual) - min(actual)) ** 2
    # denominator = range_squared / len(actual)
    denominator = range_squared
    if denominator == 0:
        raise ValueError("The range of actual values is zero. NMSE cannot be calculated.")
    nmse = mse / denominator

    mean_actual = sum(actual) / len(actual)

    standard_error = ((sum((a - mean_actual) ** 2 for a in actual) / len(actual)-1) ** 0.5)/denominator
    return nmse,standard_error


def is_key_in_dict(key,dictionary):
    return key in dictionary

def draw_mse_graph(combinations, means, colors,standard_errors, labels, folder_path, filename="FQE_MSES.png"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    group_width = 1.0
    n_bars = len(labels)
    bar_width = group_width / n_bars

    indices = np.arange(1)

    fig, ax = plt.subplots()

    for i in range(n_bars):
        bar_x_positions = indices - (group_width - bar_width) / 2 + i * bar_width

        errors_below = 2 * standard_errors[i]
        errors_above = 2 * standard_errors[i]

        ax.bar(bar_x_positions, means[i], yerr=[[errors_below], [errors_above]],
               capsize=5, alpha=0.7, color=colors[i], label=labels[i], width=bar_width)

    ax.set_ylabel('Normalized MSE')
    ax.set_title('Normalized MSE of FQE\'s J(π) estimations')

    ax.set_xticks(indices)
    ax.set_xticklabels(combinations)

    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, filename))
    plt.close()


def plot_predictions(x_axis_names, predictions, line_names, saved_folder_path, saved_name, picture_name):
    if len(predictions) != len(line_names):
        raise ValueError("predictions length is different with line name's length")

    plt.figure(figsize=(10, 6))
    for i, pred in enumerate(predictions):
        if len(x_axis_names[i]) != len(pred):
            raise ValueError(f"x axis length is different with prediction length")
        plt.plot(x_axis_names[i], pred, label=line_names[i], marker='o')


    plt.xlabel('X Axis')
    plt.ylabel('Prediction value')
    plt.title(picture_name)

    plt.xticks(ticks=x_axis_names[0], labels=[str(x) for x in x_axis_names[0]])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=len(line_names))
    plt.tight_layout()

    if not os.path.exists(saved_folder_path):
        os.makedirs(saved_folder_path)

    plt.savefig(os.path.join(saved_folder_path, saved_name))
    plt.close()
#
# def draw_mse_graph_split(combinations, means, colors,standard_errors, labels, folder_path, filename="FQE_MSES.png"):
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#
#     group_width = 1.0
#     n_bars = len(labels)
#     bar_width = group_width / n_bars
#
#     indices = np.arange(1)
#
#     fig, ax = plt.subplots()
#
#     for i in range(n_bars):
#         bar_x_positions = indices - (group_width - bar_width) / 2 + i * bar_width
#
#         errors_below = 2 * standard_errors[i]
#         errors_above = 2 * standard_errors[i]
#
#         ax.bar(bar_x_positions, means[i], yerr=[[errors_below], [errors_above]],
#                capsize=5, alpha=0.7, color=colors[i], label=labels[i], width=bar_width)
#
#     ax.set_ylabel('Normalized MSE')
#     ax.set_title('Normalized MSE of FQE\'s J(π) estimations')
#
#     ax.set_xticks(indices)
#     ax.set_xticklabels(combinations)
#
#     ax.legend(loc='upper right')
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(folder_path, filename))
#     plt.close()

def generate_unique_colors(number_of_colors):

    cmap = plt.get_cmap('tab20')

    if number_of_colors <= 20:
        colors = [cmap(i) for i in range(number_of_colors)]
    else:
        colors = [cmap(i) for i in np.linspace(0, 1, number_of_colors)]
    return colors
def calculate_top_k_precision(initial_state,env, policies,rank_list, k=2):
    #ranking_list 给了前k个policy的坐标
    num_trajectory = 5
    trajectory_length = 1000
    state_dim = 11
    action_dim = 3
    noise_std = 0.1
    cur_policy = policies

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    proportion = 0

    policy_performance = []
    for i in range(len(policies)):
        policy_performance.append(calculate_policy_value(env, policies[i], gamma=0.99, num_run=30))
    policy_ranking_groundtruth = rank_elements(policy_performance)
    for pos in rank_list :
        if (rank_list[pos]<=k-1 and policy_ranking_groundtruth[pos] <= k-1):
            proportion+=1
    proportion = proportion/k

    return proportion
def calculate_top_k_normalized_regret(ranking_list, policy_list,env,k=2):
    print("calcualte top k normalized regret")
    policy_performance = []
    for i in range(len(policy_list)):
        policy_performance.append(calculate_policy_value(env,policy_list[i],gamma=0.99,
                                                         num_run=30))
    ground_truth_value = max(policy_performance)
    worth_value = min(policy_performance)
    if((ground_truth_value - worth_value) == 0):
        print("the maximum is equal to worth value, error!!!!")
        return 99999
    gap_list = []
    for i in range(len(ranking_list)):
        if(ranking_list[i]<=k-1):
            value = policy_performance[i]
            norm = (ground_truth_value - value) / (ground_truth_value - worth_value)
            gap_list.append(norm)
    return min(gap_list)


def plot_multiple_graphs(data_groups, save_path, graph_names, y_axis_name, colors):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(15, 5))  #

    for data, name, color in zip(data_groups, graph_names, colors):
        means = np.mean(data, axis=0)
        std_errors = np.std(data, axis=0, ddof=1) / np.sqrt(len(data))
        conf_intervals = 2 * std_errors
        x_values = list(range(1, len(means) + 1))

        plt.errorbar(x_values, means, yerr=conf_intervals, label=name, color=color)


    plt.xlabel('k')
    plt.ylabel(y_axis_name)
    plt.title('K regret and K precision graph')
    plt.legend()

    plt.savefig(os.path.join(save_path, 'regret-precision.png'))
    plt.close()



