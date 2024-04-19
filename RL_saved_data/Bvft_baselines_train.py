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

def pick_policy(m,device):
    policy_folder = 'policy_trained'
    if not os.path.exists(policy_folder ):
        os.makedirs(policy_folder )
    policy_files = sample_files(policy_folder, m)
    policy_name_list = []
    policy_list = []
    for policy_file_name in policy_files:          #policy we want to evaluate
        policy_path = os.path.join(policy_folder, policy_file_name)
        policy = d3rlpy.load_learnable(policy_path, device=device)

        policy_name_list.append(policy_file_name[:-3])
        policy_list.append(policy)
    return policy_name_list, policy_list
def pick_FQE(policy_name_list,device,number_FQE,FQE_total_step,FQE_episode_step,replay_buffer):
    policy_folder = 'policy_trained'
    FQE_lr_list = [1e-4,2e-5]
    FQE_hl_list = [[128,256],[128,1024]]
    Q_FQE = []
    for policy_file_name in policy_name_list:
        policy_path = os.path.join(policy_folder, policy_file_name)
        policy = d3rlpy.load_learnable(policy_path + ".d3", device=device)
        for FQE_learning_rate in FQE_lr_list:
            for FQE_hidden_layer in FQE_hl_list:
                Q_list = []
                FQE_directory = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer)
                if not os.path.exists(FQE_directory):
                    os.makedirs(FQE_directory)
                for i in range(number_FQE):
                    FQE_step = int(FQE_total_step / number_FQE)
                    FQE_model_pre = 'FQE_' + str(FQE_learning_rate) + '_' + str(FQE_hidden_layer) + '_' + str(
                        FQE_step * (i + 1)) + "step" + "_"
                    FQE_model_name = FQE_model_pre + policy_file_name
                    FQE_model_name = FQE_model_name+ ".pt"
                    FQE_file_path = os.path.join(FQE_directory, FQE_model_name)
                    fqeconfig = d3rlpy.ope.FQEConfig(
                        learning_rate=FQE_learning_rate,
                        encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=FQE_hidden_layer)
                    )
                    fqe = FQE(algo=policy, config=fqeconfig, device=device)
                    fqe.build_with_dataset(replay_buffer)
                    fqe.load_model(FQE_file_path)
                    Q_list.append(fqe)
        Q_FQE.append(Q_list)
    return Q_FQE
def Bvft_ranking(policy_name_list,Q_FQE,test_data,gamma, rmax, rmin, record,batch_dim,num_runs):
    for i in range(len(Q_FQE)):
        q_functions = Q_FQE[i]
        bvft_instance = BVFT(q_functions, test_data, gamma, rmax, rmin, policy_name_list[i],record, "torch_actor_critic_cont", verbose=True,batch_dim=1000)
        bvft_instance.run()
    Bvft_folder = "Bvft_Records"
    if not os.path.exists(Bvft_folder  ):
        os.makedirs(Bvft_folder  )
    ranking_list = []
    avg_q_list = []
    for bvft_result in os.listdir(Bvft_folder):
        Bvft_path = os.path.join(Bvft_folder, bvft_result)
        bvft_result = BvftRecord.load(Bvft_path)
        ranking_list.append(bvft_result.ranking.tolist())
        avg_q_list.append(bvft_result.avg_q)
    ranking_list = []
    avg_q_list = []
    for bvft_result in os.listdir(Bvft_folder):
        Bvft_path = os.path.join(Bvft_folder, bvft_result)
        bvft_result = BvftRecord.load(Bvft_path)
        ranking_list.append(bvft_result.ranking.tolist())
        avg_q_list.append(bvft_result.avg_q)
    best_ranking_index = []
    for ele in ranking_list:
        best_ranking_index.append(np.argmin(ele))
    best_avg_q = []
    for i in range(len(best_ranking_index)):
        best_avg_q.append(avg_q_list[i][best_ranking_index[i]])
    bvft_ranking = rank_elements(best_avg_q)
    Bvft_saving_folder = "Bvft_saving_place"
    Bvft_ranking_saving_folder = "Bvft_FQE_avgQ_ranking"
    Bvft_ranking_saving_path_before = os.path.join(Bvft_saving_folder,Bvft_ranking_saving_folder)
    if not os.path.exists(Bvft_ranking_saving_path_before):
        os.makedirs(Bvft_ranking_saving_path_before  )
    runs_folder = str(num_runs)+"run"
    Bvft_ranking_saving_path = os.path.join(Bvft_ranking_saving_path_before,runs_folder)
    if not os.path.exists(Bvft_ranking_saving_path):
        os.makedirs(Bvft_ranking_saving_path  )
    policy_name_folder = "policy_names"
    ranking_name_folder = "rankings"
    Policy_name_saving_path = os.path.join(Bvft_ranking_saving_path,policy_name_folder)
    Ranking_saving_path = os.path.join(Bvft_ranking_saving_path,ranking_name_folder)
    save_as_txt(Policy_name_saving_path,policy_name_list)
    save_as_pkl(Policy_name_saving_path,policy_name_list)
    save_as_txt(Ranking_saving_path,bvft_ranking)
    save_as_pkl(Ranking_saving_path,bvft_ranking)
    delete_files_in_folder(Bvft_folder)

def Initial_Q_ranking(policy_list,policy_name_list,env,num_runs):
    performance_folder = "policy_returned_result"
    total_name = "policy_returned_total.txt"
    performance_total_path = os.path.join(performance_folder,total_name)
    performance_dict = load_dict_from_txt(performance_total_path)

    performance_list = []

    policy_folder = 'policy_trained'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for policy_name in policy_name_list:
        included = False
        if policy_name in performance_dict:
            performance_list.append(performance_dict[policy_name])
            included = True
        if not included:
            policy_path = os.path.join(policy_folder, policy_name)
            policy = d3rlpy.load_learnable(policy_path+".d3", device=device)
            performance_list.append(calculate_policy_value(env, policy, gamma=0.99,num_run=100))
    initialQ_rankings = rank_elements(performance_list)
    Bvft_saving_folder = "Bvft_saving_place"
    Initial_Q_saving_folder = "InitialQ_ranking"
    Initial_Q_ranking_saving_path_before = os.path.join(Bvft_saving_folder,Initial_Q_saving_folder)
    if not os.path.exists(Initial_Q_ranking_saving_path_before):
        os.makedirs(Initial_Q_ranking_saving_path_before  )
    runs_folder = str(num_runs)+"run"
    Initial_Q_ranking_saving_path = os.path.join(Initial_Q_ranking_saving_path_before,runs_folder)
    if not os.path.exists(Initial_Q_ranking_saving_path):
        os.makedirs(Initial_Q_ranking_saving_path )
    policy_name_folder = "policy_names"
    Initial_Q_name_folder = "rankings"
    Policy_name_saving_path = os.path.join(Initial_Q_ranking_saving_path,policy_name_folder)
    Ranking_saving_path = os.path.join(Initial_Q_ranking_saving_path,Initial_Q_name_folder)
    save_as_txt(Policy_name_saving_path,policy_name_list)
    save_as_pkl(Policy_name_saving_path,policy_name_list)
    save_as_txt(Ranking_saving_path,initialQ_rankings)
    save_as_pkl(Ranking_saving_path,initialQ_rankings)


def random_ranking(policy_name_list,num_runs):
    r_ranking = list(range(1,len(policy_name_list)+1))
    random.shuffle(r_ranking)
    Bvft_saving_folder = "Bvft_saving_place"
    random_ranking_saving_folder = "Random_ranking"
    Random_ranking_saving_path_before = os.path.join(Bvft_saving_folder,random_ranking_saving_folder)
    if not os.path.exists(Random_ranking_saving_path_before):
        os.makedirs(Random_ranking_saving_path_before  )
    runs_folder = str(num_runs)+"run"
    Random_ranking_saving_path = os.path.join(Random_ranking_saving_path_before,runs_folder)
    if not os.path.exists(Random_ranking_saving_path):
        os.makedirs(Random_ranking_saving_path )
    policy_name_folder = "policy_names"
    ranking_name_folder = "rankings"
    Policy_name_saving_path = os.path.join(Random_ranking_saving_path,policy_name_folder)
    Ranking_saving_path = os.path.join(Random_ranking_saving_path,ranking_name_folder)
    save_as_txt(Policy_name_saving_path,policy_name_list)
    save_as_pkl(Policy_name_saving_path,policy_name_list)
    save_as_txt(Ranking_saving_path,r_ranking)
    save_as_pkl(Ranking_saving_path,r_ranking)


def run_baseline(FQE_episode_step, FQE_Pickup_number,FQE_total_step,m ,num_runs):
    print("begin bvft")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
    train_episodes = whole_dataset.episodes[0:2000]
    test_episodes = whole_dataset.episodes[2000:2276]

    buffer = FIFOBuffer(limit=50000)
    replay_buffer_test = ReplayBuffer(buffer=buffer, episodes=test_episodes)
    buffer = FIFOBuffer(limit=50000)
    replay_buffer = ReplayBuffer(buffer=buffer, episodes=train_episodes)

    gamma = 0.99
    rmax, rmin = 1.0, 0.0
    record = BvftRecord()
    batch_dim = 1000
    test_data = CustomDataLoader(replay_buffer_test,    batch_size=batch_dim)

    for i in range(num_runs):
        policy_name_list, policy_list = pick_policy(m, device)
        Q_FQE = pick_FQE(policy_name_list, device, FQE_Pickup_number, FQE_total_step, FQE_episode_step, replay_buffer)

        Bvft_ranking(policy_name_list, Q_FQE, test_data, gamma, rmax, rmin, record, batch_dim,i)
        random_ranking(policy_name_list,i)
        Initial_Q_ranking(policy_list, policy_name_list, env,i)
    print("finished baseline")

def load_rankings(num_runs):
    Bvft_saving_folder = "Bvft_saving_place"
    Bvft_ranking_folder = "Bvft_FQE_avgQ_ranking"
    InitialQ_ranking_folder = "InitialQ_ranking"
    Random_ranking_folder = "Random_ranking"
    Bvft_ranking_path = os.path.join(Bvft_saving_folder,Bvft_ranking_folder)
    InitialQ_ranking_path = os.path.join(Bvft_saving_folder,InitialQ_ranking_folder)
    Random_ranking_path = os.path.join(Bvft_saving_folder,Random_ranking_folder)
    Bvft_ranking_list = []
    InitialQ_ranking_list = []
    Random_ranking_list = []

    Bvft_policy_name_list = []
    InitialQ_policy_name_list = []
    Random_policy_name_list = []
    for run in num_runs:
        run_folder = run+"run"
        Bvft_ranking_run_folder = os.path.join(Bvft_ranking_path,run_folder)
        InitialQ_ranking_run_folder = os.path.join(InitialQ_ranking_path,run_folder)
        Random_ranking_run_folder = os.path.join(Random_ranking_path,run_folder)

        Bvft_ranking_run_path = os.path.join(Bvft_ranking_run_folder,"rankings.pkl")
        InitialQ_ranking_run_path = os.path.join(InitialQ_ranking_run_folder,"rankings.pkl")
        Random_ranking_run_path = os.path.join(Random_ranking_run_folder,"rankings.pkl")

        Bvft_policy_name_path = os.path.join(Bvft_ranking_run_folder,"policy_names.pkl")
        InitialQ_policy_name_path = os.path.join(InitialQ_ranking_run_folder,"policy_names.pkl")
        Random_policy_name_path = os.path.join(Random_ranking_run_folder,"policy_names")

        Bvft_ranking_run_list = load_from_pkl(Bvft_ranking_run_path)
        InitialQ_ranking_run_list = load_from_pkl(InitialQ_ranking_run_path)
        Random_ranking_run_list = load_from_pkl(Random_ranking_run_path)

        Bvft_policy_name_run_list = load_from_pkl(Bvft_policy_name_path)
        InitialQ_policy_name_run_list = load_from_pkl(InitialQ_policy_name_path)
        Random_policy_name_run_list = load_from_pkl(Random_policy_name_path)

        Bvft_ranking_list.append(Bvft_ranking_run_list)
        InitialQ_ranking_list.append(InitialQ_ranking_run_list)
        Random_ranking_list.append(Random_ranking_run_list)

        Bvft_policy_name_list.append(Bvft_policy_name_run_list)
        InitialQ_policy_name_list.append(InitialQ_policy_name_run_list)
        Random_policy_name_list.append(Random_policy_name_run_list)

        return Bvft_ranking_list,InitialQ_ranking_list,Random_ranking_list,Bvft_policy_name_list, InitialQ_policy_name_list,Random_policy_name_list



def calculate_k(num_runs):
    Bvft_ranking_list,InitialQ_ranking_list, Random_ranking_list,Bvft_policy_name_list, InitialQ_policy_name_list, Random_policy_name_list = load_rankings(num_runs)

    k_precision = calculate_top_k_precision(initial_state,env,policy_list,bvft_ranking,k)
    k_regret = calculate_top_k_normalized_regret(bvft_ranking,policy_list,env,k)
    delete_files_in_folder(Bvft_folder)
    return k_precision, k_regret

def plot_normalized_k(num_interval,FQE_number_epoch,FQE_episode_step,initial_state,m,k,num_runs):
    Bvft_saving_place = 'Bvft_saving_place'
    Bvft_k = 'Bvft_k_results'
    Bvft_k_save_path = os.path.join(Bvft_saving_place, Bvft_k)
    if not os.path.exists(Bvft_k_save_path):
        os.makedirs(Bvft_k_save_path)
    plot_precision_list = []
    plot_regret_list = []
    ci_precision_list = []
    ci_regret_list = []
    for k_i in range(k):
        k_precision = 0
        k_regret = 0
        k_precision_name = str((k_i+1))+"_precision"
        k_regret_name = str((k_i+1))+"_regret"
        k_precision_ci_name = str(k_i+1) + "_precision_ci"
        k_regret_ci_name = str(k_i+1) + "_regret_ci"
        precision_path = os.path.join(Bvft_k_save_path,k_precision_name)
        regret_path = os.path.join(Bvft_k_save_path,k_regret_name)
        precision_ci_path = os.path.join(Bvft_k_save_path, k_precision_ci_name)
        regret_ci_path = os.path.join(Bvft_k_save_path, k_regret_ci_name)

        if(os.path.exists(regret_ci_path+".pkl")):
            k_precision = load_from_pkl(precision_path)
            k_regret = load_from_pkl(regret_path)
            k_precision_ci = load_from_pkl(precision_ci_path)
            k_regret_ci = load_from_pkl(regret_ci_path)
            ci_precision_list.append(k_precision_ci[0])
            ci_regret_list.append(k_regret_ci[0])
            plot_precision_list.append(k_precision[0])
            plot_regret_list.append(k_regret[0])
        else:
            k_precision_list = []
            k_regret_list = []
            for i in range(num_runs):
                print("iteration : ", i)
                k_pre, k_reg = run_bvft(num_interval, FQE_number_epoch, FQE_episode_step, initial_state, m, (k_i+1))
                k_precision_list.append(k_pre)
                k_regret_list.append(k_reg)
            k_precision = sum(k_precision_list) / len(k_precision_list)
            k_regret = sum(k_regret_list) / len(k_regret_list)
            k_precision_ci = calculate_statistics(k_precision_list)
            k_regret_ci = calculate_statistics(k_regret_list)


            k_pre_saving_path = os.path.join(Bvft_k_save_path, k_precision_name)
            k_reg_saving_path = os.path.join(Bvft_k_save_path, k_regret_name)
            k_pre_ci_saving_path = os.path.join(Bvft_k_save_path, k_precision_ci_name)
            k_reg_ci_saving_path = os.path.join(Bvft_k_save_path, k_regret_ci_name)

            save_as_pkl(k_pre_saving_path, [k_precision])
            save_as_pkl(k_reg_saving_path, [k_regret])
            save_as_pkl(k_pre_ci_saving_path, [k_precision_ci])
            save_as_pkl(k_reg_ci_saving_path, [k_regret_ci])

            save_as_txt(k_pre_saving_path, [k_precision])
            save_as_txt(k_reg_saving_path, [k_regret])
            save_as_txt(k_pre_ci_saving_path, [k_precision_ci])
            save_as_txt(k_reg_ci_saving_path, [k_regret_ci])
            ci_precision_list.append(k_precision_ci)
            ci_regret_list.append(k_regret_ci)
            plot_precision_list.append(k_precision)
            plot_regret_list.append(k_regret)
    total_precision_name = "total_"+str(k)+"_precision"
    total_reg_name = "total"+str(k)+"_regret"
    total_precision_ci = "total_"+str(k)+"_precision_ci"
    total_reg_ci = "total"+str(k)+"_regret_ci"
    total_k_pre_saving_path = os.path.join(Bvft_k_save_path, total_precision_name)
    total_k_reg_saving_path = os.path.join(Bvft_k_save_path, total_reg_name)
    total_k_pre_ci_saving_path = os.path.join(Bvft_k_save_path, total_precision_ci)
    total_k_reg_ci_saving_path = os.path.join(Bvft_k_save_path, total_reg_ci)

    save_as_pkl(total_k_pre_saving_path, plot_precision_list)
    save_as_pkl(total_k_reg_saving_path, plot_regret_list)
    save_as_pkl(total_k_pre_ci_saving_path, ci_precision_list)
    save_as_pkl(total_k_reg_ci_saving_path, ci_regret_list)

    save_as_txt(total_k_pre_saving_path, plot_precision_list)
    save_as_txt(total_k_reg_saving_path, plot_regret_list)
    save_as_txt(total_k_pre_ci_saving_path, ci_precision_list)
    save_as_txt(total_k_reg_ci_saving_path, ci_regret_list)

    plot_list = []
    ci_list = []
    precision_list = []
    precision_list.append(plot_precision_list)
    regre_list = []
    regre_list.append(plot_regret_list)

    precision_ci_list = []
    precision_ci_list.append(ci_precision_list)
    regret_ci_list = []
    regret_ci_list.append(ci_regret_list)

    plot_list.append(precision_list)
    plot_list.append(regre_list)

    ci_list.append(precision_ci_list)
    ci_list.append(regret_ci_list)
    Bvft_plot_folder = os.path.join(Bvft_saving_place,"Bvft_plot")
    if not os.path.exists(Bvft_plot_folder):
        os.makedirs(Bvft_plot_folder)
    y_axis_names = ["k precision","k regret"]
    colors = generate_unique_colors(len(plot_list[0]))
    line_name = ["hopper-medium-expert-v0","hopper-medium-expert-v0"]

    plot_subplots(data=plot_list,save_path=Bvft_plot_folder,y_axis_names=y_axis_names,
                  line_names=line_name,colors=colors,ci=ci_list)


def main():
    parser = argparse.ArgumentParser(description="Run specific Bvft based on learning rate and combination.")
    parser.add_argument("--FQE_episode_step", type=int, default=100000, help="Number of steps in each episode of FQE")
    parser.add_argument("--FQE_Pickup_number", type=int, default=1, help="Number of FQE to pick up")
    parser.add_argument("--FQE_total_step", type=int, default=1000000,
                        help="Maximum step of FQE")
    parser.add_argument("--m", type=int, default=12345,
                        help="Number of policy")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="number of runs")
    args = parser.parse_args()

    run_baseline(FQE_episode_step=args.FQE_episode_step, FQE_Pickup_number = args.FQE_Pickup_number,
                                             FQE_total_step=args.FQE_total_step,
                                            m = args.m,num_runs=args.num_runs)
#--num_interval 5 --FQE_number_epoch 45 --FQE_episode_step 20000 --initial_state 12345 --m 10 --k 1 --num_runs 4
#--FQE_episode_step 100000 --FQE_Pickup_number 1 --FQE_total_step 900000 --m 2 --num_runs 10
if __name__ == "__main__":
    main()