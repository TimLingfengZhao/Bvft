import sys
import pickle
import os
from datetime import datetime
import random
import heapq

def sample_files(directory, m):
    all_files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    if (len(all_files)==0):
        print("number of files in sample repo is 0")
    sampled_files = random.sample(all_files, min(m, len(all_files)))

    return sampled_files
class BvftRecord:
    def __init__(self):
        self.resolutions = []
        self.losses = []
        self.loss_matrices = []
        self.group_counts = []
        self.avg_q = []
        self.optimal_grouping_skyline = []
        self.e_q_star_diff = []
        self.bellman_residual = []
        self.ranking = []

    def record_resolution(self, resolution):
        self.resolutions.append(resolution)

    def record_ranking(self,ranking):
        self.ranking = ranking

    def record_losses(self, max_loss):
        self.losses.append(max_loss)

    def record_loss_matrix(self, matrix):
        self.loss_matrices.append(matrix)

    def record_group_count(self, count):
        self.group_counts.append(count)

    def record_avg_q(self, avg_q):
        self.avg_q.append(avg_q)

    def record_optimal_grouping_skyline(self, skyline):
        self.optimal_grouping_skyline.append(skyline)

    def record_e_q_star_diff(self, diff):
        self.e_q_star_diff = diff

    def record_bellman_residual(self, br):
        self.bellman_residual = br

    def save(self, directory="Bvft_Records", file_prefix="BvftRecord_"):
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(directory, f"{file_prefix}.pkl")
        with open(filename, "wb") as file:
            pickle.dump(self, file)
        print(f"Record saved to {filename}")
        return filename

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as file:
            return pickle.load(file)
    def summary(self):
        pass
import numpy as np
class BVFT(object):
    def __init__(self, q_functions, data, gamma, rmax, rmin,file_name_pre, record: BvftRecord = BvftRecord(), q_type='torch_actor_critic_cont',
                 verbose=False, bins=None, data_size=5000,trajectory_num=200):
        self.data = data                                                        #Data D
        self.gamma = gamma                                                      #gamma
        self.res = 0                                                            #\epsilon k (discretization parameter set)
        self.q_sa_discrete = []                                                 #discreate q function
        self.q_to_data_map = []                                                 # to do
        self.q_size = len(q_functions)                                          #how many (s,a) pairs (q function length)
        self.verbose = verbose                                                  #if true, print log
        if bins is None:
            bins = [2, 3, 4, 8, 16, 100, 1e10]
        self.bins = bins                                                        #used for discretizing Q-values
        self.q_sa = []                                                          #all trajectory q s a
        self.r_plus_vfsp = []                                                   #reward
        self.q_functions = q_functions                                          #all q functions
        self.record = record
        self.file_name = file_name_pre

        if q_type == 'tabular':
            self.n = len(data)
            rewards = np.array([t[2] for t in self.data])
            actions = [int(t[1]) for t in self.data]
            states = np.array([t[0] for t in self.data])
            for Q in q_functions:
                self.q_sa.append(np.array([Q[states[i], actions[i]] for i in range(self.n)]))
                vfsp = np.array([0.0 if t[3] is None else np.max(Q[t[3]]) for t in self.data])
                self.r_plus_vfsp.append(rewards + self.gamma * vfsp)   #value after one bellman iteration

        elif q_type == 'keras_standard':
            self.n = len(data)
            rewards = np.array([t[2] for t in self.data])
            actions = [int(t[1]) for t in self.data]
            next_states = np.array([t[3][0] for t in self.data])
            states = np.array([t[0][0] for t in self.data])
            for Q in q_functions:
                qs = Q.predict(states)
                self.q_sa.append(np.array([qs[i][actions[i]] for i in range(self.n)]))
                vfsp = np.max(Q.predict(next_states), axis=1)
                self.r_plus_vfsp.append(rewards + self.gamma * vfsp)

        elif q_type == 'torch_actor_critic_cont':              #minimum batch size
            # batch_size = 1000
            # self.data.batch_size = batch_dim                                #batch size
            self.q_sa = [np.zeros(data_size) for _ in q_functions]             #q_functions corresponding 0
            self.r_plus_vfsp = [np.zeros(data_size) for _ in q_functions]      #initialization 0
            ptr = 0
            while ptr < trajectory_num:                                             #for everything in data size
                length = self.data.get_iter_length(ptr)
                state, action, next_state, reward, done = self.data.sample(ptr)
                # print("state : ",state)
                # print("reward : ", reward)
                # print("next state : ",next_state)
                for i in range(len(q_functions)):
                    actor= q_functions[i]
                    critic= q_functions[i]
                    # self.q_sa[i][ptr:ptr + length] = critic.predict_value(state, action).cpu().detach().numpy().flatten()[
                    #                                  :length]
                    self.q_sa[i][ptr:ptr + length] = critic.predict_value(state, action).flatten()[
                                                     :length]
                    # print("self qa : ",self.q_sa[i][ptr:ptr + 20])
                    # print("done : ",done)
                    # print("reward : ",reward)
                    # print("type state : ",type(state))
                    # print("type next state : ",type(next_state))
                    # print("action : ",actor.predict(next_state))
                    # print("predicted qa value : ",critic.predict_value(next_state, actor.predict(next_state)))
                    vfsp = (reward + critic.predict_value(next_state, actor.predict(next_state)) * done * self.gamma)

                    # self.r_plus_vfsp[i][ptr:ptr + length] = vfsp.cpu().detach().numpy().flatten()[:length]
                    self.r_plus_vfsp[i][ptr:ptr + length] = vfsp.flatten()[:length]
                    # print("self r plus vfsp : ",self.r_plus_vfsp[i][ptr:ptr + 20])
                ptr += 1
            self.n = data_size  #total number of data points

        if self.verbose:
            print(F"Data size = {self.n}")
        self.record.avg_q = [np.sum(qsa) for qsa in self.q_sa]
        self.vmax = np.max(self.q_sa)
        self.vmin = np.min(self.q_sa)


    def discretize(self):                                       #discritization step
        self.q_sa_discrete = []
        self.q_to_data_map = []
        bins = int((self.vmax - self.vmin) / self.res) + 1

        for q in self.q_sa:
            discretized_q = np.digitize(q, np.linspace(self.vmin, self.vmax, bins), right=True) #q belong to which interval
            self.q_sa_discrete.append(discretized_q)
            q_to_data_map = {}
            for i, q_val in enumerate(discretized_q):
                if q_val not in q_to_data_map:
                    q_to_data_map[q_val] = i
                else:
                    if isinstance(q_to_data_map[q_val], int):
                        q_to_data_map[q_val] = [q_to_data_map[q_val]]
                    q_to_data_map[q_val].append(i)
            self.q_to_data_map.append(q_to_data_map)                      #from q value to the position it in discretized_q

    def get_groups(self, q1, q2):
        q1_dic = self.q_to_data_map[q1]
        q2_inds, q2_dic = self.q_sa_discrete[q2], self.q_to_data_map[q2] #dic: indices from q value in the map
        groups = []
        for key in q1_dic:
            if isinstance(q1_dic[key], list):
                q1_list = q1_dic[key]
                set1 = set(q1_list)
                for p1 in q1_list:
                    if p1 in set1 and isinstance(q2_dic[q2_inds[p1]], list):
                        set2 = set(q2_dic[q2_inds[p1]])
                        intersect = set1.intersection(set2)              #intersection
                        set1 = set1.difference(intersect)                #in set1 but not in intersection
                        if len(intersect) > 1:
                            groups.append(list(intersect))               #piecewise constant function
        return groups

    def compute_loss(self, q1, groups):                                 #
        Tf = self.r_plus_vfsp[q1].copy()
        for group in groups:
            Tf[group] = np.mean(Tf[group])
        diff = self.q_sa[q1] - Tf
        return np.sqrt(np.mean(diff ** 2))  #square loss function

    def get_bins(self, groups):
        group_sizes = [len(g) for g in groups]                                  #group size
        bin_ind = np.digitize(group_sizes, self.bins, right=True)               #categorize each group size to bins
        percent_bins = np.zeros(len(self.bins) + 1)    #total group size
        count_bins = np.zeros(len(self.bins) + 1)      #count of groups in each bin
        for i in range(len(group_sizes)):
            count_bins[bin_ind[i] + 1] += 1
            percent_bins[bin_ind[i] + 1] += group_sizes[i]
        percent_bins[0] = self.n - np.sum(percent_bins)
        count_bins[0] = percent_bins[0]    #groups that do not fit into any of predefined bins
        return percent_bins, count_bins

    def run(self, resolution=1e-2):
        self.res = resolution
        if self.verbose:
            print(F"Being discretizing outputs of Q functions on batch data with resolution = {resolution}")
        self.discretize()
        if self.verbose:
            print("Starting pairwise comparison")
        percent_histos = []
        count_histos = []
        group_count = []
        loss_matrix = np.zeros((self.q_size, self.q_size))
        for q1 in range(self.q_size):
            for q2 in range(q1, self.q_size):
                groups = self.get_groups(q1, q2)
                # percent_bins, count_bins = self.get_bins(groups)
                # percent_histos.append(percent_bins)
                # count_histos.append(count_bins)
                group_count.append(len(groups))

                loss_matrix[q1, q2] = self.compute_loss(q1, groups)
                # if self.verbose:
                #     print("loss |Q{}; Q{}| = {}".format(q1, q2, loss_matrix[q1, q2]))

                if q1 != q2:
                    loss_matrix[q2, q1] = self.compute_loss(q2, groups)
                    # if self.verbose:
                    #     print("loss |Q{}; Q{}| = {}".format(q2, q1, loss_matrix[q2, q1]))

        # average_percent_bins = np.mean(np.array(percent_histos), axis=0) / self.n
        # average_count_bins = np.mean(np.array(count_histos), axis=0)
        average_group_count = np.mean(group_count)
        if self.verbose:
            print(np.max(loss_matrix, axis=1))
        self.record.resolutions.append(resolution)
        self.record.losses.append(np.max(loss_matrix, axis=1))
        self.record.loss_matrices.append(loss_matrix)
        # self.record.percent_bin_histogram.append(average_percent_bins)
        # self.record.count_bin_histogram.append(average_count_bins)
        self.get_br_ranking()
        self.record.group_counts.append(average_group_count)
        if not os.path.exists("Bvft_Records"):
            os.makedirs("Bvft_Records")
        self.record.save(directory="Bvft_Records",file_prefix=self.file_name)


    def compute_optimal_group_skyline(self):
        groups = self.get_groups(self.q_size-1, self.q_size-1)
        loss = [self.compute_loss(q, groups) for q in range(self.q_size)]
        self.record.optimal_grouping_skyline.append(np.array(loss))

    def compute_e_q_star_diff(self):
        q_star = self.q_sa[-1]
        e_q_star_diff = [np.sqrt(np.mean((q - q_star) ** 2)) for q in self.q_sa[:-1]] + [0.0]
        self.record.e_q_star_diff = np.array(e_q_star_diff)


    def get_br_ranking(self):
        br = [np.sqrt(np.sum((self.q_sa[q] - self.r_plus_vfsp[q]) ** 2)) for q in range(self.q_size)]
        br_rank = np.argsort(br)
        self.record.bellman_residual = br
        self.record.record_ranking(br_rank)
        return br_rank

class CustomDataLoader:
    def __init__(self, dataset, batch_size=1024):
        self.dataset = dataset
        self.batch_size = batch_size
        self.current = 0
    def __iter__(self):
        self.current = 0
        np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current >= len(self.dataset):
            raise StopIteration
        return self.sample(self.batch_size)
    def __len__(self):
        return len(self.dataset)

    def get_iter_length(self,iteration_number):
        return len(self.dataset.episodes[iteration_number].observations)
    def get_state_shape(self):
        first_state = self.dataset.observations[0]
        return np.array(first_state).shape
    def sample(self, iteration_number):
        states = self.dataset.episodes[iteration_number].observations
        actions =  self.dataset.episodes[iteration_number].actions
        padded_next_states =  self.dataset.episodes[iteration_number].observations[1:len(self.dataset.episodes[iteration_number].observations)]
        np.append(padded_next_states ,self.dataset.episodes[iteration_number].observations[-1])
        rewards = self.dataset.episodes[iteration_number].rewards
        done = self.dataset.episodes[iteration_number].terminated

        return states, actions, padded_next_states, rewards, done





def delete_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print("The folder does not exist.")
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted {file_path}")
            elif os.path.isdir(file_path):
                print(f"Skipping directory {file_path}")
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')



