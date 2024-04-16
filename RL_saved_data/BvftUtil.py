import sys
import pickle
import os
from datetime import datetime

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


    def get_state_shape(self):
        first_state = self.dataset.observations[0]
        return np.array(first_state).shape
    def sample(self, length):
        length += 1
        sampled_traj = self.dataset.sample_trajectory(length)
        states = sampled_traj.observations[0:(length-1)]
        actions = sampled_traj.actions[0:(length-1)]
        padded_next_states = sampled_traj.observations[1:length]
        rewards = sampled_traj.rewards[0:(length-1)]
        done = sampled_traj.terminals[0:(length-1)]
        return states, actions, padded_next_states, rewards, done


