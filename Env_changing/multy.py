import time
import concurrent.futures
import numpy as np
import os
import copy
import gymnasium
import d3rlpy
import torch
from top_k_cal import *

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# env = gymnasium.make("Hopper-v4")
# policy_op = "Policy_operation"
# policy_trained = "Policy_trained"
# policy_fo = os.path.join(policy_op, policy_trained)
# policy_folder = os.path.join(policy_fo, "Hopper-v4_gravity_[0.0, 0.0, -4.9]_magnetic_[0.0, 0.0, 0.0]_wind_[10.0, 0.0, 0.0]")
# policy_na = os.path.join(policy_folder, "DDPG_300000_0.0001_[64, 256]_300000step.d3")
# policy = d3rlpy.load_learnable(policy_na, device=device)
#
# def create_deep_copy(env):
#     return copy.deepcopy(env)
#
# def run_simulation(state_action_policy_env):
#     state, action, policy, env = state_action_policy_env
#     total_rewards = 0
#     num_step = 0
#     discount_factor = 1
#     observation = state
#     env.reset()
#     env.observation = observation
#     ui = env.step(action)
#     state = ui[0]
#     reward = ui[1]
#     total_rewards += reward
#     done = ui[2]
#     while not done and num_step < 1000:
#         action = policy.predict(np.array([state]))
#         ui = env.step(action[0])
#         state = ui[0]
#         reward = ui[1]
#         done = ui[2]
#         total_rewards += reward * discount_factor
#         discount_factor *= 0.99
#         num_step += 1
#     return total_rewards
#
# batch_list = [100]
# time_list = []
#
# hun_state = []
# for i in range(1000):
#     hun_state.append([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
# hun_state = np.array(hun_state)
# tot = 0
# for i in range(100):
#     st = time.time()
#     policy.predict(hun_state)
#     en = time.time()
#     tot += (en-st)
# print("tot : ",tot/100)
#
# ten_state = []
# for i in range(100):
#     ten_state.append([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
# ten_state = np.array(ten_state)
# tota = 0
# for i in range(100):
#     sta = time.time()
#     for j in range(10):
#         policy.predict(ten_state)
#     end = time.time()
#     tota += end-sta
# print("tota : ",tota/100)
#
# state = np.array([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]])
# tot_one = 0
# for i in range(100):
#     sta = time.time()
#     for j in range(1000):
#         policy.predict(state)
#     end = time.time()
#     tot_one += end-sta
# print("tota one : ",tot_one / 100)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
env = gymnasium.make("Hopper-v4")
policy_op = "Policy_operation"
policy_trained = "Policy_trained"
policy_fo = os.path.join(policy_op, policy_trained)
policy_folder = os.path.join(policy_fo, "Hopper-v4_gravity_[0.0, 0.0, -4.9]_magnetic_[0.0, 0.0, 0.0]_wind_[10.0, 0.0, 0.0]")
policy_na = os.path.join(policy_folder, "DDPG_300000_0.0001_[64, 256]_300000step.d3")
policy = d3rlpy.load_learnable(policy_na, device=device)

offline_data = "Offline_data"
offlin = os.path.join(offline_data,"DDPG_Hopper-v4_gravity_[0.0, 0.0, -4.9]_magnetic_[0.0, 0.0, 0.0]_wind_[10.0, 0.0, 0.0]_100_maxStep_10_trajectory_0")
data = load_from_pkl(offlin).dataset
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


    # def run_simulation(self, state_action_policy_env):
    #     state, action, policy, env = state_action_policy_env
    #     # total_rewards = [0 for _ in range(len(state))]
    #     total_rewards = 0
    #     num_step = 0
    #     discount_factor = 1
    #     observation = state
    #     env.reset()
    #     env.observation = observation
    #     ui = env.step(action)
    #     state = ui[0]
    #     reward = ui[1]
    #     total_rewards += reward
    #     done = ui[2]
    #     while not done and num_step < self.max_timestep:
    #         action = policy.predict(np.array([state]))
    #         ui = env.step(action[0])
    #         state = ui[0]
    #         reward = ui[1]
    #         done = ui[2]
    #         total_rewards += reward * discount_factor
    #         discount_factor *= self.gamma
    #         num_step += 1
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
    #         state_action_policy_env_pairs = [(states[i+j], actions[i+j], policy, env_copy_list[j]) for j in range(actual_batch_size)]
    #         with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    #             batch_results = list(executor.map(self.run_simulation, state_action_policy_env_pairs))
    #             results.extend(batch_results)
    #     return results


    # def get_qa(self,policy_number,environment_number,states,actions):
    #     env = self.env_list[environment_number]
    #     policy = self.policy_list[policy_number]
    #     result_list = []
    #     for i in range(len(states)):
    #         total_rewards = 0
    #         for j in range(1):
    #             num_step = 0
    #             discount_factor = 1
    #             # print("len states : ",len(states))
    #             # print("len actions : ",len(actions)
    #             observation =states[i]
    #             action = actions[i]
    #             env.reset()
    #             #print("before env.observation : ",observation)
    #             env.observation= observation
    #             # env.state = observation
    #             # print("env.observation : ", env.observation)
    #             # sys.exit()
    #             ui = env.step(action)
    #             state = ui[0]
    #             reward = ui[1]
    #             total_rewards += reward
    #             done = ui[2]
    #             while ((not done) and (num_step < self.max_timestep)):
    #                 action = policy.predict(np.array([state]))
    #                 # print("predicted actioin : ",action)
    #                 ui = env.step(action[0])
    #                 state = ui[0]
    #                 reward = ui[1]
    #                 done = ui[2]
    #                 # print("state=E fr e step : ",state)
    #                 total_rewards += reward * discount_factor
    #                 discount_factor *= self.gamma
    #                 num_step += 1
    #
    #         total_rewards = total_rewards
    #         result_list.append(total_rewards)
    #     return result_list



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
    #         state_action_policy_env_pairs = (states[i:i + actual_batch_size], actions[i:i + actual_batch_size], policy,
    #                                          env_copy_list[:actual_batch_size])
    #         batch_results = self.run_simulation(state_action_policy_env_pairs)
    #         results.extend(batch_results)
    #
    #     return results
    #
    # def create_deep_copies(self, env, batch_size):
    #     return [copy.deepcopy(env) for _ in range(batch_size)]
    #
    # def process_env(self, env_index,  env_copy_lists):
    #     env = self.env_list[env_index]
    #     env_copy_list = env_copy_lists[env_index]
    #     for j in range(len(self.policy_list)):
    #         ptr = 0
    #         trajectory_length = 0
    #         policy_number = j
    #         while ptr < self.trajectory_num:  # for everything in data size
    #             length = self.data.get_iter_length(ptr)
    #             state, action, next_state, reward, done = self.data.sample(ptr)
    #             self.q_sa[(env_index + 1) * len(self.policy_list) + (j + 1) - 1][
    #             trajectory_length:trajectory_length + length] = self.get_qa(policy_number, env_copy_list, state, action,
    #                                                                         self.batch_size)
    #             vfsp = (reward + self.get_qa(policy_number, env_copy_list, next_state,
    #                                          self.policy_list[j].predict(next_state), self.batch_size) * (
    #                                 1 - np.array(done)) * self.gamma)
    #             self.r_plus_vfsp[(env_index + 1) * (j + 1) - 1][
    #             trajectory_length:trajectory_length + length] = vfsp.flatten()[:length]
    #             trajectory_length += length
    #             ptr += 1
    #         self.data_size = trajectory_length
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
    #     if not self.whether_file_exists(data_q_path + ".pkl"):
    #         print("enter get qa calculate loop")
    #         gamma = self.gamma
    #         start_time = time.time()
    #
    #         env_copy_lists = []
    #         for i in range(len(self.env_list)):
    #             env = self.env_list[i]
    #             env_copy_lists.append(self.create_deep_copies(env, self.batch_size))
    #         env_copy_time = time.time()
    #         print(f"full env copy time: {env_copy_time - start_time} for batch size: {self.batch_size}")
    #
    #         with concurrent.futures.ThreadPoolExecutor() as executor:
    #             futures = [executor.submit(self.process_env, i, env_copy_lists) for i in
    #                        range(len(self.env_list))]
    #             for future in as_completed(futures):
    #                 future.result()
    #
    #         end_time = time.time()
    #         print(f"running time get_qa: {self.batch_size}, running time: {end_time - start_time}")
    #         self.save_as_pkl(data_q_path, self.q_sa)
    #         self.save_as_pkl(data_r_path, self.r_plus_vfsp)
    #         self.save_as_pkl(data_size_path, self.data_size)
    #     else:
    #         self.q_sa = self.load_from_pkl(data_q_path)
    #         self.r_plus_vfsp = self.load_from_pkl(data_r_path)
    #         self.data_size = self.load_from_pkl(data_size_path)



    def run_simulation(self, state_action_policy_env_batch):
        states, actions, policy, envs = state_action_policy_env_batch

        # Initialize environments with states
        for env, state in zip(envs, states):
            env.reset()
            env.observation = state

        total_rewards = np.zeros(len(states))
        num_steps = np.zeros(len(states))
        discount_factors = np.ones(len(states))
        done_flags = np.zeros(len(states), dtype=bool)

        while not np.all(done_flags) and np.any(num_steps < self.max_timestep):
            actions_batch = policy.predict(np.array(states))
            for idx, (env, action) in enumerate(zip(envs, actions_batch)):
                if not done_flags[idx]:
                    next_state, reward, done, _, _ = env.step(action)
                    total_rewards[idx] += reward * discount_factors[idx]
                    discount_factors[idx] *= self.gamma
                    num_steps[idx] += 1
                    states[idx] = next_state
                    done_flags[idx] = done

        return total_rewards

    def get_qa(self, policy_number, env_copy_list, states, actions, batch_size=8):
        policy = self.policy_list[policy_number]
        results = []

        total_len = len(states)
        for i in range(0, total_len, batch_size):
            actual_batch_size = min(batch_size, total_len - i)
            state_action_policy_env_pairs = (states[i:i + actual_batch_size], actions[i:i + actual_batch_size], policy,
                                             env_copy_list[:actual_batch_size])
            batch_results = self.run_simulation(state_action_policy_env_pairs)
            results.extend(batch_results)

        return results

    def create_deep_copies(self, env, batch_size):
        return [copy.deepcopy(env) for _ in range(batch_size)]

    def process_env(self, env_index, env_copy_lists):
        logging.info(f"Starting processing for env_index {env_index}")
        start_time = time.time()
        env = self.env_list[env_index]
        env_copy_list = env_copy_lists[env_index]
        for j in range(len(self.policy_list)):
            ptr = 0
            trajectory_length = 0
            policy_number = j
            while ptr < self.trajectory_num:
                length = self.data.get_iter_length(ptr)
                state, action, next_state, reward, done = self.data.sample(ptr)
                self.q_sa[(env_index + 1) * len(self.policy_list) + (j + 1) - 1][
                trajectory_length:trajectory_length + length] = self.get_qa(policy_number, env_copy_list, state, action,
                                                                            self.batch_size)
                vfsp = (reward + self.get_qa(policy_number, env_copy_list, next_state,
                                             self.policy_list[j].predict(next_state), self.batch_size) * (
                                1 - np.array(done)) * self.gamma)
                self.r_plus_vfsp[(env_index + 1) * (j + 1) - 1][
                trajectory_length:trajectory_length + length] = vfsp.flatten()[:length]
                trajectory_length += length
                ptr += 1
            self.data_size = trajectory_length
        end_time = time.time()
        logging.info(f"Finished processing for env_index {env_index} in {end_time - start_time} seconds")

    def get_whole_qa(self, algorithm_index):
        Offline_data_folder = "Offline_data"
        self.create_folder(Offline_data_folder)
        data_folder_name = f"{self.algorithm_name_list[algorithm_index]}_{self.env_name}"
        for j in range(len(self.parameter_list[self.true_env_num])):
            param_name = self.parameter_name_list[j]
            param_value = self.parameter_list[self.true_env_num][j].tolist()
            data_folder_name += f"_{param_name}_{str(param_value)}"
        data_folder_name += f"_{self.max_timestep}_maxStep_{self.trajectory_num}_trajectory_{self.true_env_num}"
        data_q_name = data_folder_name + "_q"
        data_q_path = os.path.join(Offline_data_folder, data_q_name)
        data_r_name = data_folder_name + "_r"
        data_r_path = os.path.join(Offline_data_folder, data_r_name)
        data_size_name = data_folder_name + "_size"
        data_size_path = os.path.join(Offline_data_folder, data_size_name)

        if not self.whether_file_exists(data_q_path + ".pkl"):
            logging.info("Enter get qa calculate loop")
            start_time = time.time()

            env_copy_start_time = time.time()
            env_copy_lists = [self.create_deep_copies(env, self.batch_size) for env in self.env_list]
            env_copy_end_time = time.time()
            logging.info(f"Environment copy time: {env_copy_end_time - env_copy_start_time} seconds")

            active_threads = 0
            lock = threading.Lock()

            def track_active_threads(future):
                nonlocal active_threads
                with lock:
                    active_threads -= 1
                logging.info(f"Active threads: {active_threads}")

            threading_start_time = time.time()
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = [executor.submit(self.process_env, i, env_copy_lists) for i in range(len(self.env_list))]
                for future in futures:
                    with lock:
                        active_threads += 1
                    future.add_done_callback(track_active_threads)
                for future in as_completed(futures):
                    future.result()
            threading_end_time = time.time()
            logging.info(f"Threading (env only) time: {threading_end_time - threading_start_time} seconds")

            end_time = time.time()
            logging.info(f"Total running time get_qa: {end_time - start_time} seconds")

            self.save_as_pkl(data_q_path, self.q_sa)
            self.save_as_pkl(data_r_path, self.r_plus_vfsp)
            self.save_as_pkl(data_size_path, self.data_size)
        else:
            self.q_sa = self.load_from_pkl(data_q_path)
            self.r_plus_vfsp = self.load_from_pkl(data_r_path)
            self.data_size = self.load_from_pkl(data_size_path)