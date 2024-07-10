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
    # def process_env(self, env_index, env_copy_lists):
    #     logging.info(f"Starting processing for env_index {env_index}")
    #     start_time = time.time()
    #     env = self.env_list[env_index]
    #     env_copy_list = env_copy_lists[env_index]
    #     for j in range(len(self.policy_list)):
    #         ptr = 0
    #         trajectory_length = 0
    #         policy_number = j
    #         while ptr < self.trajectory_num:
    #             length = self.data.get_iter_length(ptr)
    #             state, action, next_state, reward, done = self.data.sample(ptr)
    #             self.q_sa[(env_index + 1) * len(self.policy_list) + (j + 1) - 1][
    #             trajectory_length:trajectory_length + length] = self.get_qa(policy_number, env_copy_list, state, action,
    #                                                                         self.batch_size)
    #             vfsp = (reward + self.get_qa(policy_number, env_copy_list, next_state,
    #                                          self.policy_list[j].predict(next_state), self.batch_size) * (
    #                             1 - np.array(done)) * self.gamma)
    #             self.r_plus_vfsp[(env_index + 1) * (j + 1) - 1][
    #             trajectory_length:trajectory_length + length] = vfsp.flatten()[:length]
    #             trajectory_length += length
    #             ptr += 1
    #         self.data_size = trajectory_length
    #     end_time = time.time()
    #     logging.info(f"Finished processing for env_index {env_index} in {end_time - start_time} seconds")
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
    #         env_copy_lists = [self.create_deep_copies(env, self.batch_size) for env in self.env_list]
    #         env_copy_end_time = time.time()
    #         logging.info(f"Environment copy time: {env_copy_end_time - env_copy_start_time} seconds")
    #
    #         active_threads = 0
    #         lock = threading.Lock()
    #
    #         def track_active_threads(future):
    #             nonlocal active_threads
    #             with lock:
    #                 active_threads -= 1
    #             logging.info(f"Active threads: {active_threads}")
    #
    #         threading_start_time = time.time()
    #         with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    #             futures = [executor.submit(self.process_env, i, env_copy_lists) for i in range(len(self.env_list))]
    #             for future in futures:
    #                 with lock:
    #                     active_threads += 1
    #                 future.add_done_callback(track_active_threads)
    #             for future in as_completed(futures):
    #                 future.result()
    #         threading_end_time = time.time()
    #         logging.info(f"Threading (env only) time: {threading_end_time - threading_start_time} seconds")
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
    # def create_deep_copies(self, env, batch_size):
    #     return [copy.deepcopy(env) for _ in range(batch_size)]
    #
    # @profile
    # def process_env(self, env_index, policy_index, env_copy_lists_name, policy_list_name, data_name, env_list_name,
    #                 q_sa_name, r_plus_vfsp_name, q_sa_shape, r_plus_vfsp_shape):
    #     start_time = time.time()
    #     logging.info(f"Starting processing for env_index {env_index}, policy_index {policy_index}")
    #
    #     # Attach to existing shared memory blocks
    #     shm_env_copy_lists = shared_memory.SharedMemory(name=env_copy_lists_name)
    #     env_copy_lists = dill.loads(bytes(shm_env_copy_lists.buf))
    #
    #     shm_policy_list = shared_memory.SharedMemory(name=policy_list_name)
    #     policy_list = dill.loads(bytes(shm_policy_list.buf))
    #
    #     shm_data = shared_memory.SharedMemory(name=data_name)
    #     data = dill.loads(bytes(shm_data.buf))
    #
    #     shm_env_list = shared_memory.SharedMemory(name=env_list_name)
    #     env_list = dill.loads(bytes(shm_env_list.buf))
    #
    #     # Attach to shared memory for q_sa and r_plus_vfsp
    #     shm_q_sa = shared_memory.SharedMemory(name=q_sa_name)
    #     q_sa = np.ndarray(q_sa_shape, dtype=np.float64, buffer=shm_q_sa.buf)
    #
    #     shm_r_plus_vfsp = shared_memory.SharedMemory(name=r_plus_vfsp_name)
    #     r_plus_vfsp = np.ndarray(r_plus_vfsp_shape, dtype=np.float64, buffer=shm_r_plus_vfsp.buf)
    #
    #     env = env_list[env_index]
    #     env_copy_list = env_copy_lists[env_index][policy_index]
    #     policy = policy_list[policy_index]
    #     ptr = 0
    #     trajectory_length = 0
    #
    #     while ptr < data.length:
    #         length = data.get_iter_length(ptr)
    #         state, action, next_state, reward, done = data.sample(ptr)
    #
    #         q_sa[(env_index + 1) * len(policy_list) + (policy_index + 1) - 1][
    #         trajectory_length:trajectory_length + length] = self.get_qa(policy_index, env_copy_list, state, action,
    #                                                                     self.batch_size)
    #
    #         vfsp = (reward + self.get_qa(policy_index, env_copy_list, next_state,
    #                                      policy.predict(next_state), self.batch_size) * (
    #                         1 - np.array(done)) * self.gamma)
    #         r_plus_vfsp[(env_index + 1) * (policy_index + 1) - 1][
    #         trajectory_length:trajectory_length + length] = vfsp.flatten()[:length]
    #         trajectory_length += length
    #         ptr += 1
    #
    #         # Manually trigger garbage collection
    #         del state, action, next_state, reward, done, vfsp
    #         gc.collect()
    #
    #     self.data_size = trajectory_length
    #
    #     end_time = time.time()
    #     logging.info(
    #         f"Finished processing for env_index {env_index}, policy_index {policy_index} in {end_time - start_time} seconds")
    #
    #     # Cleanup shared memory in worker
    #     shm_env_copy_lists.close()
    #     shm_policy_list.close()
    #     shm_data.close()
    #     shm_env_list.close()
    #     shm_q_sa.close()
    #     shm_r_plus_vfsp.close()
    #
    #     # Force garbage collection
    #     del env_copy_lists, policy_list, data, env_list, q_sa, r_plus_vfsp
    #     gc.collect()
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
    #         env_copy_lists = np.array(
    #             [[self.create_deep_copies(env, self.batch_size) for env in self.env_list] for _ in
    #              self.policy_list],
    #             dtype=object)
    #         env_copy_serialized = dill.dumps(env_copy_lists)
    #         env_copy_shm = shared_memory.SharedMemory(create=True, size=len(env_copy_serialized))
    #         env_copy_shm.buf[:len(env_copy_serialized)] = env_copy_serialized
    #
    #         policy_list_serialized = dill.dumps(self.policy_list)
    #         policy_list_shm = shared_memory.SharedMemory(create=True, size=len(policy_list_serialized))
    #         policy_list_shm.buf[:len(policy_list_serialized)] = policy_list_serialized
    #
    #         data_serialized = dill.dumps(self.data)  # Serialize the entire data object
    #         data_shm = shared_memory.SharedMemory(create=True, size=len(data_serialized))
    #         data_shm.buf[:len(data_serialized)] = data_serialized
    #
    #         env_list_serialized = dill.dumps(self.env_list)
    #         env_list_shm = shared_memory.SharedMemory(create=True, size=len(env_list_serialized))
    #         env_list_shm.buf[:len(env_list_serialized)] = env_list_serialized
    #
    #         env_copy_end_time = time.time()
    #         logging.info(f"Environment copy time: {env_copy_end_time - env_copy_start_time} seconds")
    #
    #         # Convert lists to numpy arrays
    #         q_sa_array = np.array(self.q_sa, dtype=object)
    #         r_plus_vfsp_array = np.array(self.r_plus_vfsp, dtype=object)
    #         q_sa_shape = q_sa_array.shape
    #         r_plus_vfsp_shape = r_plus_vfsp_array.shape
    #
    #         # Initialize shared memory for q_sa and r_plus_vfsp
    #         q_sa_shm = shared_memory.SharedMemory(create=True, size=q_sa_array.nbytes)
    #         r_plus_vfsp_shm = shared_memory.SharedMemory(create=True, size=r_plus_vfsp_array.nbytes)
    #
    #         q_sa = np.ndarray(q_sa_shape, dtype=object, buffer=q_sa_shm.buf)
    #         r_plus_vfsp = np.ndarray(r_plus_vfsp_shape, dtype=object, buffer=r_plus_vfsp_shm.buf)
    #
    #         # Initialize shared memory arrays with the existing data
    #         np.copyto(q_sa, q_sa_array)
    #         np.copyto(r_plus_vfsp, r_plus_vfsp_array)
    #
    #         # Determine total memory and set maximum memory usage to 90% of it
    #         total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Total memory in GB
    #         max_memory_usage = total_memory * 0.7  # Set to 70% of total memory
    #         logging.info(f"Total memory: {total_memory} GB, Max memory usage: {max_memory_usage} GB")
    #
    #         threading_start_time = time.time()
    #         with Pool(processes=1) as pool:
    #             results = [pool.apply_async(self.process_env, args=(
    #                 i, j, env_copy_shm.name, policy_list_shm.name, data_shm.name, env_list_shm.name, q_sa_shm.name, r_plus_vfsp_shm.name, q_sa_shape, r_plus_vfsp_shape)) for i in
    #                        range(len(self.env_list))
    #                        for j in range(len(self.policy_list))]
    #
    #             # Memory usage check loop
    #             while results:
    #                 if self.get_memory_usage() > max_memory_usage:
    #                     logging.info(
    #                         f"Memory usage exceeded {max_memory_usage}GB, waiting for tasks to complete...")
    #                     results.pop(0).get()  # Wait for the first task in the list to complete
    #                 else:
    #                     break  # Exit loop if memory usage is within limits
    #
    #             # Ensure all tasks are completed
    #             for result in results:
    #                 result.get()
    #
    #             pool.close()
    #             pool.join()
    #         threading_end_time = time.time()
    #         logging.info(f"Threading (env-policy) time: {threading_end_time - threading_start_time} seconds")
    #
    #         end_time = time.time()
    #         logging.info(f"Total running time get_qa: {end_time - start_time} seconds")
    #
    #         # Convert shared memory arrays back to lists
    #         self.q_sa = q_sa.tolist()
    #         self.r_plus_vfsp = r_plus_vfsp.tolist()
    #
    #         self.save_as_pkl(data_q_path, self.q_sa)
    #         self.save_as_pkl(data_r_path, self.r_plus_vfsp)
    #         self.save_as_pkl(data_size_path, self.data_size)
    #
    #         # Cleanup shared memory
    #         env_copy_shm.close()
    #         env_copy_shm.unlink()
    #         policy_list_shm.close()
    #         policy_list_shm.unlink()
    #         data_shm.close()
    #         data_shm.unlink()
    #         env_list_shm.close()
    #         env_list_shm.unlink()
    #         q_sa_shm.close()
    #         q_sa_shm.unlink()
    #         r_plus_vfsp_shm.close()
    #         r_plus_vfsp_shm.unlink()
    #
    #         # Force garbage collection
    #         gc.collect()
    #     else:
    #         self.q_sa = self.load_from_pkl(data_q_path)
    #         self.r_plus_vfsp = self.load_from_pkl(data_r_path)
    #         self.data_size = self.load_from_pkl(data_size_path)
    #
    # def get_memory_usage(self):
    #     # 获取当前进程的内存使用情况（以GB为单位）
    #     process = psutil.Process()
    #     mem_info = process.memory_info()
    #     return mem_info.rss / (1024 ** 3)  # 转换为GB


