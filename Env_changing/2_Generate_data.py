import sys
from Env_change_util import *

from Math_util import *
# env = gymnasium.make("Hopper-v4")
# print(env.unwrapped.model.opt)  # change this

env = gymnasium.make("Hopper-v4")
# print(env.unwrapped.model.opt)
from top_k_cal import *
if __name__ == '__main__':
    gravity = [np.array([0.0, 0.0, -9.8]), np.array([0.0, 0.0, -4.9]), np.array([0.0, 0.0, -15.1])]
    magnetic = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    wind = [np.array([10.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 10.0, 0.0])]
    bahavioral_policy_map =    {"policy_total_step":5000,
                     "policy_episode_step":1000,
                            "policy_saving_number" : 5,"policy_learning_rate":0.0001,"policy_hidden_layer":[64,1024],
                            "algorithm_name_list":["DDPG"]}

    env_parameter_map = {"env_name" : "Hopper-v4",
                         "parameter_list":[[
        [gravity[1], magnetic[0], wind[0]],
        [gravity[2], magnetic[0], wind[0]],
        [gravity[0], magnetic[2], wind[0]]]],
                         "parameter_name_list":[["gravity","magnetic","wind"]]}
    hopper_exp = Bvft_()
    trajectory_number = 10
    trajectory_max_timestep = 1000
    hopper_exp.generate_offline_data(trajectory_number=trajectory_number,true_environment_parameter_list= env_parameter_map,
                                     behaviroal_policy_parameter_map=bahavioral_policy_map,
                                     Offline_trajectory_max_timestep= trajectory_max_timestep)