import sys
from Env_change_util import *

from Math_util import *
# env = gymnasium.make("Hopper-v4")
# print(env.unwrapped.model.opt)  # change this

env = gymnasium.make("Hopper-v4")
# print(env.unwrapped.model.opt)
# env.unwrapped.model.opt.gravity = np.array([0.0,0.0,-1002])
# print(env.unwrapped.model.opt)
if __name__ == '__main__':
    gravity = [np.array([0.0, 0.0, -9.8]), np.array([0.0, 0.0, -4.9]), np.array([0.0, 0.0, -15.1])]
    magnetic = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    wind = [np.array([10.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 10.0, 0.0])]
    policy_parameter_map = {"policy_total_step":3000,
                     "policy_episode_step":1000,
                            "policy_saving_number" : 3,"policy_learning_rate":0.0001,"policy_hidden_layer":[64,256],
                            "algorithm_name_list":["DDPG"]}
    bahavioral_policy_map =    {"policy_total_step":5000,
                     "policy_episode_step":1000,
                            "policy_saving_number" : 5,"policy_learning_rate":0.0001,"policy_hidden_layer":[64,1024],
                            "algorithm_name_list":["DDPG"]}
    policy_evaluation_parameter_map = {"evaluate_time" : 30,
    "max_timestep" : 1000,
    "gamma" : 0.99}


    env_parameter_map = {"env_name" : "Hopper-v4",
                         "parameter_list":[[
        [gravity[1], magnetic[0], wind[0]],
        [gravity[2], magnetic[0], wind[0]],
        [gravity[0], magnetic[2], wind[0]]]],
                         "parameter_name_list":[["gravity","magnetic","wind"]]}
    hopper_exp = Bvft_()
    hopper_exp.train_policy(env_parameter_map=env_parameter_map,target_policy_training_parameter_map=policy_parameter_map)
    hopper_exp.train_policy(env_parameter_map=env_parameter_map,target_policy_training_parameter_map=bahavioral_policy_map)
    hopper_exp.train_policy_performance(env_parameter_map=env_parameter_map,policy_parameter_map=policy_parameter_map,policy_evaluation_parameter_map=policy_evaluation_parameter_map)
    hopper_exp.train_policy_performance(env_parameter_map=env_parameter_map,policy_parameter_map=bahavioral_policy_map,policy_evaluation_parameter_map=policy_evaluation_parameter_map)