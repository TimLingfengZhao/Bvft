from Env_change_util import *
from Math_util import *
# env = gymnasium.make("Hopper-v4")
# print(env.unwrapped.model.opt)  # change this
env = gymnasium.make("Hopper-v4")
# print(env.unwrapped.model.opt)
# env.unwrapped.model.opt.gravity = np.array([0.0,0.0,-1002])
# print(env.unwrapped.model.opt)
gravity = [np.array([0.0, 0.0, -9.8]), np.array([0.0, 0.0, -4.9]), np.array([0.0, 0.0, -15.1])]
magnetic = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])]
wind = [np.array([10.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 10.0, 0.0])]

parameter_list = [
    [gravity[1], magnetic[0], wind[0]],
    [gravity[2], magnetic[0], wind[0]],
    [gravity[0], magnetic[2], wind[0]]
]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
policy_parameter_map = {"policy_total_step":30000,
                 "policy_episode_step":1000,
                        "policy_saving_number" : 30,"policy_learning_rate":0.0001,"policy_hidden_layer":[64,256],
                        "algorithm_name_list":["DDPG"]}
parameter_name_list = ["gravity","magnetic","wind"]

common_params = {
    "gamma": 0.99, "trajectory_num" : 200,
"max_timestep" : 1000, "total_select_env_number" : 2,
"env_name" : "Hopper-v4"
}

hopper_exp = Bvft_(device=device,parameter_list=parameter_list,
                        parameter_name_list=parameter_name_list,policy_training_parameter_map=policy_parameter_map,
                        env_name="Hopper-v4")

# hopper_exp.generate_offline_data(10,"DDPG",2)
# hopper_exp.train_policy()
hopper_exp.run([0,1])