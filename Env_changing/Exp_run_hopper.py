from Env_change_util import *

# env = gymnasium.make("Hopper-v4")
# print(env.unwrapped.model.opt)  # change this
env = gymnasium.make("Hopper-v4")
print(env.unwrapped.model.opt)
env.unwrapped.model.opt.gravity = np.array([0.0,0.0,-1002])
print(env.unwrapped.model.opt)
parameter_list =[[np.array([0.0,0.0,-4.9]),
                  np.array([0.0,0.0,20.0]),
                  np.array([0.0,0.0,2.0])],
                 [np.array([0.0, 0.0, -15.1]),
                  np.array([0.0, 0.0, 20.0]),
                  np.array([0.0, 0.0, 2.0])],
                    [np.array([0.0, 0.0, -15.1]),
                  np.array([0.0, 0.0, 24.0]),
                  np.array([0.0, 0.0, 10.0])]
                 ]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
policy_parameter_map = {"policy_total_step":3000,
                 "policy_episode_step":1000,
                        "policy_saving_number" : 3,"policy_learning_rate":0.0001,"policy_hidden_layer":[64,256],
                        "algorithm_name_list":["DDPG","SAC"]}



parameter_name_list = ["gravity","magnetic","wind"]
hopper_exp = Hopper_edi(device=device,parameter_list=parameter_list,
                        parameter_name_list=parameter_name_list,policy_training_parameter_map=policy_parameter_map,
                        env_name="Hopper-v4")
hopper_exp.train_policy()

