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
    torch.multiprocessing.set_start_method('spawn')
    ctx._force_start_method('spawn')
    gravity = [np.array([0.0, 0.0, -9.8]), np.array([0.0, 0.0, -4.9]), np.array([0.0, 0.0, -15.1])]
    magnetic = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    wind = [np.array([10.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 10.0, 0.0])]

    parameter_list = [
        [gravity[1], magnetic[0], wind[0]],
        [gravity[2], magnetic[0], wind[0]],
        [gravity[0], magnetic[2], wind[0]]
    ]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # print("device : ",device)
    # sys.exit()
    policy_parameter_map = {"policy_total_step":300000,
                     "policy_episode_step":10000,
                            "policy_saving_number" : 30,"policy_learning_rate":0.0001,"policy_hidden_layer":[64,256],
                            "algorithm_name_list":["DDPG"]}
    parameter_name_list = ["gravity","magnetic","wind"]

    common_params = {
        "gamma": 0.99, "trajectory_num" : 200,
    "max_timestep" : 1000, "total_select_env_number" : 2,
    "env_name" : "Hopper-v4"
    }
    # batch_size = [8,16,32,64,100]
    # batch_size = [200,128,100,64,32]
    batch_size = [100, 150, 200]
    # hopper_exp.generate_offline_data(10,"DDPG",2)
    # hopper_exp.train_policy()
    time_list = []

    for i in range(len(batch_size)):
        # hopper_exp = Bvft_(device=device, parameter_list=parameter_list,
        #                    parameter_name_list=parameter_name_list, policy_training_parameter_map=policy_parameter_map,
        #                    method_name_list=["Bvft"], self_method_name="Bvft_ranking", batch_size=batch_size[i],**common_params,
        #                    )
        hopper_exp = Bvft_(device=device, parameter_list=parameter_list,
                           parameter_name_list=parameter_name_list, policy_training_parameter_map=policy_parameter_map,
                           method_name_list=["Bvft"], self_method_name="Bvft_ranking", batch_size=batch_size[i],
                           )
        time = hopper_exp.run([0])
        time_list.append(time)
        print(f"{batch_size[i]} batch size runnign time : {time}")
    print("time lsit : ",time_list)