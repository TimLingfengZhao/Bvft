import sys
from Env_change_util import *

from Math_util import *
# env = gymnasium.make("Hopper-v4")
# print(env.unwrapped.model.opt)  # change this

env = gymnasium.make("Hopper-v4")
# print(env.unwrapped.model.opt)
from top_k_cal import *
# env.unwrapped.model.opt.gravity = np.array([0.0,0.0,-1002])
# print(env.unwrapped.model.opt)
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    ctx._force_start_method('spawn')
    # torch.multiprocessing.set_start_method('fork')
    # ctx._force_start_method('fork')
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
    "max_timestep" : 1000, "total_select_env_number" : 1,
    "env_name" : "Hopper-v4"
    }
    # batch_size = [8,16,32,64,100]
    # batch_size = [200,128,100,64,32]
    # batch_size = [80, 90, 100, 150, 200]
    # process_number = [2,3,4,5,6,7,8, 9]
    batch_size = [100]
    process_number = [9]
    # hopper_exp.generate_offline_data(10,"DDPG",2)
    # hopper_exp.train_policy()
    time_list = []
    result_list = []
    policy_choose = [0,1]
    for i in range(len(policy_choose)):
        # hopper_exp = Bvft_(device=device, parameter_list=parameter_list,
        #                    parameter_name_list=parameter_name_list, policy_training_parameter_map=policy_parameter_map,
        #                    method_name_list=["Bvft"], self_method_name="Bvft_ranking", batch_size=batch_size[0],process_num=process_number[j],**common_params,
        #                    )
        hopper_exp = Bvft_(device=device, parameter_list=parameter_list,
                           parameter_name_list=parameter_name_list, policy_training_parameter_map=policy_parameter_map,
                           method_name_list=["Bvft"], self_method_name="Bvft_ranking", batch_size=batch_size[0],
                           process_num= process_number[0],policy_choose=policy_choose[i]
                           )
        time = hopper_exp.run([0])
        time_list.append(time)
        # result_list.append([batch_size[i], process_number[j], time])
        result_list.append([policy_choose[i], time])
        save_as_txt("Time_result_policy",result_list)
    # for i in range(len(batch_size)):
    #     for j in range(len(process_number)):
    #         hopper_exp = Bvft_(device=device, parameter_list=parameter_list,
    #                            parameter_name_list=parameter_name_list, policy_training_parameter_map=policy_parameter_map,
    #                            method_name_list=["Bvft"], self_method_name="Bvft_ranking", batch_size=batch_size[i],process_num=process_number[j],**common_params,
    #                            )
    #         # hopper_exp = Bvft_(device=device, parameter_list=parameter_list,
    #         #                    parameter_name_list=parameter_name_list, policy_training_parameter_map=policy_parameter_map,
    #         #                    method_name_list=["Bvft"], self_method_name="Bvft_ranking", batch_size=batch_size[i],
    #         #                    process_num= process_number[j]
    #         #                    )
    #         time = hopper_exp.run([0])
    #         time_list.append(time)
    #         print(f"{batch_size[i]} batch size , process number {process_number[j]} ,  runnign time : {time}")
    #         # result_list.append([batch_size[i], process_number[j], time])
    #         result_list.append([policy_choose[i], time])
    #         save_as_txt("Time_result_policy",result_list)
    print("time lsit : ",time_list)