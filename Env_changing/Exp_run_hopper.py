import sys
from Env_change_util import *
from Math_util import *
from top_k_cal import *
from Run_hopper_parameter import *

if __name__ == '__main__':
    hopper_exp = Bvft_()

    # 第一步：训练策略
    hopper_exp.train_policy(env_parameter_map=env_parameter_map,
                            target_policy_training_parameter_map=policy_parameter_map)
    hopper_exp.train_policy(env_parameter_map=env_parameter_map,
                            target_policy_training_parameter_map=behavioral_policy_map)

    # 第二步：训练策略性能
    hopper_exp.train_policy_performance(env_parameter_map=env_parameter_map, policy_parameter_map=policy_parameter_map,
                                        policy_evaluation_parameter_map=policy_evaluation_parameter_map)
    hopper_exp.train_policy_performance(env_parameter_map=env_parameter_map, policy_parameter_map=behavioral_policy_map,
                                        policy_evaluation_parameter_map=policy_evaluation_parameter_map)

    # 第三步：生成离线数据
    trajectory_number = 10
    trajectory_max_timestep = 1000
    hopper_exp.generate_offline_data(trajectory_number=trajectory_number,
                                     true_environment_parameter_list=env_parameter_map,
                                     behaviroal_policy_parameter_map=behavioral_policy_map,
                                     Offline_trajectory_max_timestep=trajectory_max_timestep)

    # 第四步：训练QA
    offline_data_name_list = ["10_trajectory_20240720-163657.pkl"]
    hopper_exp.train_whole_qa(offline_trajectory_name_list=offline_data_name_list,
                              behavioral_env_parameter_map=env_parameter_map,
                              behavioral_policy_parameter_map=behavioral_policy_map,
                              target_env_parameter_map=env_parameter_map, target_parameter_map=policy_parameter_map,
                              batch_size=batch_size, max_timestep=max_timestep, gamma=gamma)

    # 第五步：获取排名
    algorithm_trajectory_list = [behavioral_policy_map, ["10_trajectory_20240720-163657.pkl"]]
    true_env_list, true_env_name_list = hopper_exp.get_env_list(true_env_parameter_map)
    experiment_name_list = []
    for i in range(len(true_env_name_list)):
        hopper_exp.get_ranking(experiment_name=str(i) + "_" + experiment_dataset_name,
                               ranking_method_name=ranking_method_name,
                               algorithm_trajectory_list=algorithm_trajectory_list, true_env_name=true_env_name_list[i],
                               target_env_parameter_map=env_parameter_map,
                               target_policy_parameter_map=policy_parameter_map, gamma=gamma)
        experiment_name_list.append(str(i) + "_" + experiment_dataset_name)

    # 第六步：绘制图表
    hopper_exp.draw_figure_6L(saving_folder_name="experiment_3env_3policy", experiment_name_list=experiment_name_list,
                              method_name_list=["BVFT"], k=k, **policy_evaluation_parameter_map)
