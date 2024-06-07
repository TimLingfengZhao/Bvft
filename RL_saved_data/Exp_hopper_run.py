from Policy_selection import *
from Self_defined_method import *
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# whole_dataset, env = get_d4rl('hopper-medium-v2')
#
# train_episodes = whole_dataset.episodes[0:1500]
# test_episodes = whole_dataset.episodes[1500:2186]
whole_dataset, env = get_d4rl('hopper-medium-expert-v0')
print("len : ",len(whole_dataset.episodes))
train_episodes = whole_dataset.episodes[0:2000]
test_episodes = whole_dataset.episodes[2000:2277]

buffer_one = FIFOBuffer(limit=1000000)
replay_buffer_test = ReplayBuffer(buffer=buffer_one, episodes=test_episodes)
Bvft_batch_dim = 1000
test_data = CustomDataLoader(replay_buffer_test, batch_size=Bvft_batch_dim)
k = 5
num_runs = 300
FQE_saving_step_list = [2000000]
initial_state = 12345
# data_saving_path = ["Bvft_ranking","Bvft_res_0","Bvft_abs"]
data_saving_path = ["Bvft_ranking","Bvft_res_0","FQE_0.0001_256","FQE_0.0001_1024","FQE_0.00002_256","FQE_0.00002_1024","l1_norm","arg_i_max_j"]
normalization_factor = 0
# data_saving_path = ["Bvft_ranking"]
gamma = 0.99
common_params = {
    "device": device,
    "data_list": data_saving_path,
    "whole_dataset": whole_dataset,
    "train_episodes": train_episodes,
    "test_episodes": test_episodes,
    "test_data": test_data,
    "replay_buffer": replay_buffer_test,
    "env": env,
    "k": k,
    "num_runs": num_runs,
    "FQE_saving_step_list": FQE_saving_step_list,
    "gamma": gamma,
    "initial_state": initial_state,
    "normalization_factor": normalization_factor
}
bvft_poli = Bvft_poli(data_name_self="Bvft_ranking", **common_params)
bvft_obj = Bvft_zero(data_name_self="Bvft_res_0", **common_params)
FQE_zero = FQE_zero(data_name_self="FQE_0.0001_256", **common_params)
FQE_one = FQE_one(data_name_self="FQE_0.0001_1024", **common_params)
FQE_two = FQE_two(data_name_self="FQE_0.00002_256", **common_params)
FQE_three = FQE_three(data_name_self="FQE_0.00002_1024", **common_params)
l1_norm = Bvft_abs(data_name_self="l1_norm", **common_params)
arg_i_max_j = arg_i_max_j(data_name_self="arg_i_max_j", **common_params)
# bvft_poli = Bvft_poli(device=device,data_list =data_saving_path,data_name_self = "Bvft_ranking",whole_dataset= whole_dataset,train_episodes=train_episodes,
#                      test_episodes=test_episodes,test_data=test_data,replay_buffer=replay_buffer_test,env=env,k=k,
#                      num_runs=num_runs,FQE_saving_step_list=FQE_saving_step_list,
#                  gamma=gamma,initial_state=initial_state,normalization_factor=normalization_factor)
# bvft_obj = Bvft_zero(device=device,data_list =data_saving_path,data_name_self = "Bvft_res_0",whole_dataset= whole_dataset,train_episodes=train_episodes,
#                      test_episodes=test_episodes,test_data=test_data,replay_buffer=replay_buffer_test,env=env,k=k,
#                      num_runs=num_runs,FQE_saving_step_list=FQE_saving_step_list,
#                  gamma=gamma,initial_state=initial_state,normalization_factor=normalization_factor)
# FQE_zero = FQE_zero(device=device,data_list =data_saving_path,data_name_self = "FQE_0.0001_256",whole_dataset= whole_dataset,train_episodes=train_episodes,
#                      test_episodes=test_episodes,test_data=test_data,replay_buffer=replay_buffer_test,env=env,k=k,
#                      num_runs=num_runs,FQE_saving_step_list=FQE_saving_step_list,
#                  gamma=gamma,initial_state=initial_state,normalization_factor=normalization_factor)
# FQE_one = FQE_one(device=device,data_list =data_saving_path,data_name_self = "FQE_0.0001_1024",whole_dataset= whole_dataset,train_episodes=train_episodes,
#                      test_episodes=test_episodes,test_data=test_data,replay_buffer=replay_buffer_test,env=env,k=k,
#                      num_runs=num_runs,FQE_saving_step_list=FQE_saving_step_list,
#                  gamma=gamma,initial_state=initial_state,normalization_factor=normalization_factor)
# FQE_two = FQE_two(device=device,data_list =data_saving_path,data_name_self = "FQE_0.00002_256",whole_dataset= whole_dataset,train_episodes=train_episodes,
#                      test_episodes=test_episodes,test_data=test_data,replay_buffer=replay_buffer_test,env=env,k=k,
#                      num_runs=num_runs,FQE_saving_step_list=FQE_saving_step_list,
#                  gamma=gamma,initial_state=initial_state,normalization_factor=normalization_factor)
# FQE_three = FQE_three(device=device,data_list =data_saving_path,data_name_self = "FQE_0.00002_1024",whole_dataset= whole_dataset,train_episodes=train_episodes,
#                      test_episodes=test_episodes,test_data=test_data,replay_buffer=replay_buffer_test,env=env,k=k,
#                      num_runs=num_runs,FQE_saving_step_list=FQE_saving_step_list,
#                  gamma=gamma,initial_state=initial_state,normalization_factor=normalization_factor)
# l1_norm = Bvft_abs(device=device,data_list =data_saving_path,data_name_self = "l1_norm",whole_dataset= whole_dataset,train_episodes=train_episodes,
#                      test_episodes=test_episodes,test_data=test_data,replay_buffer=replay_buffer_test,env=env,k=k,
#                      num_runs=num_runs,FQE_saving_step_list=FQE_saving_step_list,
#                  gamma=gamma,initial_state=initial_state,normalization_factor=normalization_factor)
# arg_i_max_j = arg_i_max_j(device=device,data_list =data_saving_path,data_name_self = "arg_i_max_j",whole_dataset= whole_dataset,train_episodes=train_episodes,
#                      test_episodes=test_episodes,test_data=test_data,replay_buffer=replay_buffer_test,env=env,k=k,
#                      num_runs=num_runs,FQE_saving_step_list=FQE_saving_step_list,
#                  gamma=gamma,initial_state=initial_state,normalization_factor=normalization_factor)
# bvft_poli.get_self_ranking()
# bvft_obj.get_self_ranking()
# FQE_zero.get_self_ranking()
# FQE_one.get_self_ranking()
# FQE_two.get_self_ranking()
# FQE_three.get_self_ranking()
# l1_norm.get_self_ranking()
arg_i_max_j.get_self_ranking()
arg_i_max_j.draw_figure_6R()
arg_i_max_j.run()
# bvft_obj.draw_figure_6R()
