# FQE_load_fit   #open RL saved data
1. use pip import d3rlpy, scope_rl
2. mujoco: 2.3.3 mujoco-py:2.1.2.14  https://blog.guptanitish.com/guide-to-install-openais-mujoco-on-ubuntu-linux-1ac22a9678b4
3. mujoco helpful (debug) website: https://github.com/openai/mujoco-py/issues/410
4. https://www.reddit.com/r/Ubuntu/comments/rmz3mn/why_my_export_path_doesnt_work_mujoco_gcc_error/?rdt=40047
5. import typing, pandas, time, pickle, numpy, torch
6. install cuda : https://www.cherryservers.com/blog/install-cuda-ubuntu (self define the driver version) install cuDNN with following command :
conda install -c nvidia cuda-nvcc
conda install -c "nvidia/label/cuda-11.3.0" cuda-nvcc    #self define cuda version based on the detailed information given by:  nvidia-smi
7. run the script "FQE_load_two.py" directly.
8. The data will saved in FQE_load_2, remember to change the rate of saving Q-functions in the bottom of FQE_load_two, also change the saving picture name so it will not substitute original one
9. See pictures in k_figure

Changes to get environment setting:
My setting process:
1. add
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
to the package file (they said there no contrib)

2. gym.envs.registry was previously a complex class that we replaced with a dictionary for simplicity. 
if some error happend in : site-packages/pybullet_envs/__init__.py", line 6, in register
    if env_id not in gym.envs.registry:
The code should just need to be changed to if id not in gym.envs.registry

meet something like gym.env.registry fault. replace the if code to  if env_id not in gym.envs.registry

Training:
1. Get data from d4rl, hopper-v2-expert
2. Train policys with different parameters:
policy_hidden_size = [64, 1024]
policy_hidden_layer = [2,3]
policy_learning_rate = [0.001,0.00002]
policy_learning_steps = [300000]
policy_algorithm_name =  ["bcq","cql"]
policy_episode_step = 10000
3. Fit trained policies to FQE with four combinations:
learning rate 0.0001
learning rate 0.00002
Hidden layer [128,256]
Hidden layer [128,1024]
4. Feed the trained Q function to Bvft, calculate k-regret and k-precision (After generated enough policy pool)
5. Construct LSTDQ , feed trained Q function to Bvft, calculate k-regret and k precision
6. Compare results

[Direct to RL_saved_data directory]
how to train FQE: cd to FQE_simple_train/RL_saved_data  0.0001,[128,256]
python FQE_train FQE_1 --FQE_total_step 2000000 --FQE_episode_step 100000



how to train FQE: cd to FQE_simple_train/RL_saved_data  0.0001,[128,1024]
python FQE_train FQE_2 --FQE_total_step 2000000 --FQE_episode_step 100000


how to train FQE: cd to FQE_simple_train/RL_saved_data  0.00002,[128,256]
python FQE_train FQE_3 --FQE_total_step 2000000 --FQE_episode_step 100000


how to train FQE: cd to FQE_simple_train/RL_saved_data  0.00002,[128,1024]
python FQE_train FQE_4 --FQE_total_step 2000000 --FQE_episode_step 100000

Plot FQE performance:
python FQE_performance_easy.py --FQE_total_step 2000000 --FQE_episode_step 100000

How to plot Policy performance:
python Policy_performance_easy.py --input_gamma 0.99 --num_runs 100

How to plot FQE NMSE plot:
python FQE_MSE_Draw.py --FQE_total_step 2000000 --FQE_episode_step 100000 --num_interval 1  plot will be saved to "FQE_returned_result" folder

How to plot FQE NMSE plot(for each saved intervel):
python FQE_MSE_Draw_split.py --FQE_total_step 2000000 --FQE_episode_step 100000 --num_interval 1  plot will be saved to "FQE_returned_result" folder

Remember: FQE_MSE_Draw,FQE_performance_easy,FQE_train,Policy_performance_easy will train dynamicly and you will need to stop it 

plot FQE prediction performance vs epoch
python FQE_epoch_prediction.py --FQE_total_step 2000000 --FQE_episode_step 100000 --input_gamma 0.99 --num_runs 100
