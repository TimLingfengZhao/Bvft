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
# parameter_name_list = ["gravity","magnetic","wind"]
# hopper_exp = Hopper_edi(device=device,parameter_list=parameter_list,
#                         parameter_name_list=parameter_name_list,env_name="Hopper-v4")
# hopper_exp.train_policy()
policy_hidden_list = [64, 256]
policy_rate = 0.0001
ddpg = DDPGConfig(
    actor_encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=policy_hidden_list),
    critic_encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=policy_hidden_list),
    actor_learning_rate=policy_rate,
    critic_learning_rate=policy_rate,
).create(device=device)
buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=1000000, env=env)
explorer = d3rlpy.algos.ConstantEpsilonGreedy(0.3)
ddpg.fit_online(env=env,
                buffer = buffer,
                explorer=explorer,
                n_steps = 300000,
                eval_env = env,
                n_steps_per_epoch=1000,
                update_start_step=1000,
                with_timestamp=False,
                )