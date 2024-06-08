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
hopper_exp = Hopper_edi(device=device,parameter_list=parameter_list,env_name="Hopper-v4")