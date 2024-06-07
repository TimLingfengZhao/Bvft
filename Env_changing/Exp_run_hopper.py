from Env_change_util import *
import gymnasium
# env = gymnasium.make("Hopper-v4")
# print(env.unwrapped.model.opt)  # change this
env = gym.make("Hopper-v4")

env.unwrapped.model.set_opt = np.array([0.0,0.0,-1002])
print(env.unwrapped.model.opt)