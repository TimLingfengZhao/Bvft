from Env_change_util import *
import gymnasium
# env = gymnasium.make("Hopper-v4")
# print(env.unwrapped.model.opt)  # change this
env = gymnasium.make("Hopper-medium-v2")
print(env.unwrapped.model.opt)