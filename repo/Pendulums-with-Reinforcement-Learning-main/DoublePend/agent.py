from collections import deque

import gym
import matplotlib.pyplot as plt
import user_env_gym.double_pendulum as dpend
from Model import *
from tqdm import tqdm
from utils import *

### resolve matplotlib error
resolve_matplotlib_error()
plt.ion()

# create env
env = dpend.DoublePendEnv(reward_mode=3)

# initialize agent, set epoch length
agent = SACAgent(env=env)
epochs = int(input("EPOCHS: "))

# train the agent
save = bool(input("Save Agent? (True or False): "))
scores = agent.train(epochs, save=save)
print("------Training Completed------")

# turn off plotting interactive mode
plt.ioff()
plt.plot(agent.save_epi_reward)

# plot moving average 
final_plot(scores)

# see how the trained agent performs
showcase(agent, env=dpend.DoublePendEnv(reward_mode=3, render_mode='human'))