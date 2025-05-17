import gym
import matplotlib.pyplot as plt
from Model import *
from utils import *

### resolve matplotlib error
resolve_matplotlib_error()
plt.ion()

### Train
env = gym.make("InvertedDoublePendulum-v4", render_mode="human")
agent = SACAgent(env=env)

epochs = 1000  # int(input("EPOCHS: "))
trained_agent = agent.train(epochs)

showcase(trained_agent)

plt.ioff()
