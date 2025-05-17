import random
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_matplotlib_error():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def showcase(model):
    n_episodes = 3
    env = gym.make('InvertedDoublePendulum-v4', render_mode='human')

    for episode in range(n_episodes):
        state, _ = env.reset()

        terminated = False
        truncated = False

        while not terminated and not truncated:
            state = torch.tensor(state, dtype=torch.float32)
            action = model.get_action(state)
            action = np.clip(action, -model.action_bound, model.action_bound)
            state, reward, terminated, truncated, _ = env.step(action)
            print(reward)
            
            
    env.close() 


def test_env(env):

    n_episodes = 1
    for episode in range(n_episodes):
        state, _ = env.reset()

        terminated = False
        truncated = False
        action = env.action_space.sample()

        while not terminated and not truncated:
            state = torch.tensor(state, dtype=torch.float32)
            state, reward, terminated, truncated, _ = env.step(action)
            print(reward)
            
            
    env.close() 


def live_plot(g1, g2):
    plt.clf()

    plt.subplot(2,1,1)
    plt.plot(g1)
    plt.xlabel('epochs')
    plt.ylabel('score')

    plt.subplot(2,1,2)
    plt.plot(g2)
    plt.xlabel('timesteps')
    plt.ylabel('action')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
    
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0

    def add_buffer(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        # 버퍼가 꽉 찼는지 확인
        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else: 
            self.buffer.popleft()
            self.buffer.append(transition)

    def sample_batch(self, batch_size):
        random.shuffle(self.buffer)
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        states = np.asarray([i[0] for i in batch])
        actions = np.asarray([i[1] for i in batch])
        rewards = np.asarray([i[2] for i in batch])
        next_states = np.asarray([i[3] for i in batch])
        dones = np.asarray([i[4] for i in batch])
        return states, actions, rewards, next_states, dones

    def buffer_count(self):
        return self.count

    def clear_buffer(self):
        self.buffer = deque()
        self.count = 0
