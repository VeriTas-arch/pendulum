import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_matplotlib_error():
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def showcase(actor, env):

    n_episodes = 3
    for _ in range(n_episodes):
        state, _ = env.reset()

        terminated = False
        truncated = False

        while not terminated and not truncated:
            state = torch.tensor(state, dtype=torch.float32)
            action = actor.get_action(state, deterministic=True)
            action = np.clip(action, -actor.action_bound, actor.action_bound)
            action = np.array(action).reshape(-1)
            state, reward, terminated, truncated, _ = env.step(action)

    env.close()


def final_plot(g1):
    import numpy as np

    resolve_matplotlib_error()
    plt.ioff()

    window_size = 10
    moving_average = np.convolve(g1, np.ones(window_size) / window_size, mode="valid")

    plt.plot(moving_average, label="mAverage", color="red", linewidth=2.5)
    plt.legend()
    plt.draw()
    plt.show()


def live_plot(scores):

    plt.plot(scores)
    plt.grid(True)
    plt.draw()
    plt.gca().grid(True)
    plt.xlabel("episodes")
    plt.ylabel("scores")
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
