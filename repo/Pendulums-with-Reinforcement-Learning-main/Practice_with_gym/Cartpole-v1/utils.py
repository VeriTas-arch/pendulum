import random
from collections import deque

import numpy as np
import torch


class OUNoise:
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=1e6):
        self.mu = mu
        self.theta = theta
        self.sigma = max
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        rand = np.random.randn(self.action_dim)
        dx = self.theta * (self.mu - x) + rand
        self.state = x + dx
        return self.state
    
    def process_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        noised_action = action.detach().numpy() + ou_state
        return torch.tensor(np.clip(noised_action, self.low, self.high))


class ExperienceReplay:
    def __init__(self, max_length, batch_size):
        self.max_length = max_length
        self.batch_size = batch_size
        
        self.buffer = deque(maxlen=self.max_length)

    def append(self, state, action, reward, next_state, done):
        exp = (state, action, reward, next_state, done)
        self.buffer.append(exp)

    def sample(self):

        minibatch = random.sample(self.buffer, self.batch_size)

        state_batch = torch.cat([s1.view(1, -1) for (s1, a, r, s2, d) in minibatch], dim=0)
        action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch]).view(-1, 1)
        reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch]).view(-1, 1)
        next_state_batch = torch.cat([s2.view(1, -1) for (s1, a, r, s2, d) in minibatch])
        done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch]).view(-1, 1)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def __len__(self):
        return len(self.buffer)
    
def matplotlib_error():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'