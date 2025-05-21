import gymnasium as gym
import numpy as np


class PerturbWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.perturbation = np.zeros(self.env.action_space.shape)

    def set_perturbation(self, perturbation):
        self.perturbation = np.array(perturbation)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.perturbation = np.zeros(self.env.action_space.shape)
        return obs, info

    def step(self, action):
        perturbed_action = action + self.perturbation
        obs, reward, terminated, truncated, info = self.env.step(perturbed_action)
        return obs, reward, terminated, truncated, info
