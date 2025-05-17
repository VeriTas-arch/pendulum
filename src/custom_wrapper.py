import gymnasium as gym
import numpy as np


class PerturbWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.perturbation = np.zeros(self.env.action_space.shape)

    def set_perturbation(self, perturbation):
        self.perturbation = np.array(perturbation)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)  # gymnasium: reset返回(obs, info)
        self.perturbation = np.zeros(self.env.action_space.shape)
        return obs, info  # gymnasium: 返回(obs, info)

    def step(self, action):
        perturbed_action = action + self.perturbation
        obs, reward, terminated, truncated, info = self.env.step(perturbed_action)  # gymnasium: step返回5项
        return obs, reward, terminated, truncated, info
