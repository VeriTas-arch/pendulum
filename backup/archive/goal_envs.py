import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.inverted_double_pendulum_v5 import \
    InvertedDoublePendulumEnv

ASSET_DIR = f"{Path(__file__).parent.parent}/assets"
RDIP_XML_DIR = f"{ASSET_DIR}/rotary_inverted_double_pendulum.xml"


class RotaryInvertedDoublePendulumGoalEnv(InvertedDoublePendulumEnv):
    def __init__(self, mode="test", custom_xml_file=RDIP_XML_DIR, *args, **kwargs):
        super().__init__(xml_file=custom_xml_file, *args, **kwargs)
        self.mode = mode

        # 目标角度为直立状态 [theta1, theta2] = [0, 0]
        self.goal = np.array([0.0, 0.0], dtype=np.float32)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            self.logger.addHandler(handler)

        obs_dim = 9
        goal_dim = 2  # theta1 and theta2

        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                ),
                "achieved_goal": spaces.Box(
                    low=-np.pi, high=np.pi, shape=(goal_dim,), dtype=np.float32
                ),
                "desired_goal": spaces.Box(
                    low=-np.pi, high=np.pi, shape=(goal_dim,), dtype=np.float32
                ),
            }
        )

    def compute_reward(self, achieved_goal, desired_goal, info):
        diff = achieved_goal - desired_goal
        distance = np.linalg.norm(diff)
        reward = 0.0 if distance > 0.2 else 1.0

        # sparse reward: distance within threshold → success
        return np.array(reward, dtype=np.float32)

        # optional: continuous reward version
        # return -distance

    def _get_achieved_goal(self):
        # 返回当前关节角度 [theta1, theta2]
        theta1 = self.data.qpos[1]
        theta2 = self.data.qpos[2]
        return np.array([theta1, theta2], dtype=np.float32)

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel

        obs = np.array(
            [
                np.cos(qpos[0]),
                np.sin(qpos[0]),
                np.cos(qpos[1]),
                np.sin(qpos[1]),
                np.cos(qpos[2]),
                np.sin(qpos[2]),
                qvel[0],
                qvel[1],
                qvel[2],
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.mode == "test":
            self.init_qpos = np.array([0.0, np.pi, 0.0])  # 初始角度 [θ_base, π, 0]
        elif self.mode == "stable":
            self.init_qpos = np.array([0.0, 0.0, 0.0])  # 初始角度为竖直向上
        else:
            raise ValueError("mode must be 'test' or 'stable'")

        self.set_state(
            self.init_qpos + self.np_random.uniform(-0.01, 0.01, size=self.model.nq),
            self.init_qvel + self.np_random.normal(scale=0.01, size=self.model.nv),
        )

        obs = {
            "observation": self._get_obs(),
            "achieved_goal": self._get_achieved_goal(),
            "desired_goal": self.goal.copy(),
        }
        return obs, {}

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        achieved_goal = self._get_achieved_goal()
        obs = self._get_obs()
        reward = self.compute_reward(achieved_goal, self.goal, {})

        terminated = reward == 1.0 if self.mode == "stable" else False
        truncated = False

        observation_dict = {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal.copy(),
        }

        return observation_dict, reward, terminated, truncated, {}
