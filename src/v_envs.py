import logging
import random  # noqa: F401
from pathlib import Path

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.inverted_double_pendulum_v5 import InvertedDoublePendulumEnv

ASSET_DIR = f"{Path(__file__).parent.parent}/assets"
RDIP_XML_DIR = f"{ASSET_DIR}/rotary_inverted_double_pendulum.xml"


class CustomRotaryInvertedDoublePendulumEnv(InvertedDoublePendulumEnv):
    def __init__(self, mode=None, custom_xml_file=RDIP_XML_DIR, *args, **kwargs):
        super().__init__(xml_file=custom_xml_file, *args, **kwargs)

        self.mode = mode

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            self.logger.addHandler(handler)
        self.logger.info(
            f"Custom Rotary Inverted Double Pendulum Env initialized with mode: {mode}"
        )

        self.isUp = False
        self.stable_counter = 0

        high = np.inf * np.ones(6, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def reset_model(self):

        self.isUp = False
        self.stable_counter = 0

        # angamp = 1.0
        # angle_offset_1 = 0.16
        # angle_offset_2 = 0.36

        # sign = random.choice([-1, 1])

        if self.mode == "test":
            self.init_qpos = np.array([0.0, np.pi, 0.0])
        elif self.mode == "stable":
            self.init_qpos = np.array([0.0, 0.0, 0.0])

            # init with a small angle offset
            # self.init_qpos = np.array([0.0, 0.13, -0.52])

            # self.init_qpos = np.array(
            #     [0.0, sign * angamp * angle_offset_1, -sign * angamp * angle_offset_2]
            # )
            # amp = 0.4
            # velo1 = np.random.normal(0.7 * amp, 0.005)
            # velo2 = np.random.normal(1 * amp, 0.005)

            # self.init_qvel = np.array([0.0, -sign * velo1, sign * velo2])
        else:
            raise ValueError(
                "Invalid mode. Choose 'test' for swing up task, 'stable' for stable control task."
            )

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        self.set_state(
            self.init_qpos
            + self.np_random.uniform(
                low=noise_low, high=noise_high, size=self.model.nq
            ),
            self.init_qvel
            + self.np_random.standard_normal(self.model.nv) * self._reset_noise_scale,
        )
        return self._get_obs()

    def step(self, action):
        # noise = np.random.normal(0, 0.001, size=np.shape(action))
        # action = action + noise

        self.do_simulation(action, self.frame_skip)

        _, _, y = self.data.site_xpos[4]
        reward, reward_info = None, None
        observation = self._get_obs()

        # absolute angle of the pendulum segments
        theta0, theta1, theta2 = self._get_theta()

        if self.mode == "stable":
            terminated = bool(y <= 0.2)

            # 临时设置 stable 模式也不终止
            # terminated = False
            reward, reward_info = self.reward_stable(theta0, theta1, theta2, terminated)
        elif self.mode == "test":
            terminated = False
            reward, reward_info = self.reward_swingup(
                theta0, theta1, theta2, terminated
            )

        info = reward_info

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`

        return observation, reward, terminated, False, info

    def _get_obs(self):
        theta0, theta1, theta2 = self._get_theta()

        obs = np.array(
            [
                np.cos(theta0),
                np.sin(theta0),
                np.cos(theta1),
                np.sin(theta1),
                np.cos(theta2),
                np.sin(theta2),
            ],
            dtype=np.float32,
        )

        # 为观测添加高斯噪声
        # obs_noise = np.random.normal(0, 0.002, size=obs.shape).astype(np.float32)
        # obs = obs + obs_noise

        return obs

    def _get_theta(self):
        qpos = self.data.qpos  # [theta0, theta1, theta2-theta1]
        return qpos[0], qpos[1], qpos[1] + qpos[2]

    def guassian_function(self, x, mu, sigma):
        """
        Gaussian function for reward shaping.
        """
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    def reward_stable(self, theta0, theta1, theta2, terminated):
        reward_theta0 = self.guassian_function(theta0, 0.0, 0.1)
        reward_theta1 = self.guassian_function(theta1, 0.0, 0.1)
        reward_theta2 = self.guassian_function(theta2, 0.0, 0.1)

        # normalize the reward
        reward_raw = reward_theta0 + reward_theta1 + reward_theta2
        max_gauss = self.guassian_function(0.0, 0.0, 0.1)
        reward = (reward_raw / (3 * max_gauss)) * 10
        reward = np.clip(reward, 0, 10)

        reward_info = {
            "reward_theta0": reward_theta0,
            "reward_theta1": reward_theta1,
            "reward_theta2": reward_theta2,
            "reward_raw": reward_raw,
            "reward_normalized": reward,
        }

        return reward, reward_info

    def reward_swingup(self, theta0, theta1, theta2, terminated):
        ...

        # return reward, reward_info
