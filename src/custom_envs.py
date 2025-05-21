import logging
from pathlib import Path

import numpy as np
from gymnasium.envs.mujoco.inverted_double_pendulum_v5 import \
    InvertedDoublePendulumEnv

ASSET_DIR = f"{Path(__file__).parent.parent}/assets"
DIP_XML_DIR = f"{ASSET_DIR}/inverted_double_pendulum.xml"
RDIP_XML_DIR = f"{ASSET_DIR}/rotary_inverted_double_pendulum.xml"


class CustomInvertedDoublePendulumEnv(InvertedDoublePendulumEnv):
    def __init__(self, mode="none", *args, **kwargs):
        super().__init__(xml_file=DIP_XML_DIR, *args, **kwargs)

        self.mode = mode

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self.logger.addHandler(handler)
        self.logger.info(
            f"Custom Inverted Double Pendulum Env initialized with mode: {mode}"
        )

    def reset_model(self):
        if self.mode == "test":
            self.init_qpos = np.array([0.0, np.pi, 0])
        elif self.mode == "stable":
            self.init_qpos = np.array([0.0, 0.0, 0])
        elif self.mode == "none":
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
        self.do_simulation(action, self.frame_skip)

        x, _, y = self.data.site_xpos[0]
        observation = self._get_obs()
        terminated = bool(y <= 1)
        reward, reward_info = self._get_rew(x, y, terminated)

        info = reward_info

        if self.render_mode == "human":
            self.render()

        if self.mode == "stable":
            pass
        elif self.mode == "test":
            terminated = False

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info


class CustomRotaryInvertedDoublePendulumEnv(InvertedDoublePendulumEnv):
    def __init__(self, mode="none", *args, **kwargs):
        super().__init__(xml_file=RDIP_XML_DIR, *args, **kwargs)

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

    def reset_model(self):
        if self.mode == "test":
            self.init_qpos = np.array([0.0, np.pi, 0.0])
        elif self.mode == "stable":
            self.init_qpos = np.array([0.0, 0.0, 0.0])
        elif self.mode == "none":
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
        self.do_simulation(action, self.frame_skip)

        x, _, y = self.data.site_xpos[4]
        observation = self._get_obs()
        terminated = bool(y <= 0.45)
        reward, reward_info = self._get_rew(x, y, terminated)

        info = reward_info

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_rew(self, x, y, terminated):
        v0, v1, v2 = self.data.qvel
        theta = self.data.qpos[0]
        dist_penalty = 0.01 * (x - 0.2159) ** 2 + (y - 0.5365) ** 2 + 0.02 * abs(theta)
        vel_penalty = 1e-4 * v0 + 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = self._healthy_reward * int(not terminated)

        reward = alive_bonus - dist_penalty - vel_penalty

        reward_info = {
            "reward_survive": alive_bonus,
            "distance_penalty": -dist_penalty,
            "velocity_penalty": -vel_penalty,
        }

        return reward, reward_info
