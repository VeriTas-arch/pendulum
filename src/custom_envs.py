import logging
from pathlib import Path

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.inverted_double_pendulum_v5 import InvertedDoublePendulumEnv

ASSET_DIR = f"{Path(__file__).parent.parent}/assets"
DIP_XML_DIR = f"{ASSET_DIR}/inverted_double_pendulum.xml"
RDIP_XML_DIR = f"{ASSET_DIR}/rotary_inverted_double_pendulum.xml"
RIP_XML_DIR = f"{ASSET_DIR}/rotary_inverted_pendulum.xml"


class CustomInvertedDoublePendulumEnv(InvertedDoublePendulumEnv):
    def __init__(self, mode=None, *args, **kwargs):
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
    def __init__(self, mode=None, *args, **kwargs):
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
        self.do_simulation(action, self.frame_skip)

        x, _, y = self.data.site_xpos[4]
        v0, v1, v2 = self.data.qvel
        observation = self._get_obs()

        if self.mode == "stable":
            terminated = bool(y <= 0.45)
        elif self.mode == "test":
            terminated = False

        # reward, reward_info = self._get_rew(x, y, terminated)
        reward, reward_info = self.compute_reward(x, y, terminated)

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

    def compute_reward(self, x, y, terminated):
        target_pos = np.array([0, 0, 0.5365])
        # theta = self.data.qpos[0]
        v0, v1, v2 = self.data.qvel

        posture_reward = 0
        ctrl_penalty = 0
        if y > 0.3:
            posture_reward = 5 * y
            ctrl_penalty = np.sum(self.data.ctrl[0] ** 2)

        alive_bonus = posture_reward - 10 * (y - target_pos[2]) ** 2
        dist_penalty = 1e-2 * (x - 0.2159) ** 2
        vel_penalty = (7 * v0**2 + 1 * v1**2 + 2 * v2**2) * 7e-3 * (
            0.5365 + y
        ) + 7e-2 * ctrl_penalty

        reward = alive_bonus - dist_penalty - vel_penalty
        reward_info = {
            "reward_survive": alive_bonus,
            "distance_penalty": -dist_penalty,
            "velocity_penalty": -vel_penalty,
        }

        return reward, reward_info

    def compute_reward_multi_stage(self, x, y, terminated):
        target_pos = np.array([0, 0, 0.5365])
        v0, v1, v2 = self.data.qvel
        ctrl_penalty = np.sum(self.data.ctrl[0] ** 2)

        # 阶段划分阈值
        swing_threshold = -0.3  # 小于此值为“蓄能阶段”
        upright_threshold = 0.5  # 超过此值视为“成功起摆”

        # -----------------------------
        # 阶段 1：鼓励振荡（角速度 + 能量）
        # -----------------------------
        swing_reward = 0
        if y < swing_threshold:
            angular_momentum = abs(v1) + abs(v2)
            swing_reward = 0.5 * angular_momentum - 0.1 * ctrl_penalty

        # -----------------------------
        # 阶段 2：鼓励竖直上摆 + 姿态奖励
        # -----------------------------
        upright_bonus = 0
        posture_reward = 0
        if y >= swing_threshold:
            upright_bonus = -10 * (y - target_pos[2]) ** 2
            if y > upright_threshold:
                posture_reward = 5 * y  # 奖励越高越好

        # -----------------------------
        # 惩罚项：位置偏差、能量消耗
        # -----------------------------
        dist_penalty = 1e-2 * (x - 0.2159) ** 2 + (y - 0.5365) ** 2
        vel_penalty = (7 * v0**2 + 1 * v1**2 + 2 * v2**2) * 7e-3 * (0.5365 + y)
        ctrl_penalty_term = 7e-2 * ctrl_penalty

        # -----------------------------
        # 最终奖励加权
        # -----------------------------
        reward = (
            swing_reward
            + upright_bonus
            + posture_reward
            - dist_penalty
            - vel_penalty
            - ctrl_penalty_term
        )

        reward_info = {
            "reward_swing": swing_reward,
            "reward_upright": upright_bonus,
            "reward_posture": posture_reward,
            "distance_penalty": -dist_penalty,
            "velocity_penalty": -vel_penalty,
        }

        return reward, reward_info


class CustomRotaryInvertedPendulumEnv(InvertedDoublePendulumEnv):
    def __init__(self, mode=None, *args, **kwargs):
        super().__init__(xml_file=RIP_XML_DIR, *args, **kwargs)

        self.mode = mode
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            self.logger.addHandler(handler)
        self.logger.info(
            f"Custom Rotary Inverted Pendulum Env initialized with mode: {mode}"
        )

    def reset_model(self):
        if self.mode == "test":
            self.init_qpos = np.array([0.0, np.pi])
        elif self.mode == "stable":
            self.init_qpos = np.array([0.0, 0.0])
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
        self.do_simulation(action, self.frame_skip)

        x, _, y = self.data.site_xpos[4]
        observation = self._get_obs()

        if self.mode == "stable":
            terminated = bool(y <= 0.45)
        elif self.mode == "test":
            terminated = False

        # reward, reward_info = self._get_rew(x, y, terminated)
        reward, reward_info = self.compute_reward(x, y, terminated)

        info = reward_info

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`

        return observation, reward, terminated, False, info

    def _get_rew(self, x, y, terminated):
        v0, v1 = self.data.qvel
        theta = self.data.qpos[0]
        dist_penalty = 0.01 * (x - 0.2159) ** 2 + (y - 0.5365) ** 2 + 0.02 * abs(theta)
        vel_penalty = 1e-4 * v0 + 1e-3 * v1**2
        alive_bonus = self._healthy_reward * int(not terminated)

        reward = alive_bonus - dist_penalty - vel_penalty

        reward_info = {
            "reward_survive": alive_bonus,
            "distance_penalty": -dist_penalty,
            "velocity_penalty": -vel_penalty,
        }

        return reward, reward_info

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos[:1],  # cart x pos
                np.sin(self.data.qpos[1:]),  # link angles
                np.cos(self.data.qpos[1:]),
                np.clip(self.data.qvel, -10, 10),
                np.clip(self.data.qfrc_constraint, -10, 10)[:1],
            ]
        ).ravel()

    def compute_reward(self, x, y, terminated):
        target_pos = np.array([0, 0, 0.5365])
        theta = self.data.qpos[0]
        v0, v1 = self.data.qvel
        _healthy_reward = 0

        posture_reward = 0
        if y > 0.3:
            posture_reward = 3 * y

        ctrl_penalty = np.sum(self.data.ctrl[0] ** 2)

        alive_bonus = _healthy_reward - 10 * (y - target_pos[2]) ** 2
        dist_penalty = 1e-2 * (x - 0.2159) ** 2 + 1e-2 * abs(theta)
        vel_penalty = (7 * v0**2 + 1 * v1**2) * 7e-3 + 7e-2 * ctrl_penalty

        reward = alive_bonus - dist_penalty - vel_penalty + posture_reward
        reward_info = {
            "reward_survive": alive_bonus,
            "distance_penalty": -dist_penalty,
            "velocity_penalty": -vel_penalty,
        }

        return reward, reward_info
