import logging
import random  # noqa: F401
from pathlib import Path
from collections import deque
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.inverted_double_pendulum_v5 import InvertedDoublePendulumEnv

ASSET_DIR = f"{Path(__file__).parent.parent}/assets"
DIP_XML_DIR = f"{ASSET_DIR}/inverted_double_pendulum.xml"
RDIP_XML_DIR = f"{ASSET_DIR}/rotary_inverted_double_pendulum.xml"
RIP_XML_DIR = f"{ASSET_DIR}/rotary_inverted_pendulum.xml"


class CustomInvertedDoublePendulumEnv(InvertedDoublePendulumEnv):
    def __init__(self, mode=None, custom_xml_file=DIP_XML_DIR, *args, **kwargs):
        super().__init__(xml_file=custom_xml_file, *args, **kwargs)

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

        self.vel_history = deque(maxlen=10)

        high = np.inf * np.ones(9, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def reset_model(self):

        self.vel_history.clear()

        # angle_offset = np.pi / 36
        # sign = random.choice([-1, 1])
        if self.mode == "test":
            self.init_qpos = np.array([0.0, np.pi, 0.0])
        elif self.mode == "stable":
            # self.init_qpos = np.array([0.0, 0.0, 0.0])

            # init with a small angle offset
            self.init_qpos = np.array([0.0, 0.13, -0.52])

            # self.init_qpos = np.array([0.0, sign * angle_offset, 0])
            # amp = 1.0
            # self.init_qvel = np.array([0.0, -0.8 * amp, 0.5 * amp])
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
        # noise = np.random.normal(0, 0.0001, size=np.shape(action))
        # action = action + noise

        self.do_simulation(action, self.frame_skip)

        x, _, y = self.data.site_xpos[4]
        observation = self._get_obs()

        reward, reward_info = None, None

        if self.mode == "stable":
            terminated = bool(y <= 2.0)
            reward, reward_info = self.compute_reward_stable_test_2(x, y, terminated)
        elif self.mode == "test":
            terminated = False
            reward, reward_info = self.compute_reward_test_4(x, y, terminated)

        # reward, reward_info = self._get_rew(x, y, terminated)

        info = reward_info

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`

        return observation, reward, terminated, False, info

    def _get_obs(self):
        qpos = self.data.qpos  # [theta_base, theta1, theta2]
        qvel = self.data.qvel  # [dtheta_base, dtheta1, dtheta2]

        theta_base = qpos[0]
        theta1 = qpos[1]
        theta2 = qpos[2]

        dtheta_base = qvel[0]
        dtheta1 = qvel[1]
        dtheta2 = qvel[2]

        obs = np.array(
            [
                np.cos(theta_base),
                np.sin(theta_base),
                np.cos(theta1),
                np.sin(theta1),
                np.cos(theta2),
                np.sin(theta2),
                dtheta_base,
                dtheta1,
                dtheta2,
            ],
            dtype=np.float32,
        )

        # 为观测添加高斯噪声（均值0，标准差0.02）
        # obs_noise = np.random.normal(0, 0.01, size=obs.shape).astype(np.float32)
        # obs = obs + obs_noise

        return obs

    def _get_rew(self, x, y, terminated):
        v0, v1, v2 = self.data.qvel
        theta = self.data.qpos[0]
        dist_penalty = (
            0.01 * (x - 0.2159) ** 2 + (y - 0.5365) ** 2 + 0.02 * (theta) ** 2
        )
        vel_penalty = 1e-4 * v0 + 2e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = self._healthy_reward * int(not terminated)

        reward = alive_bonus - dist_penalty - vel_penalty

        reward_info = {
            "reward_survive": alive_bonus,
            "distance_penalty": -dist_penalty,
            "velocity_penalty": -vel_penalty,
        }

        return reward, reward_info

    def compute_reward_stable(self, x, y, terminated):
        theta1, theta2 = self.data.qpos[1], self.data.qpos[2]
        v0, v1, v2 = self.data.qvel
        # move the reward to above 0
        shift = 2

        posture_reward = 0
        vel_penalty = 0
        swing_reward = 0
        alive_bonus = 0

        ctrl_penalty = np.sum(self.data.ctrl[0] ** 2)

        if y < -0.3:
            angular_momentum = abs(v1) + abs(v2)
            swing_reward = 0.5 * angular_momentum - 0.1 * ctrl_penalty

        if y > 0.3:
            posture_reward = 2 - 3 * abs(0.5365 - y) - abs(theta1 - theta2) * 0.5
            ctrl_penalty *= 2

        vel_penalty = (7 * v0**2 + 3 * v1**2 + 3 * v2**2) * 7e-3 * (
            0.6 + y
        ) + 7e1 * ctrl_penalty

        if y > 0.5:
            vel_penalty += (v2**2) * 0.1 + (v1**2) * 0.2
            alive_bonus = (posture_reward + 5) * int(not terminated)

        dist_penalty = 1e-1 * (x - 0.2159) ** 2

        # 新增奖励：靠近最高点时速度小
        peak_slow_bonus = 0
        if y > 0.5 and abs(v2) < 0.6 and abs(v1) < 0.6:
            peak_slow_bonus = 2 * (1.2 - abs(v2) - abs(v1))  # 奖励速度低

        reward = (
            alive_bonus
            - dist_penalty
            - vel_penalty
            + peak_slow_bonus
            + shift
            + swing_reward
        )
        reward_info = {
            "reward_survive": alive_bonus,
            "distance_penalty": -dist_penalty,
            "velocity_penalty": -vel_penalty,
            "peak_slow_bonus": peak_slow_bonus,
        }

        return reward, reward_info

    def compute_reward_stable_test(self, x, y, terminated):
        theta1, theta2 = self.data.qpos[1], self.data.qpos[2]
        v0, v1, v2 = self.data.qvel
        ctrl = self.data.ctrl[0]

        # 参数
        y_target = 0.5365  # 稳态的 y 高度
        theta_align = abs(theta1 - theta2)

        # 奖励项
        y_reward = 2.0 * np.exp(-20 * (y - y_target) ** 2)  # y靠近目标位置时奖励高
        angle_reward = 2.0 * np.exp(
            -5 * (theta_align) ** 2 - 5 * theta1**2
        )  # 角度差越小越好
        speed_penalty = 0.05 * (v0**2 + v1**2 + v2**2)
        ctrl_penalty = 0.01 * (ctrl**2)

        # 如果稳定时速度足够小，额外奖励
        stable_bonus = 0
        if abs(v1) < 0.3 and abs(v2) < 0.3:
            stable_bonus = 1.0

        # 活着就奖励
        alive_bonus = 3.0 if not terminated else 0.0

        # 总奖励
        reward = (
            y_reward
            + angle_reward
            + stable_bonus
            + alive_bonus
            - speed_penalty
            - ctrl_penalty
        )

        reward_info = {
            "y_reward": y_reward,
            "angle_reward": angle_reward,
            "stable_bonus": stable_bonus,
            "alive_bonus": alive_bonus,
            "speed_penalty": -speed_penalty,
            "ctrl_penalty": -ctrl_penalty,
        }

        return reward, reward_info

    def compute_reward_stable_test_2(self, x, y, terminated):
        v0, v1, v2 = self.data.qvel
        theta0, theta1, theta2 = self.data.qpos
        ctrl = self.data.ctrl[0]

        # 初始状态 & 目标状态
        # q_init = np.array([0.13, -0.52])

        # 使用另一状态作为初始状态，避免摆杆始终受到惩罚而没有奖励
        q_init = np.array([np.pi / 2, 0.0])
        q_goal = np.array([0.0, 0.0])
        q_now = np.array([theta1, theta2])

        # --- 状态距离 ---
        dist_to_goal = np.linalg.norm(q_now - q_goal)
        dist_to_init = np.linalg.norm(q_now - q_init)

        # --- 插值权重 ---
        # 1. Normalized progress toward goal (0 = at init, 1 = at goal)
        total_path = np.linalg.norm(q_init - q_goal) + 1e-6  # avoid div by zero
        progress_ratio = np.clip(1.0 - dist_to_goal / total_path, 0.0, 1.0)
        progress_ratio = progress_ratio * 2 - 1

        # 2. 基于插值的奖励（靠近目标越多，奖励越大）
        interp_reward = 3.0 * progress_ratio  # 线性插值

        # --- 惩罚项 ---

        # --- 累计速度惩罚 ---
        vel_norm_sq = v0**2 + v1**2 + v2**2
        self.vel_history.append(vel_norm_sq)

        if len(self.vel_history) > 0:
            integrated_vel_penalty = 0.05 * (
                sum(self.vel_history) / len(self.vel_history)
            )
        else:
            integrated_vel_penalty = 0.0

        ctrl_penalty = 0.01 * (ctrl**2)
        alive_bonus = 5.0 if not terminated else 0.0

        # --- 总奖励 ---
        reward = interp_reward + alive_bonus - integrated_vel_penalty - ctrl_penalty

        reward_info = {
            "interp_reward": interp_reward,
            "progress_ratio": progress_ratio,
            "dist_to_goal": -dist_to_goal,
            "dist_to_init": dist_to_init,
            "velocity_penalty": -integrated_vel_penalty,
            "vel_penalty_raw": -vel_norm_sq,
            "ctrl_penalty": -ctrl_penalty,
            "alive_bonus": alive_bonus,
        }

        return reward, reward_info

    def compute_reward_test(self, x, y, terminated):
        # --- 获取状态变量 ---
        theta0, theta1, theta2 = self.data.qpos
        v0, v1, v2 = self.data.qvel
        ctrl = (
            self.data.ctrl[0]
            if isinstance(self.data.ctrl, np.ndarray)
            else self.data.ctrl
        )
        ctrl = np.array(ctrl)

        # --- 常量与参数 ---
        target_y = 0.5365
        shift = 0.0

        # --- 奖励组件初始化 ---
        posture_reward = 0.0
        swing_reward = 0.0
        alive_bonus = 0.0
        peak_slow_bonus = 0.0

        # --- 控制惩罚 ---
        ctrl_penalty = 0.2 * np.sum(ctrl**2)

        # --- 摆动阶段奖励：鼓励产生角动量（当高度很低时） ---
        if y < -0.32:
            theta_shift = abs(theta1 - np.pi) + abs(theta2)
            swing_reward = 1.2 * theta_shift

        if y > 0:
            swing_reward += 1.5

        # --- 姿态奖励：鼓励靠近顶部并抑制角度差 ---
        if y > 0.35:
            height_bonus = np.exp(-8 * (y - target_y) ** 2) * 3
            angle_bonus = (
                np.exp(-3 * abs(theta1 - theta2) - 5 * abs(np.sin(theta1))) * 1.5
            )
            posture_reward = height_bonus + angle_bonus
            ctrl_penalty *= 1.5

        # --- 顶端速度惩罚 ---
        base_vel_penalty = 7 * v0**2 + 3 * v1**2 + 3 * v2**2
        height_factor = 0.5 + 0.5 * np.tanh(5 * (y - 0.45))
        vel_penalty = base_vel_penalty * 7e-3 * height_factor + 0.07 * ctrl_penalty

        if y > 0.5:
            alive_bonus = (posture_reward + 3.0) * int(not terminated)

        # --- 额外：顶端速度尽量小 ---
        if y > 0.5:
            speed_sum = abs(v1) + abs(v2)
            peak_slow_bonus = 2.0 * max(1.2 - speed_sum, 0.0)

        # --- 额外：距离偏移惩罚（保持在轨道中央） ---
        # if y > 0.5 and not terminated:
        # theta0_penalty = 2.0 * (np.sin(theta0)) ** 2

        # --- 平滑动作惩罚 ---
        # if hasattr(self, "prev_action"):
        #     delta_u = np.sum((self.prev_action - ctrl) ** 2)
        #     action_smooth_penalty = 0.5 * delta_u
        # self.prev_action = np.copy(ctrl)

        # --- 总奖励计算 ---
        reward = (
            alive_bonus
            + swing_reward
            + peak_slow_bonus
            - vel_penalty
            - ctrl_penalty
            # - theta0_penalty
            # - action_smooth_penalty
            + shift
        )

        # --- 可视化用信息 ---
        reward_info = {
            "reward_survive": alive_bonus,
            "velocity_penalty": -vel_penalty,
            "ctrl_penalty": -ctrl_penalty,
            "peak_slow_bonus": peak_slow_bonus,
            "swing_reward": swing_reward,
            # "theta0_penalty": -theta0_penalty,
            # "action_smooth_penalty": -action_smooth_penalty,
        }

        return reward, reward_info

    def compute_reward_test_2(self, x, y, terminated):
        target_pos = np.array([0, 0, 0.5365])
        theta0, theta1, theta2 = self.data.qpos
        v0, v1, v2 = self.data.qvel
        # move the reward to above 0
        shift = 2

        posture_reward = 0
        ctrl_penalty = 0
        vel_penalty = 0

        if y > 0.4:
            angle_bonus = (
                np.exp(-5 * abs(theta1 - theta2) - 7 * abs(np.sin(theta1))) * 2
            )
            posture_reward = 3 * y + angle_bonus
            ctrl_penalty = np.sum(self.data.ctrl[0] ** 2)

        vel_penalty = (7 * v0**2 + 1 * v1**2 + 2 * v2**2) * 7e-3 * (
            0.54 + y
        ) + 7e-2 * ctrl_penalty

        if y > 0.4:
            vel_penalty += (v2**2) * 0.1

        alive_bonus = (posture_reward - 15 * (y - target_pos[2]) ** 2) * int(
            bool(y > 0.5 and not terminated)
        )
        dist_penalty = 1e-1 * (x - 0.2159) ** 2

        # 新增奖励：靠近最高点时速度小
        peak_slow_bonus = 0
        if y > 0.5 and abs(v2) < 0.5:
            peak_slow_bonus = 3 * max((1.2 - abs(v2) - abs(v1)), 0)  # 奖励速度低

        reward = alive_bonus - dist_penalty - vel_penalty + peak_slow_bonus + shift
        reward_info = {
            "reward_survive": alive_bonus,
            "distance_penalty": -dist_penalty,
            "velocity_penalty": -vel_penalty,
            "peak_slow_bonus": peak_slow_bonus,
        }

        return reward, reward_info

    def compute_reward_test_3(self, x, y, terminated):
        target_pos = np.array([0, 0, 0.5365])
        theta0, theta1, theta2 = self.data.qpos
        v0, v1, v2 = self.data.qvel
        shift = 2  # 保证奖励为正

        # 控制代价
        ctrl_penalty = np.sum(self.data.ctrl[0] ** 2)

        # === 姿态奖励项 ===
        posture_reward = 0
        angle_bonus = 0

        if y > 0.4:
            # 1. 两杆之间角度差越小越好（对齐）
            angle_diff_penalty = abs(theta1 - theta2)

            # 2. 绝对角度偏离竖直越小越好（越接近0越好）
            uprightness_penalty = theta1**2 + theta2**2

            # 综合姿态奖励项（越小越好，用 exp 凸显直立的优势）
            angle_bonus = (
                np.exp(-6 * angle_diff_penalty - 6 * uprightness_penalty) * 3.0
            )

            # 基于 y 高度的基本奖励
            posture_reward = 3 * y + angle_bonus

        # === 速度惩罚项 ===
        vel_penalty = (7 * v0**2 + 1 * v1**2 + 2 * v2**2) * 7e-3 * (
            0.54 + y
        ) + 7e-2 * ctrl_penalty
        if y > 0.4:
            vel_penalty += (v2**2) * 0.1

        # === 存活奖励 ===
        alive_bonus = (posture_reward - 15 * (y - target_pos[2]) ** 2) * int(
            y > 0.5 and not terminated
        )

        # === 位置惩罚（不偏离 x 轴） ===
        dist_penalty = 1e-1 * (x - 0.2159) ** 2

        # === 顶端慢速奖励（惩罚摆动）===
        peak_slow_bonus = 0
        if y > 0.5 and abs(v2) < 0.5:
            peak_slow_bonus = 3 * max((1.2 - abs(v2) - abs(v1)), 0)

        # === 最终 reward ===
        reward = alive_bonus - dist_penalty - vel_penalty + peak_slow_bonus + shift

        reward_info = {
            "reward_survive": alive_bonus,
            "distance_penalty": -dist_penalty,
            "velocity_penalty": -vel_penalty,
            "peak_slow_bonus": peak_slow_bonus,
            "angle_bonus": angle_bonus,
            "upright_penalty": -(theta1**2 + theta2**2),
            "angle_diff_penalty": -abs(theta1 - theta2),
        }

        return reward, reward_info

    def compute_reward_test_4(self, x, y, terminated):
        target_pos = np.array([0, 0, 0.5365])
        theta0, theta1, theta2 = self.data.qpos
        v0, v1, v2 = self.data.qvel
        ctrl_penalty = np.sum(self.data.ctrl[0] ** 2)
        shift = 2  # 保持奖励为正

        # 计算角度偏离竖直的量（0度为竖直）
        uprightness = theta1**2 + theta2**2
        angle_diff = abs(theta1 - theta2)

        # 判断当前阶段：起摆阶段 if 摆杆角度较大（远离竖直），稳摆阶段 if 角度较小（接近竖直）
        swingup_threshold = 0.5  # 约28.6度，可以根据实验调整

        if uprightness > swingup_threshold**2:
            # === 起摆阶段奖励 ===
            # 鼓励角度远离竖直（摆杆展开）
            swingup_bonus = np.exp(-3 * uprightness) * (-1) + 1.5  # 角度大时奖励高
            # 保留一定的角度差惩罚防止两杆完全反向展开
            angle_align_penalty = 0.3 * angle_diff

            posture_reward = swingup_bonus - angle_align_penalty + 2 * y
        else:
            # === 稳摆阶段奖励 ===
            # 鼓励角度接近0，且两杆对齐
            posture_reward = 4 * np.exp(-8 * uprightness - 8 * angle_diff)
            # 加分鼓励y高，促进稳定在最高点
            posture_reward += 3 * y

        # 速度惩罚，防止大幅摆动
        vel_penalty = (7 * v0**2 + 1 * v1**2 + 2 * v2**2) * 7e-3 * (
            0.54 + y
        ) + 7e-2 * ctrl_penalty
        if y > 0.4:
            vel_penalty += (v2**2) * 0.1

        # 存活奖励，考虑是否终止
        alive_bonus = (posture_reward - 15 * (y - target_pos[2]) ** 2) * int(
            y > 0.4 and not terminated
        )

        # 位置惩罚，限制x轴偏移
        dist_penalty = 1e-1 * (x - 0.2159) ** 2

        # 顶端低速奖励，奖励稳定
        peak_slow_bonus = 0
        if y > 0.5 and abs(v2) < 0.5:
            peak_slow_bonus = 3 * max((1.2 - abs(v2) - abs(v1)), 0)

        reward = alive_bonus - dist_penalty - vel_penalty + peak_slow_bonus + shift

        reward_info = {
            "reward_survive": alive_bonus,
            "distance_penalty": -dist_penalty,
            "velocity_penalty": -vel_penalty,
            "peak_slow_bonus": peak_slow_bonus,
            "posture_reward": posture_reward,
            "uprightness": uprightness,
            "angle_diff": angle_diff,
        }

        return reward, reward_info


class CustomRotaryInvertedPendulumEnv(InvertedDoublePendulumEnv):
    def __init__(self, mode=None, custom_xml_file=RIP_XML_DIR, *args, **kwargs):
        super().__init__(xml_file=custom_xml_file, *args, **kwargs)

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
