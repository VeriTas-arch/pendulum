import logging
from pathlib import Path

import numpy as np
from gymnasium.envs.mujoco.inverted_double_pendulum_v5 import \
    InvertedDoublePendulumEnv

ASSET_DIR = f"{Path(__file__).parent.parent}/assets"
DIP_XML_DIR = f"{ASSET_DIR}/inverted_double_pendulum_exp.xml"
RDIP_XML_DIR = f"{ASSET_DIR}/rotary_inverted_double_pendulum.xml"


class CustomInvertedDoublePendulumEnv(InvertedDoublePendulumEnv):
    def __init__(self, mode="none", *args, **kwargs):
        super().__init__(xml_file=DIP_XML_DIR, *args, **kwargs)

        self.mode = mode
        self.external_force = 0.0

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            )
            self.logger.addHandler(handler)
        self.logger.info(
            f"Custom Inverted Double Pendulum Env initialized with mode: {mode}"
        )

    def _get_body_id_by_name(self, name):
        model = self.model
        if hasattr(model, "body_name2id"):
            return model.body_name2id(name)
        else:
            for i in range(model.nbody):
                if hasattr(model, "body"):
                    body_name = model.body(i).name
                else:
                    body_name = model.names[
                        model.name_bodyadr[i] : model.name_bodyadr[i + 1]
                    ]
                if isinstance(body_name, bytes):
                    body_name = body_name.decode()
                if body_name == name:
                    return i
            raise ValueError(f"Body name {name} not found")

    def reset_model(self):
        if self.mode == "test":
            self.init_qpos = np.array([0.0, np.pi, 0])
        elif self.mode == "stable":
            pass
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
        if (
            hasattr(self, "sim")
            and self.sim is not None
            and hasattr(self, "cart_body_id")
        ):
            self.sim.data.xfrc_applied[:] = 0.0
            self.sim.data.xfrc_applied[self.cart_body_id][0] = self.external_force
        obs, reward, terminated, truncated, info = super().step(action)

        # 尝试暂时取消掉边界条件限制，不然会直接结束，不利于获取奖励
        # x = obs[0]
        # terminated = bool(np.abs(x) >= 3.95)
        terminated = False
        # print(obs[0])

        return obs, reward, terminated, truncated, info


class CustomRotaryInvertedDoublePendulumEnv(InvertedDoublePendulumEnv):
    def __init__(self, mode="none", *args, **kwargs):
        super().__init__(xml_file=RDIP_XML_DIR, *args, **kwargs)

        self.mode = mode
        self.external_force = 0.0

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            )
            self.logger.addHandler(handler)
        self.logger.info(
            f"Custom Rotary Inverted Double Pendulum Env initialized with mode: {mode}"
        )

    def _get_body_id_by_name(self, name):
        model = self.model
        if hasattr(model, "body_name2id"):
            return model.body_name2id(name)
        else:
            for i in range(model.nbody):
                if hasattr(model, "body"):
                    body_name = model.body(i).name
                else:
                    body_name = model.names[
                        model.name_bodyadr[i] : model.name_bodyadr[i + 1]
                    ]
                if isinstance(body_name, bytes):
                    body_name = body_name.decode()
                if body_name == name:
                    return i
            raise ValueError(f"Body name {name} not found")

    def reset_model(self):
        if self.mode == "test":
            self.init_qpos = np.array([0.0, np.pi, 0])
        elif self.mode == "stable":
            pass
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
        if (
            hasattr(self, "sim")
            and self.sim is not None
            and hasattr(self, "cart_body_id")
        ):
            self.sim.data.xfrc_applied[:] = 0.0
            self.sim.data.xfrc_applied[self.cart_body_id][0] = self.external_force
        obs, reward, terminated, truncated, info = super().step(action)

        # 尝试暂时取消掉边界条件限制，不然会直接结束，不利于获取奖励
        # x = obs[0]
        # terminated = bool(np.abs(x) >= 3.95)
        terminated = False
        # print(obs[0])

        return obs, reward, terminated, truncated, info
