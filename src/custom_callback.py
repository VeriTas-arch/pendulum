from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

import utils

DATA_DIR = Path(__file__).resolve().parent.parent / "data"  # 数据目录


class LoggingCallback(BaseCallback):
    def __init__(
        self,
        verbose=0,
        log_interval=1000,
        model_name=None,
        mode=None,
        env_type=None,
        extra=None,
    ):

        if model_name is None:
            raise ValueError("Model name must be provided.")
        if mode is None:
            raise ValueError("Mode must be provided.")
        if env_type is None:
            raise ValueError("Environment must be provided.")

        super(LoggingCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.model_name = model_name
        self.mode = mode
        self.env_type = env_type
        self.extra = extra

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval == 0:
            print(f"Step: {self.num_timesteps}, Reward: {self.locals['rewards']}")

            utils.save_model(
                self.model, self.env_type, self.model_name, self.mode, self.extra
            )

        return True
