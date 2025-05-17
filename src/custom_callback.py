from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

DATA_DIR = Path(__file__).resolve().parent.parent / "data"  # 数据目录


class LoggingCallback(BaseCallback):
    def __init__(
        self, verbose=0, log_interval=1000, model_name=None, mode=None, env_type=None
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

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval == 0:
            print(f"Step: {self.num_timesteps}, Reward: {self.locals['rewards']}")

            if self.env_type == 0:
                self.model.save(
                    f"{DATA_DIR}/{self.model_name}_pendulum_{self.mode}.zip"
                )
            elif self.env_type == 1:
                self.model.save(
                    f"{DATA_DIR}/{self.model_name}_inverted_double_pendulum_{self.mode}.zip"
                )
        return True
