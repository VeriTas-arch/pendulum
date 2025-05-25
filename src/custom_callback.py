import csv
import os
import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

import utils

DATA_DIR = Path(__file__).resolve().parent.parent / "data"  # 数据目录
CSV_DIR = DATA_DIR / "csv"  # CSV目录


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

        current_time = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = CSV_DIR / f"{model_name}_{mode}_{extra}_{current_time}.csv"
        self.header_written = False

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval == 0:
            print(f"Step: {self.num_timesteps}, Reward: {self.locals['rewards']}")

            utils.save_model(
                self.model, self.env_type, self.model_name, self.mode, self.extra
            )

            # 获取当前日志字典
            log_dict = {
                k: v
                for k, v in self.logger.name_to_value.items()
                if isinstance(v, (int, float))
            }
            if log_dict:
                write_header = (
                    not os.path.exists(self.csv_path) or not self.header_written
                )
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=log_dict.keys())
                    if write_header:
                        writer.writeheader()
                        self.header_written = True
                    writer.writerow(log_dict)

        return True
