from stable_baselines3.common.callbacks import BaseCallback


class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0, log_interval=1000, model_name="model", mode="test"):
        super(LoggingCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.model_name = model_name
        self.mode = mode

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval == 0:
            print(f"Step: {self.num_timesteps}, Reward: {self.locals['rewards']}")

            self.model.save(
                f"{self.model_name}_inverted_double_pendulum_{self.mode}.zip"
            )
        return True
