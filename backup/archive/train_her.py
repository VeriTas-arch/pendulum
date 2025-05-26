from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

import utils
from custom_callback import LoggingCallback

MODE = "test"
DATA_DIR = utils.DATA_DIR
LOG_DIR = str(utils.LOG_DIR)
ENV_TYPE = 2
LOAD_MODEL = False  # 是否加载模型
EXTRA = "new_alg_1"  # 额外的后缀，不加则设为 None

gym.register(
    id="RotaryInvertedDoublePendulumGoalEnv-v1",
    entry_point="goal_envs:RotaryInvertedDoublePendulumGoalEnv",
)
env = make_vec_env(
    "RotaryInvertedDoublePendulumGoalEnv-v1",
    n_envs=32,
    wrapper_class=gym.wrappers.TimeLimit,
    wrapper_kwargs={"max_episode_steps": 4000},
    env_kwargs={
        "mode": MODE,
        # "render_mode": "human"
    },
)

if LOAD_MODEL:
    model = SAC.load(
        f"{DATA_DIR}/sac_her_rotary_inverted_double_pendulum_{MODE}.zip", env=env
    )
else:
    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        buffer_size=int(1e6),
        batch_size=256,
        gamma=0.98,
        verbose=1,
        learning_starts=128000,
        tensorboard_log=LOG_DIR,
    )

model.learn(
    total_timesteps=1e6,
    callback=LoggingCallback(
        log_interval=2000, model_name="HER", mode=MODE, env_type=ENV_TYPE, extra=EXTRA
    ),
)

save_path = f"{DATA_DIR}/sac_her_rotary_inverted_double_pendulum_{MODE}.zip"
model.save(save_path)
