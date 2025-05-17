from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

from custom_callback import LoggingCallback

ENV_TYPE = 1  # 0 for Pendulum, 1 for InvertedDoublePendulum
MODEL_TYPE = "SAC"  # SAC or PPO
MODE = "stable"  # test for swing up, stable for stable control
LOAD_MODEL = True  # 是否加载模型
DATA_DIR = Path(__file__).resolve().parent.parent / "data"  # 数据目录



if ENV_TYPE == 0:
    env_name = "Pendulum-v1"
    env = make_vec_env(env_name, n_envs=1)

    if LOAD_MODEL:
        model = SAC.load(f"{DATA_DIR}/sac_pendulum_{MODE}.zip", env=env)
    else:
        model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-3)

    model.learn(
        total_timesteps=20000, callback=LoggingCallback(log_interval=1000, mode=MODE)
    )
    model.save(f"{DATA_DIR}/sac_pendulum_{MODE}.zip")
elif ENV_TYPE == 1:
    gym.register(
        id="CustomInvertedDoublePendulum-v1",
        entry_point="custom_envs:CustomInvertedDoublePendulumEnv",
    )
    env = make_vec_env(
        "CustomInvertedDoublePendulum-v1",
        n_envs=4,
        wrapper_class=gym.wrappers.TimeLimit,
        wrapper_kwargs={"max_episode_steps": 4000},
        env_kwargs={
            "mode": MODE,
            # "render_mode": "human"
        },
    )

    if MODEL_TYPE == "SAC":
        if LOAD_MODEL:
            model = SAC.load(
                f"{DATA_DIR}/sac_inverted_double_pendulum_{MODE}.zip", env=env
            )
        else:
            model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-3)

        model.learn(
            total_timesteps=1e5,
            callback=LoggingCallback(log_interval=2000, model_name="sac"),
        )
        model.save(f"{DATA_DIR}/sac_inverted_double_pendulum_{MODE}.zip")
    elif MODEL_TYPE == "PPO":
        if LOAD_MODEL:
            model = PPO.load(f"ppo_inverted_double_pendulum_{MODE}.zip", env=env)
        else:
            model = PPO(
                "MlpPolicy",
                env,
                batch_size=128,
                n_steps=128,
                gamma=0.98,
                learning_rate=0.000155454,
                ent_coef=1.05057e-06,
                clip_range=0.4,
                n_epochs=10,
                gae_lambda=0.8,
                max_grad_norm=0.5,
                vf_coef=0.695929,
            )

        model.learn(
            total_timesteps=1e6,
            callback=LoggingCallback(log_interval=1000, model_name="ppo"),
        )
        model.save(f"{DATA_DIR}/ppo_inverted_double_pendulum_{MODE}.zip")
