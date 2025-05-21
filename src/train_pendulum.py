from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

from custom_callback import LoggingCallback

ENV_TYPE = 2  # 0 for Pendulum, 1 for InvertedDoublePendulum, 2 for RotaryInvertedDoublePendulum
MODEL_TYPE = "SAC"  # SAC or PPO
MODE = "stable"  # test for swing up, stable for stable control
LOAD_MODEL = True  # 是否加载模型
DATA_DIR = Path(__file__).resolve().parent.parent / "data"  # 数据目录
EXTRA = None  # 额外的后缀，不加则设为 None


if ENV_TYPE == 0:
    env_name = "Pendulum-v1"
    env = make_vec_env(env_name, n_envs=4)

    if MODE != "stable":
        raise ValueError("Pendulum-v1 is only tested for stable control. ")
    if MODEL_TYPE != "SAC":
        raise ValueError("Pendulum-v1 is only tested for SAC. ")

    if LOAD_MODEL:
        try:
            if EXTRA is not None:
                model = SAC.load(f"{DATA_DIR}/sac_pendulum_{MODE}_{EXTRA}.zip", env=env)
            else:
                model = SAC.load(f"{DATA_DIR}/sac_pendulum_{MODE}.zip", env=env)
        except FileNotFoundError:
            print("Model not found. Training a new model.")
            model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-3)
    else:
        model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-3)

    model.learn(
        total_timesteps=1e5,
        callback=LoggingCallback(
            log_interval=1000, model_name="sac", mode=MODE, env_type=ENV_TYPE
        ),
    )
    if EXTRA is not None:
        model.save(f"{DATA_DIR}/sac_pendulum_{MODE}_{EXTRA}.zip")
    else:
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
            if EXTRA is not None:
                model = SAC.load(
                    f"{DATA_DIR}/sac_inverted_double_pendulum_{MODE}_{EXTRA}.zip",
                    env=env,
                )
            else:
                model = SAC.load(
                    f"{DATA_DIR}/sac_inverted_double_pendulum_{MODE}.zip", env=env
                )
        else:
            model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-3)

        model.learn(
            total_timesteps=1e6,
            callback=LoggingCallback(
                log_interval=2000,
                model_name="sac",
                mode=MODE,
                env_type=ENV_TYPE,
                extra=EXTRA,
            ),
        )

        if EXTRA is not None:
            model.save(f"{DATA_DIR}/sac_inverted_double_pendulum_{MODE}_{EXTRA}.zip")
        else:
            model.save(f"{DATA_DIR}/sac_inverted_double_pendulum_{MODE}.zip")

    elif MODEL_TYPE == "PPO":
        if LOAD_MODEL:
            if EXTRA is not None:
                model = PPO.load(
                    f"{DATA_DIR}/ppo_inverted_double_pendulum_{MODE}_{EXTRA}.zip",
                    env=env,
                )
            else:
                model = PPO.load(
                    f"{DATA_DIR}/ppo_inverted_double_pendulum_{MODE}.zip", env=env
                )
        else:
            model = PPO(
                "MlpPolicy",
                env,
                batch_size=128,
                n_steps=128,
                gamma=0.98,
                learning_rate=0.000155,
                ent_coef=1.05e-06,
                clip_range=0.4,
                gae_lambda=0.8,
                max_grad_norm=0.5,
                vf_coef=0.65,
            )

        model.learn(
            total_timesteps=1e6,
            callback=LoggingCallback(
                log_interval=1000,
                model_name="ppo",
                mode=MODE,
                env_type=ENV_TYPE,
                extra=EXTRA,
            ),
        )

        if EXTRA is not None:
            model.save(f"{DATA_DIR}/ppo_inverted_double_pendulum_{MODE}_{EXTRA}.zip")
        else:
            model.save(f"{DATA_DIR}/ppo_inverted_double_pendulum_{MODE}.zip")

elif ENV_TYPE == 2:
    gym.register(
        id="CustomRotaryInvertedDoublePendulum-v1",
        entry_point="custom_envs:CustomRotaryInvertedDoublePendulumEnv",
    )
    env = make_vec_env(
        "CustomRotaryInvertedDoublePendulum-v1",
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
            if EXTRA is not None:
                model = SAC.load(
                    f"{DATA_DIR}/sac_rotary_inverted_double_pendulum_{MODE}_{EXTRA}.zip",
                    env=env,
                )
            else:
                model = SAC.load(
                    f"{DATA_DIR}/sac_rotary_inverted_double_pendulum_{MODE}.zip",
                    env=env,
                )
        else:
            model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-4, ent_coef=0.5)

        model.learn(
            total_timesteps=1e6,
            callback=LoggingCallback(
                log_interval=2000,
                model_name="sac",
                mode=MODE,
                env_type=ENV_TYPE,
                extra=EXTRA,
            ),
        )

        if EXTRA is not None:
            model.save(
                f"{DATA_DIR}/sac_rotary_inverted_double_pendulum_{MODE}_{EXTRA}.zip"
            )
        else:
            model.save(f"{DATA_DIR}/sac_rotary_inverted_double_pendulum_{MODE}.zip")

    elif MODEL_TYPE == "PPO":
        if LOAD_MODEL:
            if EXTRA is not None:
                model = PPO.load(
                    f"{DATA_DIR}/ppo_rotary_inverted_double_pendulum_{MODE}_{EXTRA}.zip",
                    env=env,
                )
            else:
                model = PPO.load(
                    f"{DATA_DIR}/ppo_rotary_inverted_double_pendulum_{MODE}.zip",
                    env=env,
                )
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
            callback=LoggingCallback(
                log_interval=1000,
                model_name="ppo",
                mode=MODE,
                env_type=ENV_TYPE,
                extra=EXTRA,
            ),
        )

        if EXTRA is not None:
            model.save(
                f"{DATA_DIR}/ppo_rotary_inverted_double_pendulum_{MODE}_{EXTRA}.zip"
            )
        else:
            model.save(f"{DATA_DIR}/ppo_rotary_inverted_double_pendulum_{MODE}.zip")
