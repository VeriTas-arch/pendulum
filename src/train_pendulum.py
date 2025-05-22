import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

import utils
from custom_callback import LoggingCallback

ENV_TYPE = 2  # 0 for Pendulum, 1 for InvertedDoublePendulum, 2 for RotaryInvertedDoublePendulum, 3 for RotaryInvertedPendulum
MODEL_TYPE = "SAC"  # SAC or PPO
MODE = "test"  # test for swing up, stable for stable control
LOAD_MODEL = False  # 是否加载模型
EXTRA = "train_test_1"  # 额外的后缀，不加则设为 None


if ENV_TYPE == 0:
    env_name = "Pendulum-v1"
    env = make_vec_env(env_name, n_envs=4)

    if MODE != "stable":
        raise ValueError("Pendulum-v1 is only tested for stable control. ")
    if MODEL_TYPE != "SAC":
        raise ValueError("Pendulum-v1 is only tested for SAC. ")

    if LOAD_MODEL:
        model = utils.load_model(env, ENV_TYPE, MODEL_TYPE, MODE, EXTRA)
    else:
        model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-3)

    model.learn(
        total_timesteps=1e5,
        callback=LoggingCallback(
            log_interval=1000, model_name="SAC", mode=MODE, env_type=ENV_TYPE
        ),
    )

    utils.save_model(model, ENV_TYPE, MODEL_TYPE, MODE, EXTRA)

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
            model = utils.load_model(env, ENV_TYPE, MODEL_TYPE, MODE, EXTRA)
        else:
            model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-3)

        model.learn(
            total_timesteps=1e6,
            callback=LoggingCallback(
                log_interval=2000,
                model_name="SAC",
                mode=MODE,
                env_type=ENV_TYPE,
                extra=EXTRA,
            ),
        )

        utils.save_model(model, ENV_TYPE, MODEL_TYPE, MODE, EXTRA)

    elif MODEL_TYPE == "PPO":
        if LOAD_MODEL:
            model = utils.load_model(env, ENV_TYPE, MODEL_TYPE, MODE, EXTRA)
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
                model_name="PPO",
                mode=MODE,
                env_type=ENV_TYPE,
                extra=EXTRA,
            ),
        )

        utils.save_model(model, ENV_TYPE, MODEL_TYPE, MODE, EXTRA)

elif ENV_TYPE == 2:
    gym.register(
        id="CustomRotaryInvertedDoublePendulum-v1",
        entry_point="custom_envs:CustomRotaryInvertedDoublePendulumEnv",
    )
    env = make_vec_env(
        "CustomRotaryInvertedDoublePendulum-v1",
        n_envs=4,
        wrapper_class=gym.wrappers.TimeLimit,
        wrapper_kwargs={"max_episode_steps": 1000},
        env_kwargs={
            "mode": MODE,
            # "render_mode": "human"
        },
    )

    if MODEL_TYPE == "SAC":
        if LOAD_MODEL:
            model = utils.load_model(env, ENV_TYPE, MODEL_TYPE, MODE, EXTRA)
        else:
            model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-4, ent_coef=0.5)

        model.learn(
            total_timesteps=1e6,
            callback=LoggingCallback(
                log_interval=2000,
                model_name="SAC",
                mode=MODE,
                env_type=ENV_TYPE,
                extra=EXTRA,
            ),
        )

        utils.save_model(model, ENV_TYPE, MODEL_TYPE, MODE, EXTRA)

    else:
        raise NotImplementedError

elif ENV_TYPE == 3:
    gym.register(
        id="CustomRotaryInvertedPendulum-v1",
        entry_point="custom_envs:CustomRotaryInvertedPendulumEnv",
    )
    env = make_vec_env(
        "CustomRotaryInvertedPendulum-v1",
        n_envs=4,
        wrapper_class=gym.wrappers.TimeLimit,
        wrapper_kwargs={"max_episode_steps": 1000},
        env_kwargs={
            "mode": MODE,
            # "render_mode": "human"
        },
    )

    if MODEL_TYPE == "SAC":
        if LOAD_MODEL:
            model = utils.load_model(env, ENV_TYPE, MODEL_TYPE, MODE, EXTRA)
        else:
            model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-3)

        model.learn(
            total_timesteps=1e6,
            callback=LoggingCallback(
                log_interval=2000,
                model_name="SAC",
                mode=MODE,
                env_type=ENV_TYPE,
                extra=EXTRA,
            ),
        )

        utils.save_model(model, ENV_TYPE, MODEL_TYPE, MODE, EXTRA)

    else:
        raise NotImplementedError

else:
    raise NotImplementedError(
        f"Environment type {ENV_TYPE} is not supported. "
        "Please choose from 0 (Pendulum), 1 (InvertedDoublePendulum), 2 (RotaryInvertedDoublePendulum), or 3 (RotaryInvertedPendulum)."
    )
