import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

import utils
from custom_callback import LoggingCallback

ENV_TYPE = 2  # 0 for Pendulum, 1 for InvertedDoublePendulum, 2 for RotaryInvertedDoublePendulum, 3 for RotaryInvertedPendulum
MODEL_TYPE = "SAC"  # SAC or PPO
MODE = "test"  # test for swing up, stable for stable control
LOAD_MODEL = False  # 是否加载模型
EXTRA = "test_obs_no_limit"  # 额外的后缀，不加则设为 None


if ENV_TYPE == 2:
    gym.register(
        id="CustomRotaryInvertedDoublePendulum-v2",
        entry_point="new_envs:CustomRotaryInvertedDoublePendulumEnv",
    )
    env = make_vec_env(
        "CustomRotaryInvertedDoublePendulum-v2",
        n_envs=32,
        wrapper_class=gym.wrappers.TimeLimit,
        wrapper_kwargs={"max_episode_steps": 1000},
        env_kwargs={
            "mode": MODE,
            "custom_xml_file": utils.PINOCCHIO_XML_DIR,
            # "render_mode": "human"
        },
    )

    if MODEL_TYPE == "SAC":
        if LOAD_MODEL:
            model = utils.load_model(env, ENV_TYPE, MODEL_TYPE, MODE, EXTRA)
        else:
            model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=1e-4,
                tensorboard_log=str(utils.LOG_DIR),
            )

        model.learn(
            total_timesteps=1e7,
            callback=LoggingCallback(
                log_interval=1000,
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
    raise NotImplementedError(f"Environment type {ENV_TYPE} is not supported. ")
