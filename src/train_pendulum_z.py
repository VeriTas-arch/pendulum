import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

import utils
from custom_callback import LoggingCallback

ENV_TYPE = 2  # 2 for RotaryInvertedDoublePendulum
MODEL_TYPE = "SAC"  # SAC or PPO
MODE = "stable"  # test for swing up, stable for stable control
LOAD_MODEL = True  # 是否加载模型
EXTRA = "high_speed"  # 额外的后缀，semi_high_speed, high_speed, alg_test, etc.

if ENV_TYPE == 2:
    gym.register(
        id="CustomRotaryInvertedDoublePendulum-v3",
        entry_point="v_envs:CustomRotaryInvertedDoublePendulumEnv",
    )
    env = make_vec_env(
        "CustomRotaryInvertedDoublePendulum-v3",
        n_envs=16,
        wrapper_class=gym.wrappers.TimeLimit,
        wrapper_kwargs={"max_episode_steps": 16000},
        env_kwargs={"mode": MODE, "custom_xml_file": utils.HIGH_SPEED_XML_DIR},
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
                log_interval=10000,
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
