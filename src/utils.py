from pathlib import Path
import numpy as np

from stable_baselines3 import PPO, SAC

DATA_DIR = Path(__file__).resolve().parent.parent / "data"  # 数据目录
LOG_DIR = Path(__file__).resolve().parent.parent / "log"  # 日志目录
CSV_DIR = DATA_DIR / "csv"  # CSV目录
ASSET_DIR = f"{Path(__file__).parent.parent}/assets"
PINOCCHIO_XML_DIR = f"{ASSET_DIR}/pinoc_inverted_double_pendulum.xml"


def load_model(env, env_type, model_type, mode, extra=None):
    if mode not in ["test", "stable"]:
        raise ValueError("Invalid mode. Choose 'test' or 'stable'.")

    env_name = get_env_name(env_type)

    if model_type == "SAC":
        if extra is not None:
            model = SAC.load(f"{DATA_DIR}/sac_{env_name}_{mode}_{extra}.zip", env=env)
        else:
            model = SAC.load(f"{DATA_DIR}/sac_{env_name}_{mode}.zip", env=env)
    elif model_type == "PPO":
        if extra is not None:
            model = PPO.load(f"{DATA_DIR}/ppo_{env_name}_{mode}_{extra}.zip", env=env)
        else:
            model = PPO.load(f"{DATA_DIR}/ppo_{env_name}_{mode}.zip", env=env)
    else:
        raise ValueError("Invalid model type. Choose 'SAC' or 'PPO'.")

    return model


def save_model(model, env_type, model_type, mode, extra=None):
    if mode not in ["test", "stable"]:
        raise ValueError("Invalid mode. Choose 'test' or 'stable'.")
    if model_type not in ["SAC", "PPO", "HER"]:
        raise ValueError("Invalid model type. Choose 'SAC', 'HER' or 'PPO'.")

    env_name = get_env_name(env_type)

    save_path = ""
    if extra is not None:
        save_path = DATA_DIR / f"{model_type.lower()}_{env_name}_{mode}_{extra}.zip"
    else:
        save_path = DATA_DIR / f"{model_type.lower()}_{env_name}_{mode}.zip"

    model.save(str(save_path))


def get_env_name(env_type):
    if env_type == 0:
        return "pendulum"
    elif env_type == 1:
        return "inverted_double_pendulum"
    elif env_type == 2:
        return "rotary_inverted_double_pendulum"
    elif env_type == 3:
        return "rotary_inverted_pendulum"
    else:
        raise ValueError("Invalid environment type. Choose 0, 1, 2, or 3.")


# ! DEPRECATED
def rotary_to_linear_obs(obs_rotary):
    L = 0.2
    # 从 obs_rotary 中解析数据
    cos_theta0, sin_theta0 = obs_rotary[0], obs_rotary[1]
    cos_theta1, sin_theta1 = obs_rotary[2], obs_rotary[3]
    cos_theta2, sin_theta2 = obs_rotary[4], obs_rotary[5]
    qvel0, qvel1, qvel2 = obs_rotary[6], obs_rotary[7], obs_rotary[8]

    # 使用 atan2 恢复角度
    theta0 = np.arctan2(sin_theta0, cos_theta0)
    theta1 = np.arctan2(sin_theta1, cos_theta1)
    theta2 = np.arctan2(sin_theta2, cos_theta2)

    xpos = L * theta0

    # 构造线性摆的 obs
    linear_obs = np.concatenate(
        [
            [xpos],  # 小车位置为 L*theta0
            [np.sin(theta1), np.sin(theta2)],
            [np.cos(theta1), np.cos(theta2)],
            np.clip([qvel0, qvel1, qvel2], -10, 10),
            [0.0],  # 没有 constraint 力，设为 0
        ]
    ).astype(np.float32)

    return linear_obs
