from pathlib import Path

from stable_baselines3 import PPO, SAC

DATA_DIR = Path(__file__).resolve().parent.parent / "data"  # 数据目录


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
    if model_type not in ["SAC", "PPO"]:
        raise ValueError("Invalid model type. Choose 'SAC' or 'PPO'.")

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
