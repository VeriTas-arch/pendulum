from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
import time

import utils
from custom_wrapper import PerturbWrapper

ENV_TYPE = 2
MODEL_TYPE = "SAC"  # SAC or PPO
MODE = "stable"  # test for swing up, stable for stable control
MODE_STR = "swing up" if MODE == "test" else "stable control"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXTRA = "new_obs"  # 额外的后缀，不加则设为 None


def handle_keyboard_input(step_size=0.1):
    keys = pygame.key.get_pressed()
    perturbation = 0

    if keys[pygame.K_LEFT]:
        perturbation = -step_size
    if keys[pygame.K_RIGHT]:
        perturbation = step_size

    return perturbation


# 初始化pygame窗口
pygame.init()
screen = pygame.display.set_mode((600, 450))
pygame.display.set_caption("Perturbation Controller")
font = pygame.font.SysFont("consolas", 24)

# 初始化环境和模型
if ENV_TYPE == 0:
    raise ValueError(
        "Pendulum-v1 is not supported in the perturbation mode now."
        " Refer to `test_pendulum_old.py` if you want to test it."
    )

elif ENV_TYPE == 1:
    gym.register(
        id="CustomInvertedDoublePendulum-v1",
        entry_point="custom_envs:CustomInvertedDoublePendulumEnv",
    )
    env = PerturbWrapper(
        gym.make("CustomInvertedDoublePendulum-v1", render_mode="human", mode=MODE)
    )
    model = utils.load_model(env, ENV_TYPE, MODEL_TYPE, MODE, EXTRA)

elif ENV_TYPE == 2:
    gym.register(
        id="CustomRotaryInvertedDoublePendulum-v1",
        entry_point="custom_envs:CustomRotaryInvertedDoublePendulumEnv",
    )
    env = PerturbWrapper(
        gym.make(
            "CustomRotaryInvertedDoublePendulum-v1", render_mode="human", mode=MODE
        )
    )
    model = utils.load_model(env, ENV_TYPE, MODEL_TYPE, MODE, EXTRA)
    # stable_model = utils.load_model(env, ENV_TYPE, MODEL_TYPE, "stable", "train_test_3")

elif ENV_TYPE == 3:
    gym.register(
        id="CustomRotaryInvertedPendulum-v1",
        entry_point="custom_envs:CustomRotaryInvertedPendulumEnv",
    )
    env = PerturbWrapper(
        gym.make("CustomRotaryInvertedPendulum-v1", render_mode="human", mode=MODE)
    )
    model = utils.load_model(env, ENV_TYPE, MODEL_TYPE, MODE, EXTRA)

else:
    raise NotImplementedError(
        f"ENV_TYPE {ENV_TYPE} is not supported. Please check the code."
    )

obs, info = env.reset()
clock = pygame.time.Clock()
perturbation = np.zeros(env.action_space.shape)

maxy = 0
done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            done = True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                obs, info = env.reset()
                # 略微延迟，便于观察
                pygame.time.delay(2000)

    # 处理按键输入扰动
    perturbation = handle_keyboard_input(step_size=0.2)
    env.set_perturbation(perturbation)

    # 模型预测 + 应用扰动
    _, _, y = env.unwrapped.data.site_xpos[4]

    # print("qpos:", env.unwrapped.data.qpos)

    # alpha = np.clip((y - 0.45) / (0.55 - 0.45), 0.0, 1.0)

    # action_swingup = model.predict(obs, deterministic=True)[0]
    # action_stable = stable_model.predict(obs, deterministic=True)[0]

    # action = (1 - alpha) * action_swingup + alpha * action_stable

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # print(reward)
    print(info)

    # print("action:", action)

    if y > 0.5:
        if y > maxy:
            maxy = y
        # print("maxy:", maxy)
        print("qpos:", env.unwrapped.data.qpos)
        v0, v1, v2 = env.unwrapped.data.qvel
        # print("v0:", v0, "v1:", v1, "v2:", v2)

    env.render()
    screen.fill((255, 255, 255))

    # 显示文本
    doc_str = (
        f"mode: {MODE_STR}\n"
        f"model type: {MODEL_TYPE}\n"
        f"current perturbation: {perturbation}\n"
        "\n"
        "press 'r' to reset\n"
        "press \u2190 and \u2192 to apply perturbation\n"
    )
    text = font.render(doc_str, True, (0, 0, 0))
    screen.blit(text, (20, 40))

    pygame.display.flip()
    time.sleep(0.02)
    clock.tick(60)

env.close()
pygame.quit()
