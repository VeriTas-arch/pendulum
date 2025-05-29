import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame

import utils
from custom_wrapper import PerturbWrapper

ENV_TYPE = 2
MODEL_TYPE = "SAC"  # SAC or PPO
ENV_MODE = "test"  # test for swing up, stable for stable control
MODE_STR = "swing up" if ENV_MODE == "test" else "stable control"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


STAGE1_EXTRA = "new_obs_2"
STAGE2_EXTRA = "new_obs_trans"


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

if ENV_TYPE == 2:
    gym.register(
        id="CustomRotaryInvertedDoublePendulum-v1",
        entry_point="custom_envs:CustomRotaryInvertedDoublePendulumEnv",
    )
    env = PerturbWrapper(
        gym.make(
            "CustomRotaryInvertedDoublePendulum-v1", render_mode="human", mode=ENV_MODE
        )
    )
    swingup_model = utils.load_model(env, ENV_TYPE, MODEL_TYPE, "test", STAGE1_EXTRA)
    stable_model = utils.load_model(env, ENV_TYPE, MODEL_TYPE, "stable", STAGE2_EXTRA)

else:
    raise NotImplementedError(
        f"ENV_TYPE {ENV_TYPE} is not supported. Please check the code."
    )

obs, info = env.reset()
clock = pygame.time.Clock()
perturbation = np.zeros(env.action_space.shape)

maxy = 0
done = False
action = None
last_action = None

is_up = False

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
                pygame.time.delay(1000)

    # 处理按键输入扰动
    perturbation = handle_keyboard_input(step_size=0.1)
    env.set_perturbation(perturbation)

    # 模型预测 + 应用扰动
    theta1 = env.unwrapped.data.qpos[1]

    action1, _ = swingup_model.predict(obs, deterministic=True)
    action2, _ = stable_model.predict(obs, deterministic=True)

    # min end 对应alpha=1，max end 对应alpha=0
    # max_end = 0.22
    # min_end = 0.16

    # alpha = (min_end - np.sin(theta1)) / (max_end - min_end)
    # alpha = np.clip(alpha, 0.0, 1.0)

    # action = (1 - alpha) * action1 + alpha * action2

    # if abs(theta1) > 0.15 and not env.unwrapped.isUp:
    #     action, _ = stage1_model.predict(obs, deterministic=True)
    # elif 0.08 < abs(theta1) < 0.15:
    #     env.unwrapped.isUp = True
    #     # 0.08对应alpha=1，0.15对应alpha=0
    #     alpha = (0.08 - abs(theta1)) / (0.13 - 0.08)
    #     alpha = np.clip(alpha, 0.0, 1.0)
    #     action1, _ = stage1_model.predict(obs, deterministic=True)
    #     action2, _ = stage2_model.predict(obs, deterministic=True)
    #     action = (1 - alpha) * action1 + alpha * action2
    # else:
    #     action, _ = stage2_model.predict(obs, deterministic=True)

    if abs(theta1) > 0.4 and not env.unwrapped.isUp:
        action, _ = swingup_model.predict(obs, deterministic=True)
        last_action = action
    elif 0.18 < abs(theta1) < 0.4:
        env.unwrapped.isUp = True
        action = last_action
        action[0] = 0.15
    else:
        action, _ = stable_model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    # print(reward)
    # print(info)
    # print("action:", action)

    env.render()
    screen.fill((255, 255, 255))

    # 显示文本
    doc_str = (
        f"mode: {MODE_STR}\n"
        f"model type: {MODEL_TYPE}\n"
        f"qpos: {np.around(env.unwrapped.data.qpos, 3)}\n"
        f"qvel: {np.around(env.unwrapped.data.qvel, 3)}\n"
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
