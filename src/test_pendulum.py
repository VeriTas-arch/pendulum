from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import PPO, SAC

from custom_wrapper import PerturbWrapper

ENV_TYPE = 1
MODEL_TYPE = "SAC"  # SAC or PPO
MODE = "stable"  # test for swing up, stable for stable control
MODE_STR = "swing up" if MODE == "test" else "stable control"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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
    env_name = "Pendulum-v1"
    env = gym.make(env_name, render_mode="human", mode=MODE)

    if MODEL_TYPE == "SAC":
        model = SAC.load(f"{DATA_DIR}/sac_pendulum_{MODE}.zip", env=env)
    elif MODEL_TYPE == "PPO":
        model = PPO.load(f"{DATA_DIR}/ppo_pendulum_{MODE}.zip", env=env)
elif ENV_TYPE == 1:
    gym.register(
        id="CustomInvertedDoublePendulum-v1",
        entry_point="custom_envs:CustomInvertedDoublePendulumEnv",
    )
    env = PerturbWrapper(
        gym.make("CustomInvertedDoublePendulum-v1", render_mode="human", mode=MODE)
    )
    if MODEL_TYPE == "SAC":
        model = SAC.load(f"{DATA_DIR}/sac_inverted_double_pendulum_{MODE}.zip")
    elif MODEL_TYPE == "PPO":
        model = PPO.load(f"{DATA_DIR}/ppo_inverted_double_pendulum_{MODE}.zip")


obs, info = env.reset()
clock = pygame.time.Clock()
perturbation = np.zeros(env.action_space.shape)

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
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    env.render()
    # 在pygame窗口显示perturbation的大小
    screen.fill((255, 255, 255))
    doc_str = (
        f"mode: {MODE_STR}\n"
        f"model type: {MODEL_TYPE}\n"
        f"current perturbation: {perturbation}"
    )
    text = font.render(doc_str, True, (0, 0, 0))
    screen.blit(text, (20, 40))
    pygame.display.flip()
    clock.tick(60)

env.close()
pygame.quit()
