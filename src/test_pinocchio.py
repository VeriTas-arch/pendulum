from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame

import utils
from custom_wrapper import PerturbWrapper


MODE = "test"  # test for swing up, stable for stable control

LIN_EXTRA = None
ROT_EXTRA = "new_obs"  # 额外的后缀，不加则设为 None

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

gym.register(
    id="CustomRotaryInvertedDoublePendulum-v1",
    entry_point="custom_envs:CustomRotaryInvertedDoublePendulumEnv",
)
env = PerturbWrapper(
    gym.make(
        "CustomRotaryInvertedDoublePendulum-v1", render_mode="human", mode=MODE, custom_xml_file=utils.PINOCCHIO_XML_DIR
    )
)

lin_model = utils.load_model(env, env_type=1, model_type="SAC", mode=MODE, extra=LIN_EXTRA)
rot_model = utils.load_model(env, env_type=2, model_type="SAC", mode=MODE, extra=ROT_EXTRA)


obs, info = env.reset()
obs = utils.rotary_to_linear_obs(obs)

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

    action, _ = lin_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    obs = utils.rotary_to_linear_obs(obs)

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
    clock.tick(60)

env.close()
pygame.quit()
