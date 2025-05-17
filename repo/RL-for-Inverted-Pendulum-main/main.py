from animate import animate_episode, read_ep_from_log, run_episode
from dataModels import Config, read_data_from_yaml
from models import DDPGAgent, DQNAgent, REINFORCEAgent

from pendulum import Pendulum

config = read_data_from_yaml("InputParameters.yaml", Config)

run_params = config.run
animate_params = config.animate
pendulum_params = config.pendulum
model_params = config.model

env = Pendulum(config=config)

if model_params.type == "REINFORCE":
    agent = REINFORCEAgent(config=model_params)
elif model_params.type == "DQN":
    agent = DQNAgent(config=model_params)
elif model_params.type == "DDPG":
    agent = DDPGAgent(config=model_params)

if run_params.type == "train":
    agent.train(env, config)
elif run_params.type == "test":
    if animate_params.from_log.enable:
        episode_data = read_ep_from_log(
            animate_params.from_log.file_path, animate_params.from_log.episode
        )
        animate_episode(episode_data, config)
    else:
        state_history = run_episode(env, agent)
        animate_episode(state_history, config)
