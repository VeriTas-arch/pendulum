import matplotlib.pyplot as plt
import user_env_gym.double_pendulum as dpend
from Model import SACAgent
from utils import final_plot, resolve_matplotlib_error, showcase

### resolve matplotlib error
resolve_matplotlib_error()
plt.ion()

# create env
env = dpend.DoublePendEnv(reward_mode=4)

# initialize agent, set epoch length
load = True
agent = SACAgent(env=env, load=load)

if load:
    n_showcase = 2
    showcase(
        agent,
        env=dpend.DoublePendEnv(reward_mode=4, render_mode="human"),
        n_showcase=n_showcase,
    )

# train the agent if not load
if not load:
    epochs = int(input("EPOCHS: "))
    save = bool(input("Save Agent? (True or False): "))
    scores = agent.train(epochs, save=save)
    print("------Training Completed------")

    # turn off plotting interactive mode
    plt.ioff()
    plt.plot(agent.save_epi_reward)

    # plot moving average
    final_plot(scores)

    # see how the trained agent performs
    showcase(
        agent, env=dpend.DoublePendEnv(reward_mode=4, render_mode="human"), n_showcase=5
    )
