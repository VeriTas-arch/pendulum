# Let's start with PendulumV1, I guess..

import copy
from collections import deque

import gym
import matplotlib.pyplot as plt
import torch
from Model import *
from tqdm import tqdm
from utils import *

### resolve matplotlib error
resolve_matplotlib_error()
plt.ion()

class DDPGAgent:
    
    def __init__(self, hidden_size, output_size):
        self.epochs = int(input("enter epochs: "))
        self.alpha_actor = 1e-4
        self.alpha_critic = 1e-3
        self.gamma = 0.95
        self.tau = 1e-3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(3, hidden_size, output_size).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_target.load_state_dict(self.actor_target.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.alpha_actor)
        self.actor_loss_fn = None
        self.critic = Critic(4, hidden_size, output_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.load_state_dict(self.critic_target.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.alpha_critic)
        self.critic_loss_fn = torch.nn.MSELoss()

        self.replay_buffer = ExperienceReplay(max_length=20000, batch_size=32)
        self.learning_starts = 1000

        self.best_models = dict()

    # Simulation with learned policy (includes graphic)
    def test_agent(self):
        env = gym.make('Pendulum-v1', render_mode='human')
        n_episodes = 1

        for _ in range(n_episodes):
            curr_state, _ = env.reset()

            terminated = False
            truncated = False

            self.actor.eval()
            while not terminated and not truncated:
                curr_state = torch.from_numpy(curr_state).unsqueeze(0).to(self.device)
                action = self.actor(curr_state)
                action = action.squeeze().detach().to('cpu')
                action = np.array([action])
                self.actor.train()
                next_state, reward, truncated, terminated, _ = env.step(action)
                
                curr_state = next_state
                
        env.close()  

    # Training the agent to learn the policy    
    def train_agent(self):
        env = gym.make('Pendulum-v1')
        noise = OUNoise(env.action_space)
        scores = []

        for i in tqdm(range(self.epochs)):

            curr_state, _ = env.reset()
            noise.reset()
            truncated = False
            terminated = False
            score = 0
            j = 0

            if i %(self.epochs//30) == 0 or i == self.epochs-1:
                plt.clf
                plt.plot(scores)
                plt.draw()
                plt.pause(0.001)

            while not truncated and not terminated:
                j += 1

                self.actor.eval()
                with torch.no_grad():
                    curr_state = torch.from_numpy(curr_state).unsqueeze(0).to(self.device)
                    action = self.actor(curr_state)
                    action = noise.process_action(action, j).squeeze().to('cpu')
                    action = np.array([action])
                self.actor.train()
                next_state, reward, truncated, terminated, _ = env.step(action)
                score += reward
                
                self.replay_buffer.append(curr_state, action, reward, torch.from_numpy(next_state), truncated or terminated)

                if len(self.replay_buffer) > self.learning_starts:
                    state_batch, action_batch, reward_batch, state2_batch, done_batch = self.replay_buffer.sample()

                    # Calculate Critic Loss 
                    Q = self.critic.forward(state_batch, action_batch)
                    action2_batch = self.actor_target.forward(state2_batch)
                    
                    with torch.no_grad():
                        Q_next = self.critic_target(state2_batch, action2_batch.detach())
                    
                    Q_target = reward_batch + self.gamma * (1-done_batch)*Q_next
                    critic_loss = self.critic_loss_fn(Q, Q_target)

                    # Calculate Actor Loss
                    actor_loss = -self.critic.forward(state_batch, self.actor.forward(state_batch)).mean()

                    # update networks
                    # NOTE: 반드시 actor를 먼저 update 
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    # update target networks
                    for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                    for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                curr_state = next_state

            scores.append(score)

        return scores
    

# test_env()
ddpg_agent = DDPGAgent(hidden_size=64, output_size=1)
scores = ddpg_agent.train_agent()

plt.ioff()
plt.show()

ddpg_agent.test_agent()
