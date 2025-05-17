from copy import deepcopy

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import ReplayBuffer, live_plot


class SACAgent:

    def __init__(self, env):
        self.GAMMA = 0.99
        self.BATCH_SIZE = 128
        self.BUFFER_SIZE = 1_000_000
        self.ACTOR_LEARNING_RATE = 3e-4
        self.CRITIC_LEARNING_RATE = 3e-4
        self.TAU = 5e-3
        self.ALPHA = 1/3

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        self.actor = Actor(self.action_dim, self.action_bound)
        
        self.critic_1 = Critic(self.action_dim, self.state_dim)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())

        self.critic_2 = Critic(self.action_dim, self.state_dim)
        self.target_critic_2 = deepcopy(self.critic_2)
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.ACTOR_LEARNING_RATE)
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=self.CRITIC_LEARNING_RATE)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=self.CRITIC_LEARNING_RATE)

        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        self.save_epi_reward = []

    def save_agent(self):
        import os
        from datetime import datetime 

        time = datetime.now().time().strftime('%H:%M')
        PATH = f'./saved_models/{time}'
        os.makedirs(PATH, exist_ok=True)

        torch.save(self.actor.state_dict(), PATH + '/actor.pth')
        torch.save(self.critic_1.state_dict(), PATH + '/critic_1.pth')
        torch.save(self.critic_2.state_dict(), PATH + '/critic_2.pth')
        torch.save(self.target_critic_1.state_dict(), PATH + "/target_critic_1.pth")
        torch.save(self.target_critic_2.state_dict(), PATH + "/target_critic_2.pth")

        print("------Agent Saved------")

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            mu, std = self.actor(state)
            if deterministic:
                action = mu
            else:
                action, _ = self.actor.sample_normal(mu, std, reparam=False)
        return action.numpy()[0]
    
    def update_target_network(self):
        phi_1 = self.critic_1.state_dict()
        phi_2 = self.critic_2.state_dict()
        target_phi_1 = self.target_critic_1.state_dict()
        target_phi_2 = self.target_critic_2.state_dict()

        for key in phi_1.keys():
            target_phi_1[key] = self.TAU * phi_1[key] + (1 - self.TAU) * target_phi_1[key]
            target_phi_2[key] = self.TAU * phi_2[key] + (1 - self.TAU) * target_phi_2[key]

        self.target_critic_1.load_state_dict(target_phi_1)
        self.target_critic_2.load_state_dict(target_phi_2)

    def critic_learn(self, states, actions, q_targets):
        q_1 = self.critic_1([states, actions])
        loss_1 = torch.mean( (q_1 - q_targets) ** 2)

        self.critic_1_opt.zero_grad()
        loss_1.backward()
        self.critic_1_opt.step()

        q_2 = self.critic_2([states, actions])
        loss_2 = torch.mean( (q_2 - q_targets) ** 2)

        self.critic_2_opt.zero_grad()
        loss_2.backward()
        self.critic_2_opt.step()

    def actor_learn(self, states):
        mu, std = self.actor(states)
        actions, log_pdfs = self.actor.sample_normal(mu, std, reparam=True)
        log_pdfs = log_pdfs.squeeze(1)
        soft_q_1 = self.critic_1([states, actions])
        soft_q_2 = self.critic_2([states, actions])
        soft_q = torch.min(soft_q_1, soft_q_2)

        loss = torch.mean(self.ALPHA * log_pdfs - soft_q)

        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

    def q_target(self, rewards, q_values, dones):
        y_k = torch.from_numpy(q_values).clone().detach().numpy()

        for i in range(q_values.shape[0]):
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]

        return torch.from_numpy(y_k)
    
    def train(self, max_episode_num, save=False):
        #self.update_target_network()

        for ep in tqdm(range(max_episode_num)):
            time, episode_reward, truncated, terminated = 0, 0, False, False
            state, _ = self.env.reset()

            if ep % 10 == 0: 
                live_plot(self.save_epi_reward)

            while not truncated and not terminated and time < 500:
                action = self.get_action(torch.from_numpy(state).to(torch.float32))
                action = np.clip(action, -self.action_bound, self.action_bound)
                action = np.array(action).reshape(-1)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                self.buffer.add_buffer(state, action, reward, next_state, terminated or truncated)

                if self.buffer.count > 100:
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                    with torch.no_grad():
                        next_mu, next_std = self.actor(torch.tensor(next_states).to(torch.float32))
                        next_actions, next_log_pdf = self.actor.sample_normal(next_mu, next_std, reparam=True)

                        target_qs_1 = self.target_critic_1([torch.tensor(next_states).to(torch.float32), next_actions])
                        target_qs_2 = self.target_critic_2([torch.tensor(next_states).to(torch.float32), next_actions])
                        target_qs = torch.min(target_qs_1, target_qs_2)

                        target_qi = target_qs - self.ALPHA * next_log_pdf
                    
                    y_i = self.q_target(rewards, target_qi.numpy(), dones)

                    self.critic_learn(torch.tensor(states).to(torch.float32), torch.tensor(actions).to(torch.float32), torch.tensor(y_i).to(torch.float32))
                    self.actor_learn(torch.tensor(states).to(torch.float32))
                    
                    self.update_target_network()

                state = next_state
                episode_reward += reward
                time += 1

            self.save_epi_reward.append(episode_reward)
            
        if save == True:
            self.save_agent()

        return self.save_epi_reward

class Actor(nn.Module):

    def __init__(self, action_dim, action_bound):
        super().__init__()

        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = [1e-6, 1.0]

        self.l1 = nn.Linear(6, 128) # state_dim = 6
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 16)

        self.mu = nn.Linear(16, self.action_dim)
        self.std = nn.Linear(16, self.action_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

        mu = torch.tanh(self.mu(x))
        std = F.softplus(self.std(x))

        mu = (lambda x: x * self.action_bound)(mu)
        std = torch.clamp(std, self.std_bound[0], self.std_bound[1])

        return mu, std

    def sample_normal(self, mu, std, reparam):
        normal_dist = D.Normal(mu, std)
        if reparam:
            action = normal_dist.rsample()
        else:
            action = normal_dist.sample()
        action = torch.clamp(action, -self.action_bound, self.action_bound)
        log_pdf = normal_dist.log_prob(action)
        log_pdf = torch.sum(log_pdf, dim=-1, keepdim=True)

        return action, log_pdf

class Critic(nn.Module):
    
    def __init__(self, action_dim, state_dim):
        super().__init__()

        self.action_dim = action_dim
        self.state_dim = 6

        self.v1 = nn.Linear(self.state_dim, 128)
        self.a1 = nn.Linear(self.action_dim, 128)

        self.l2 = nn.Linear(256, 128) # NOTE: 256 = v1.size + a1.size
        self.l3 = nn.Linear(128, 64)
        self.q =  nn.Linear(64, 1)

    def forward(self, state_action):
        state, action = state_action[0], state_action[1]

        v = F.relu(self.v1(state))
        a = F.relu(self.a1(action.view(-1, 1)))
        x = torch.cat([v, a], dim=-1)
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.q(x)

        return x
