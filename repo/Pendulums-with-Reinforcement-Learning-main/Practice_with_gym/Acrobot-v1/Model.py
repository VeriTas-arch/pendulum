from copy import deepcopy

import numpy as np
from networks import *
from tqdm import tqdm
from utils import ReplayBuffer, live_plot


class SACAgent:

    def __init__(self, env):
        self.GAMMA = 0.99
        self.BATCH_SIZE = 256
        self.BUFFER_SIZE = 1e7
        self.ACTOR_LEARNING_RATE = 3e-4
        self.CRITIC_LEARNING_RATE = 3e-4
        self.TAU = 2e-3
        self.ALPHA = 0.5

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

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=self.ACTOR_LEARNING_RATE
        )
        self.critic_1_opt = torch.optim.Adam(
            self.critic_1.parameters(), lr=self.CRITIC_LEARNING_RATE
        )
        self.critic_2_opt = torch.optim.Adam(
            self.critic_2.parameters(), lr=self.CRITIC_LEARNING_RATE
        )

        self.buffer = ReplayBuffer(self.BUFFER_SIZE)
        self.learning_starts = 100

        self.save_epi_reward = []
        self.save_actions = []

    def get_action(self, state):
        with torch.no_grad():
            mu, std = self.actor(state)
            action, _ = self.actor.sample_normal(mu, std)
        return action.numpy()

    def update_target_network(self):
        phi_1 = self.critic_1.state_dict()
        phi_2 = self.critic_2.state_dict()
        target_phi_1 = self.target_critic_1.state_dict()
        target_phi_2 = self.target_critic_2.state_dict()

        for key in phi_1.keys():
            target_phi_1[key] = (
                self.TAU * phi_1[key] + (1 - self.TAU) * target_phi_1[key]
            )
            target_phi_2[key] = (
                self.TAU * phi_2[key] + (1 - self.TAU) * target_phi_2[key]
            )

        self.target_critic_1.load_state_dict(target_phi_1)
        self.target_critic_2.load_state_dict(target_phi_2)

    def critic_learn(self, states, actions, q_targets):
        q_1 = self.critic_1([states, actions])
        loss_1 = torch.mean((q_1 - q_targets) ** 2)

        self.critic_1_opt.zero_grad()
        loss_1.backward()
        self.critic_1_opt.step()

        q_2 = self.critic_2([states, actions])
        loss_2 = torch.mean((q_2 - q_targets) ** 2)

        self.critic_2_opt.zero_grad()
        loss_2.backward()
        self.critic_2_opt.step()

    def actor_learn(self, states):
        mu, std = self.actor(states)
        actions, log_pdfs = self.actor.sample_normal(mu, std)
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

    def train(self, max_episode_num):
        self.update_target_network()

        for ep in tqdm(range(max_episode_num)):
            time, episode_reward, truncated, terminated = 0, 0, False, False
            state, _ = self.env.reset()

            if ep % 10 == 0:
                live_plot(self.save_epi_reward, self.save_actions)

            while not truncated and not terminated and time < 500:
                action = self.get_action(torch.from_numpy(state))
                action = np.clip(action, -self.action_bound, self.action_bound)

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                self.save_actions.append(reward)

                self.buffer.add_buffer(
                    state, action, reward, next_state, terminated or truncated
                )

                if self.buffer.count > self.learning_starts:
                    states, actions, rewards, next_states, dones = (
                        self.buffer.sample_batch(self.BATCH_SIZE)
                    )

                    next_mu, next_std = self.actor(torch.tensor(next_states))
                    next_actions, next_log_pdf = self.actor.sample_normal(
                        next_mu, next_std
                    )

                    target_qs_1 = self.target_critic_1(
                        [torch.tensor(next_states), next_actions]
                    )
                    target_qs_2 = self.target_critic_2(
                        [torch.tensor(next_states), next_actions]
                    )
                    target_qs = torch.min(target_qs_1, target_qs_2)

                    target_qi = target_qs - self.ALPHA * next_log_pdf

                    y_i = self.q_target(rewards, target_qi.detach().numpy(), dones)

                    self.critic_learn(
                        torch.from_numpy(states), torch.from_numpy(actions), y_i
                    )
                    self.actor_learn(torch.from_numpy(states))

                    self.update_target_network()

                state = next_state
                episode_reward += reward
                time += 1

            self.save_epi_reward.append(episode_reward)
            # print(episode_reward)

        return self
