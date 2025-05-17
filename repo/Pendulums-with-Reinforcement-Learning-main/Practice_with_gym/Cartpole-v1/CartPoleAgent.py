import copy

import matplotlib.pyplot as plt
import torch
import user_env_gym.cartpole_cont as cart
from Model import *
from utils import *

# RESOLVE MATPLOTLIB ERROR
matplotlib_error()

class DDPGAgent:
    
    def __init__(self, hidden_size, output_size):
        self.epochs = 4000
        self.alpha_actor = 1e-4
        self.alpha_critic = 1e-3
        self.gamma = 0.99
        self.tau = 1e-2
        self.actor = Actor(4, hidden_size, output_size)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor_target.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.alpha_actor)
        self.actor_loss_fn = None
        self.critic = Critic(5, hidden_size, output_size)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic_target.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.alpha_critic)
        self.critic_loss_fn = torch.nn.MSELoss()

        self.replay_buffer = ExperienceReplay(max_length=5000, batch_size=128)   

    # Testcode to check operation correctness of simulator
    @staticmethod
    def test_env():
        env = cart.CartPoleEnv(render_mode="human", reward_mode = 0)

        n_episodes = 1
        for episode in range(n_episodes):
            state, _ = env.reset()

            terminated = False
            truncated = False

            while not terminated or truncated:
                state = torch.tensor(state, dtype=torch.float32)
                
                action = 0
                state, reward, terminated, truncated, info = env.step(action)
                
        env.close() 

    # Simulation with learned policy (includes graphic)
    def test_agent(self):
        env = cart.CartPoleEnv(render_mode = 'human', reward_mode = 0)
        n_episodes = 1
        for episode in range(n_episodes):
            state, _ = env.reset()

            terminated = False
            truncated = False

            while not terminated and not truncated:
                state = torch.tensor(state, dtype=torch.float32)
                
                action = self.actor(torch.FloatTensor(state))
                state, reward, terminated, truncated, info = env.step(action)
                
        env.close()  

    # Training the agent to learn the policy    
    def train_agent(self):
        env = cart.CartPoleEnv(reward_mode = 2)
        noise = OUNoise(env.action_space)
        scores = []

        for i in range(self.epochs):
            print(f"currently on epoch {i}/{self.epochs}")  

            curr_state, _ = env.reset()
            noise.reset()
            truncated = False
            terminated = False
            j = 0

            while not truncated and not terminated:
                curr_state = torch.from_numpy(curr_state)
                action = self.actor(curr_state)
                action = noise.process_action(action, j)
                next_state, reward, truncated, terminated, _ = env.step(action)
                self.replay_buffer.append(curr_state, action, reward, torch.from_numpy(next_state), truncated or terminated)

                if len(self.replay_buffer) > self.replay_buffer.batch_size:
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
                j += 1
                if truncated or terminated: scores.append(j); break

        return scores


ddpg_agent = DDPGAgent(hidden_size=128, output_size=1)
scores = ddpg_agent.train_agent()

plt.plot(scores)
plt.show()

ddpg_agent.test_agent()
print(scores)
                



