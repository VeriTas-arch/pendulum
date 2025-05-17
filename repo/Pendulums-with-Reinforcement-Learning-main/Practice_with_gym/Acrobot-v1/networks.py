import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, action_dim, action_bound):
        super().__init__()

        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = [1e-6, 1.]

        self.l1 = nn.Linear(11, 256) # state_dim = 11
        self.l2 = nn.Linear(256, 256)

        self.mu = nn.Linear(256, self.action_dim)
        self.std = nn.Linear(256, self.action_dim)

    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        mu = self.mu(x)
        std = self.std(x)
        std = torch.clamp(std, min=self.std_bound[0], max=self.std_bound[1])

        return mu, std

    def sample_normal(self, mu, std):
        normal_dist = D.Normal(mu, std)
        action = normal_dist.rsample()
        action = torch.tanh(action) * torch.tensor(self.action_bound)
        log_pdf = normal_dist.log_prob(action)
        log_pdf -= torch.log(1 - action.pow(2) + self.std_bound[0])
        log_pdf = torch.sum(log_pdf, dim=-1, keepdim=True)

        return action, log_pdf

class Critic(nn.Module):
    
    def __init__(self, action_dim, state_dim):
        super().__init__()

        self.action_dim = action_dim
        self.state_dim = 11

        self.l1 = nn.Linear(self.state_dim + self.action_dim, 256)
        self.l2 = nn.Linear(256, 256) 
        self.q =  nn.Linear(256, 1)

    def forward(self, state_action):
        state, action = state_action[0].to(torch.float32), state_action[1].to(torch.float32)

        x = torch.cat([state, action], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.q(x)

        return x
