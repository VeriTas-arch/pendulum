import torch
import torch.nn as nn
import torch.nn.functional as F

# Critic: Value Network
torch.autograd.set_detect_anomaly(True)
class Critic(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        out = self.l3(x)

        return out

# Actor: Policy Network    
class Actor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        
    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        out = 20 * self.tanh(self.l3(x))

        return out
    