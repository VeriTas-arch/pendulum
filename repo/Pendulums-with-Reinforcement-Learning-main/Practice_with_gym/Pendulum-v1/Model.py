### NOTE:Seems like normalizing in any ways ruin the training process, for some reason...
### F.normalize, batchnorm, Layernorm all worsens performance.
### 

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size//2)
        self.l2 = nn.Linear(hidden_size//2, hidden_size//2)
        self.l3 = nn.Linear(hidden_size//2, hidden_size//2)
        self.l4 = nn.Linear(hidden_size//2, hidden_size//4)
        self.l5 = nn.Linear(hidden_size//4, output_size)
        
        self.lrelu = nn.LeakyReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.lrelu(self.l1(x))
        x = self.lrelu(self.l2(x))
        x = self.lrelu(self.l3(x))
        x = self.lrelu(self.l4(x))
        out = self.l5(x)

        return out.to(device)

class Actor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size//2)
        self.l3 = nn.Linear(hidden_size//2, hidden_size//4)
        self.l4 = nn.Linear(hidden_size//4, output_size)
        self.tanh = nn.Tanh()
        self.lrelu = nn.LeakyReLU()
        
    def forward(self, state):
        x = self.lrelu(self.l1(state))
        x = self.lrelu(self.l2(x))
        x = self.lrelu(self.l3(x))
        out = 2 * self.tanh(self.l4(x))

        return out.to(device)