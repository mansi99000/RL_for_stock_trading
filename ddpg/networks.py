import numpy as np
import torch
import torch.nn as nn


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, init_w=3e-3):
        super().__init__()

        self.layer_1 = nn.Linear(state_dim, 64)
        self.layer_2 = nn.Linear(64, 16)
        self.layer_3 = nn.Linear(16, action_dim)

        self.max_action = max_action

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.layer_1.weight.data = fanin_init(self.layer_1.weight.data.size())
        self.layer_2.weight.data = fanin_init(self.layer_2.weight.data.size())
        self.layer_3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))

        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super().__init__()

        self.layer_1 = nn.Linear(state_dim + action_dim, 64)
        self.layer_2 = nn.Linear(64, 16)
        self.layer_3 = nn.Linear(16, 1)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.layer_1.weight.data = fanin_init(self.layer_1.weight.data.size())
        self.layer_2.weight.data = fanin_init(self.layer_2.weight.data.size())
        self.layer_3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = torch.relu(self.layer_1(xu))
        x1 = torch.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)

        return x1
