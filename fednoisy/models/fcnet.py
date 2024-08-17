import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


class TwoLayerLinearNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TwoLayerLinearNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x.view(-1, self.input_size))
        out = self.relu(out)
        out = self.fc2(out)
        return out


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out.reshape(-1)
