"""for mlp"""
import numpy as np
import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self, num_classes=10, data_shape=(3, 32, 32)):
        super().__init__()
        self.input_shape = np.prod(data_shape)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(self.input_shape, num_classes)

    def forward(self, x):
        return self.fc(self.flat(x))


class SMLP(nn.Module):
    def __init__(self, width=1024, num_classes=10, data_shape=(3, 32, 32)):
        super().__init__()
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(np.prod(data_shape), width)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x


class DMLP(nn.Module):
    def __init__(self, width=1024, num_classes=10, data_shape=(3, 32, 32)):
        super().__init__()
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(np.prod(data_shape), width)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, width)
        self.l4 = nn.Linear(width, width)
        self.l5 = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.l5(x)
        return x
