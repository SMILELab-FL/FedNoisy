import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

"""This code is from https://github.com/HanxunH/Active-Passive-Losses/blob/master/models.py"""


class ConvBrunch(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(ConvBrunch, self).__init__()
        padding = (kernel_size - 1) // 2
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.out_conv(x)


class ToyModel(nn.Module):
    def __init__(self, type='CIFAR10'):
        super(ToyModel, self).__init__()
        self.type = type
        if type == 'CIFAR10':
            self.block1 = nn.Sequential(
                ConvBrunch(3, 64, 3),
                ConvBrunch(64, 64, 3),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.block2 = nn.Sequential(
                ConvBrunch(64, 128, 3),
                ConvBrunch(128, 128, 3),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.block3 = nn.Sequential(
                ConvBrunch(128, 196, 3),
                ConvBrunch(196, 196, 3),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Sequential(
                nn.Linear(4 * 4 * 196, 256), nn.BatchNorm1d(256), nn.ReLU()
            )
            self.fc2 = nn.Linear(256, 10)
            self.fc_size = 4 * 4 * 196
        elif type == 'MNIST':
            self.block1 = nn.Sequential(
                ConvBrunch(1, 32, 3), nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.block2 = nn.Sequential(
                ConvBrunch(32, 64, 3), nn.MaxPool2d(kernel_size=2, stride=2)
            )
            # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Sequential(
                nn.Linear(64 * 7 * 7, 128), nn.BatchNorm1d(128), nn.ReLU()
            )
            self.fc2 = nn.Linear(128, 10)
            self.fc_size = 64 * 7 * 7
        self._reset_prams()

    def _reset_prams(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        return

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x) if self.type == 'CIFAR10' else x
        # x = self.global_avg_pool(x)
        # x = x.view(x.shape[0], -1)
        x = x.view(-1, self.fc_size)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
