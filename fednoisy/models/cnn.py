from torch import nn
import torch.nn.functional as F


class Cifar10Net(nn.Module):
    """Code from FedDyn"""

    def __init__(self):
        super(Cifar10Net, self).__init__()
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        conv1_out = self.pool(F.relu(self.conv1(x)))
        conv2_out = self.pool(F.relu(self.conv2(conv1_out)))
        conv2_out = conv2_out.view(-1, 64 * 5 * 5)
        fc1_out = F.relu(self.fc1(conv2_out))
        fc2_out = F.relu(self.fc2(fc1_out))
        out = self.fc3(fc2_out)
        return out


class ConvNet5(nn.Module):
    """LeNet in PyTorch from https://github.com/Princeton-SysML/GradAttack/blob/master/gradattack/models/LeNet.py

    Currently only for MNIST
    """

    def __init__(self, num_classes=10, num_channels=3):
        super(ConvNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 12, 5, padding=5 // 2, stride=2)
        self.conv2 = nn.Conv2d(12, 12, 5, padding=5 // 2, stride=2)
        self.conv3 = nn.Conv2d(12, 12, 5, padding=5 // 2, stride=1)
        self.conv4 = nn.Conv2d(12, 12, 5, padding=5 // 2, stride=1)
        self.sigmoid = nn.Sigmoid()
        if num_channels == 3:
            self.fc = nn.Linear(768, num_classes)
        elif num_channels == 1:
            self.fc = nn.Linear(588, num_classes)

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        feature = x.view(x.size(0), -1)
        out = self.fc(feature)
        return out


class SimpleCNNMNIST(nn.Module):
    """Code from https://github.com/Xtra-Computing/NIID-Bench"""

    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    """Code from https://github.com/Xtra-Computing/NIID-Bench"""

    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
