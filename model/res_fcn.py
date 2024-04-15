import torch
import torch.nn as nn
import torch.nn.functional as F


class ResFCN(nn.Module):
    def __init__(self, input_dim):
        super(ResFCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size=3, stride=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        residual = F.relu(self.conv3(x)) + x
        x = F.relu(self.fc1(residual.view(-1, 32 * 4 * 4)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
