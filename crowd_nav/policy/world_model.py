import math
import random

import torch
from torch import nn

class autoencoder(nn.Module):
    def __init__(self, num_human, drop_rate=0.00):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_human * 4, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12))
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, num_human * 2))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class mlp(nn.Module):
    def __init__(self, num_human, drop_rate=0.5):
        super(mlp, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_human * 4, 128),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Linear(64, 12),
            nn.ReLU(True), nn.Linear(12, num_human * 2),
            nn.Tanh()
        )
        self.mse = 0
        self.device = None

    def forward(self, x):
        x = self.mlp(x)
        return x

    def noise_pre(self, x):
        x = self.forward(x)
        mean = math.sqrt(self.mse)
        bias = (torch.randn(x.shape) * mean).to(self.device)

        return x+bias

    def init_weight(self):
        torch.nn.init.xavier_uniform(self.mlp.weight)


