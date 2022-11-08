import math
import random

import torch
from torch import nn
from crowd_nav.policy.cadrl import mlp


def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


class AutoencoderWorld(nn.Module):
    def __init__(self, num_human, drop_rate=0.00):
        super().__init__()
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


class MlpWorld(nn.Module):
    def __init__(self, num_human, drop_rate=0.5, multihuman=True):
        super().__init__()
        if not multihuman:
            num_human = 1
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

        return x + bias


class AttentionWorld(nn.Module):
    def __init__(self, input_dim=4, with_global_state=True):
        super().__init__()
        mlp1_dims = [150, 100]
        mlp2_dims = [100, 50]
        attention_dims = [100, 100, 1]
        mlp3_dims = [150, 100, 100, 2]
        self.input_dim = input_dim
        self.with_global_state = with_global_state
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.mlp3_input_dim = mlp2_dims[-1] + input_dim
        self.mlp3 = mlp(self.mlp3_input_dim, mlp3_dims)
        self.attention_weights = None
        self.output_func = nn.Tanh()

    def forward(self, in_state):
        state = in_state.view((in_state.shape[0],-1,self.input_dim))
        size = state.shape
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)
        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)). \
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        mul = torch.mul(weights, features)
        weighted_feature = torch.sum(mul, dim=1, keepdim=True)
        mul_weighted_feature = torch.cat([weighted_feature]*size[1], dim=1)
        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([state, mul_weighted_feature], dim=2)
        actions = self.mlp3(joint_state.view((-1, self.mlp3_input_dim))).view((size[0],-1))
        # actions = self.output_func(actions)
        return actions
