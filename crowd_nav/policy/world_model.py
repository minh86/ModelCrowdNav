import gc
import math
from attrdict import AttrDict

import torch
from torch import nn
from crowd_nav.policy.cadrl import mlp

# SGAN import
from sgan.data.loader import data_loader
from sgan.utils import *
from sgan.data.trajectories import read_file


def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


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
        state = in_state.view((in_state.shape[0], -1, self.input_dim))
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
        mul_weighted_feature = torch.cat([weighted_feature] * size[1], dim=1)
        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([state, mul_weighted_feature], dim=2)
        actions = self.mlp3(joint_state.view((-1, self.mlp3_input_dim))).view((size[0], -1))
        # actions = self.output_func(actions)
        return actions


class SGANWorld(nn.Module):
    def __init__(self, modelPath, dataFile, device, obs_len=8, pred_len=1, skip=1, delim='tab', time_step=0.4):
        super().__init__()
        self.model = None
        self.modelPath = modelPath
        self.dataFile = dataFile
        self.device = device
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.delim = delim
        self.frameid = obs_len + 10
        self.generator = None
        self.time_step = time_step

    def data_loader(self):
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        data = read_file(self.dataFile, self.delim)
        frames = np.unique(data[:, 0]).tolist()
        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])
        curr_seq_data = np.concatenate(frame_data, axis=0)
        peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

        # padding humans to make the same numbers at every obs
        for ped_id in peds_in_curr_seq:
            curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
            curr_ped_seq = np.around(curr_ped_seq, decimals=4)
            start_f = curr_ped_seq[0, 0]
            end_f = curr_ped_seq[-1, 0]
            if start_f == frames[0] and end_f == frames[-1]:
                continue
            start_data = curr_ped_seq[0]
            end_data = curr_ped_seq[-1]
            for i in [f for f in frames if f < start_f]:
                data = np.vstack((data, [i, start_data[1], start_data[2], start_data[3]]))
            for i in [f for f in frames if f > end_f]:
                data = np.vstack((data, [i, end_data[1], end_data[2], end_data[3]]))
        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])
        curr_seq_data = np.concatenate(frame_data, axis=0)

        curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.obs_len))
        curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.obs_len))
        curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.obs_len))
        num_peds_considered = 0
        _non_linear_ped = []
        for _, ped_id in enumerate(peds_in_curr_seq):
            curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
            curr_ped_seq = np.around(curr_ped_seq, decimals=4)
            pad_front = frames.index(curr_ped_seq[0, 0])
            pad_end = frames.index(curr_ped_seq[-1, 0]) + 1
            if pad_end - pad_front != self.obs_len:
                continue
            curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
            curr_ped_seq = curr_ped_seq
            # Make coordinates relative
            rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
            rel_curr_ped_seq[:, 1:] = \
                curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
            _idx = num_peds_considered
            curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
            curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
            num_peds_considered += 1

        non_linear_ped += _non_linear_ped
        num_peds_in_seq.append(num_peds_considered)
        loss_mask_list.append(curr_loss_mask[:num_peds_considered])
        seq_list.append(curr_seq[:num_peds_considered])
        seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)

        # Convert numpy -> Torch Tensor
        obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float).to(self.device)
        obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float).to(self.device)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        seq_start_end = [(start, end)
                         for start, end in zip(cum_start_idx, cum_start_idx[1:])]

        # Data format: batch, input_size, seq_len
        # LSTM input format: seq_len, batch, input_size
        obs_traj = obs_traj.permute(2, 0, 1)
        obs_traj_rel = obs_traj_rel.permute(2, 0, 1)
        seq_start_end = torch.LongTensor(seq_start_end).to(self.device)

        return obs_traj, obs_traj_rel, seq_start_end

    def forward(self, in_state):
        gc.collect()
        self.frameid += 1
        # convert to file trajnet
        with open(self.dataFile, 'a') as cache_file_h:
            for i, cacheFrame in enumerate(in_state):
                cache_file_h.write("%s\t%s\t%s\t%s\n" % (self.frameid, i, cacheFrame[0], cacheFrame[1]))

        # remove first frame
        data = read_file(self.dataFile, self.delim)
        first_frame = data[0][0]
        data = [d for d in data if d[0] > first_frame]
        with open(self.dataFile, 'w') as cache_file_h:
            for cacheFrame in data:
                cache_file_h.write("%s\t%s\t%s\t%s\n" % (cacheFrame[0], cacheFrame[1], cacheFrame[2], cacheFrame[3]))

        # get output action
        # generator = torch.load(self.modelPath, map_location=self.device)
        num_samples = 1  # number of trajectories
        self.generator.decoder.seq_len=1
        with torch.no_grad():
            # convert to sgan input
            obs_traj, obs_traj_rel, seq_start_end = self.data_loader()
            for _ in range(num_samples):
                pred_traj_fake_rel = self.generator(obs_traj, obs_traj_rel, seq_start_end)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        # convert to action and return
        #pred_traj_fake: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
        pred_traj = pred_traj_fake.permute(1, 0, 2).reshape(-1, 2).tolist()
        obs_trj = obs_traj.permute(1, 0, 2).tolist()

        last_pos = [h[-1] for h in obs_trj]
        actions = np.subtract(pred_traj, last_pos)
        actions = actions/self.time_step
        return actions
