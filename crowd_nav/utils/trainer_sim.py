import logging
import random

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import sys

sys.path.append('../')
from crowd_nav.utils.pytorchtools import EarlyStopping
import torch


class Trainer_Sim(object):
    def __init__(self, model, memory, device, batch_size, path):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.val_data_loader = None
        self.batch_size = batch_size
        self.optimizer = None
        self.train_size = 0.8
        self.patience = 7
        self.path = path
        self.early_stopping = EarlyStopping(patience=self.patience, path=self.path)

    def set_learning_rate(self, learning_rate):
        logging.info('Current learning rate: %f', learning_rate)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def optimize_epoch(self, num_epochs):
        train_losses = []
        valid_losses = []
        train_num = int(len(self.memory) * self.train_size)
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        # if self.data_loader is None:
        random.shuffle(self.memory.memory)
        self.data_loader = DataLoader(self.memory[:train_num], self.batch_size, shuffle=True)
        self.val_data_loader = DataLoader(self.memory[train_num:], self.batch_size, shuffle=True)
        self.early_stopping.counter = 0
        self.early_stopping.early_stop = False
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            for data in self.data_loader:
                # State , Action , Next State, Reward
                cur_states, _, next_states, _ = data
                next_states = next_states[:, :, 2:]
                cur_states = cur_states.reshape(cur_states.size(0), -1)
                cur_states = Variable(cur_states).to(self.device)
                next_states = Variable(next_states).to(self.device)
                # ===================forward=====================
                output = self.model(cur_states)
                loss = self.criterion(output, next_states.reshape(next_states.size(0), -1))
                train_losses.append(loss.item())
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Evaluation
            self.model.eval()
            for data in self.val_data_loader:
                cur_states, _, next_states, _ = data
                next_states = next_states[:, :, 2:]
                cur_states = cur_states.reshape(cur_states.size(0), -1)
                cur_states = Variable(cur_states).to(self.device)
                next_states = Variable(next_states).to(self.device)
                # ===================forward=====================
                output = self.model(cur_states)
                loss = self.criterion(output, next_states.reshape(next_states.size(0), -1))
                valid_losses.append(loss.item())

            # Statistic loss
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            train_losses = []
            valid_losses = []
            self.early_stopping(valid_loss, self.model)

            if self.early_stopping.early_stop:
                # logging.info('Early stopping at epoch: %d', epoch)
                break

        self.model.load_state_dict(torch.load(self.path))  # load best model
        self.model.mse = 0-self.early_stopping.best_score
        return - self.early_stopping.best_score
