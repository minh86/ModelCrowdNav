from torch.utils.data import Dataset
import random

class ReplayMemory(Dataset):
    def __init__(self, capacity, init_value=None):
        self.capacity = capacity
        self.memory = list()
        self.position = 0
        if init_value is not None:
            for _ in range(capacity):
                self.push(init_value)

    def push(self, item):
        # replace old experience with new experience
        if len(self.memory) < self.position + 1:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def is_full(self):
        return len(self.memory) == self.capacity

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = list()

    def shuffle(self):
        random.shuffle(self.memory)
