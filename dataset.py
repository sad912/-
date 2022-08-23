import numpy as np
from torch.utils.data import Dataset
from config import DefaultConfig as Config
import torch


class DataLoader(Dataset):
    def __init__(self):
        self.data = np.load(Config.data_directory, allow_pickle=True)
        self.train = {
            'features': torch.tensor(self.data['train_x']),
            'labels': torch.tensor(self.data['train_y'])
        }
        self.test = {
            'features': torch.tensor(self.data['test_x'])
        }

    def __getitem__(self, index, mode='train'):
        return {
            'feature': self[mode]['features'][index],
            'label': self[mode]['labels'][index],
        }

    def __len__(self, mode='train'):
        return len(self[mode]['features'])
