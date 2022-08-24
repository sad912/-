from torch.utils.data import Dataset
import torch
from config import DefaultConfig as Config


class FlowDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        self.mean = None
        self.standard = None
        # print(Config.data.keys()) # ['train_x', 'train_y', 'test_x']
        if mode == 'train':
            self.features = self.pre_process_data(Config.data['train_x'])
            self.labels = self.pre_process_data(Config.data['train_y'], True)
        elif mode == 'infer':
            self.features = self.pre_process_data(Config.data['test_x'])

    def __getitem__(self, index):
        assert index < len(self.features)
        return {
            'features': self.features[index],
            'labels': self.labels[index] if self.mode == 'train' else None
        }

    def pre_process_data(self, data, is_self_mean_and_std=False):
        tensor_data = FlowDataset.to_tensor(data)
        mean_tensor = tensor_data.mean()
        standard_tensor = tensor_data.std()
        if is_self_mean_and_std:
            self.mean = mean_tensor.item()
            self.standard = standard_tensor.item()
        discrete_normalize_tensor_data = tensor_data.normal_(mean=mean_tensor, std=standard_tensor)
        return discrete_normalize_tensor_data

    @staticmethod
    def to_tensor(data):
        device = torch.device('mps')  # 直接使用 mps，未做 mps 设备和 torch 版本的验证
        return torch.FloatTensor(data).to(device)

    @staticmethod
    def split_data(data, array):
        return torch.utils.data.random_split(data, array)

    @staticmethod
    def recover_tensor_data(data, mean, standard):
        return data * standard + mean

# 绘制 features, labels 点图，查看分布情况
# import matplotlib.pyplot as plt
# x = FlowDataset().features
# y = FlowDataset().labels
# plt.scatter(x.to('cpu').numpy(), y.to('cpu').numpy(), s=0.1)
# plt.show()
