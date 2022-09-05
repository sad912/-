import torch
from dataset import FlowDataset
from config import DefaultConfig as config
from torch.utils.data import DataLoader
from model import LSTM
from train import train
import numpy as np


def infer(config, model, infer_data_loader, infer_data):
    model.load_state_dict(torch.load(config.load_path))
    model.eval()
    with torch.no_grad():  # 关闭梯度
        pre_flow = np.zeros([config.batch_size, 12, 221])
        for data_ in infer_data_loader:
            pre_value = model(data_)
            y_train_norm = infer_data.get_norm()
            pre_value = dataset.Data.recoverd_data(
                pre_value.cpu().detach().numpy(),
                y_train_norm[0].squeeze(1),  # max_data
                y_train_norm[1].squeeze(1),  # min_data
            )
            pre_flow = np.concatenate([pre_flow, pre_value])
        pre_flow = pre_flow[config.batch_size:]
    return pre_flow


print(config.device)
model = LSTM(
    config.features_size, config.hidden_size, config.layers_count, config.labels_size
).to(config.device)

infer_data = FlowDataset("infer")
infer_data_loader = DataLoader(
    infer_data, batch_size=config.batch_size, shuffle=False, drop_last=False
)

test_y = infer(config, model, infer_data_loader, infer_data)
np.savez("pre_y_baseline.npz", test_y=test_y)
print('result saved!')
