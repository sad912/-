from dataset import FlowDataset
import torch
import numpy as np


def infer(config, model, infer_data_loader, mean, standard):
    model.load_state_dict(torch.load(config.load_path))
    model.eval()
    with torch.no_grad():  # 关闭梯度
        labels = np.zeros([config.batch_size, 12, 221])
        for data in infer_data_loader:
            infer_labels = model(data)
            infer_labels = FlowDataset.recover_tensor_data(infer_labels, mean, standard)
            labels = np.concatenate([labels, infer_labels])
        labels = labels[config.batch_size:]
    np.savez("pre_y_baseline.npz", test_y=labels)
    print('123')
