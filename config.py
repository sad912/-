import numpy as np


class DefaultConfig:
    data = np.load("./data/dataset_plusOne.npz", allow_pickle=True)
    batch_size = 128
    lr = 0.01

    input_size = 221
    hidden_size = 50
    num_output = 221

    num_layers = 5
    max_epoch = 5
