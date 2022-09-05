import numpy as np


class DefaultConfig:
    data = np.load("./data/dataset_plusOne.npz", allow_pickle=True)
    batch_size = 10
    learning_rate = 0.01

    features_size = 221
    hidden_size = 1
    labels_size = 221

    layers_count = 1
    epoch_count = 1
    require_improvement = 1000

    load_path = "./save_model/best_model.pt"
    save_path = "./save_model/"

    batch_size = 1024
    learning_rate = 0.01

    features_size = 221
    hidden_size = 100
    labels_size = 221

    layers_count = 10
    epoch_count = 1000
    require_improvement = 1000