import numpy as np


class DefaultConfig:
    data = np.load("./data/dataset_plusOne.npz", allow_pickle=True)
    batch_size = 1024
    learning_rate = 0.01

    features_size = 221
    hidden_size = 100
    labels_size = 221

    layers_count = 10
    epoch_count = 1000
    require_improvement = 1000

    save_name = (
            "LSTM_"
            + str(layers_count)
            + "_"
            + str(epoch_count)
            + "_"
            + str(require_improvement)
    )
    load_path = "./save_model/baseline"
    save_path = "./save_model" + "/" + save_name
