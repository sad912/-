import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, features_size, hidden_size, layers_count, labels_size, batch_size, batch_first=True):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=features_size,
            hidden_size=hidden_size,
            num_layers=layers_count,
            batch_first=batch_first,
        )
        self.label = nn.Linear(hidden_size, labels_size)

    def forward(self, data):
        data_feature = data["feature"]
        data_label, (final_hidden_state, final_cell_state) = self.lstm(data_feature, None)
        return self.label(data_label)
