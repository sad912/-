import torch
import torch.nn as nn
from torch.autograd import Variable
from config import DefaultConfig as Config


class LSTM(nn.Module):
    def __init__(self, features_size, hidden_size, layers_count, labels_size, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=features_size,
            hidden_size=hidden_size,
            num_layers=layers_count,
            batch_first=batch_first,
            bidirectional=True
        )
        self.label = nn.Linear(hidden_size, labels_size)

    def forward(self, data):
        data_feature = data["feature"]
        h_0 = Variable(torch.zeros(1, Config.batch_size, self.hidden_size, device='mps'))
        c_0 = Variable(torch.zeros(1, Config.batch_size, self.hidden_size, device='mps'))
        output, (final_hidden_state, final_cell_state) = self.lstm(data_feature, (h_0, c_0))
        return self.label(final_hidden_state[-1])
