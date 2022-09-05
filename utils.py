import torch
import os
import shutil
import torch.nn as nn


def save_model(config, best_model):
    if not os.path.exists(config.save_path):
        os.mkdir(config.save_path)
    else:
        shutil.rmtree(config.save_path)
        assert not os.path.exists(config.save_path)
        os.mkdir(config.save_path)
    torch.save(best_model.state_dict(), config.save_path + '/best_model.pt')


def evaluate(model, data_iter):
    loss_total = 0
    loss_func = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for data in data_iter:
            labels = model(data)
            loss = loss_func(labels, data["label"])
            loss_total += loss
    return loss_total / len(data_iter)
