import time
import torch
import torch.nn as nn
from utils import save_model, evaluate


def train(config, model, train_data_loader, eval_data_loader):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    batch_count = 0  # 批次数
    last_improve_batch_count = 0  # 上次 loss 下降的批次
    dev_best_loss = float("inf")
    best_model = model
    for epoch in range(config.epoch_count):
        print('------------{} epoch------------'.format(epoch))
        epoch_loss = 0.0
        for data in train_data_loader:
            print('------------{} batch------------'.format(batch_count))
            model.train()
            start_time = time.time()
            model.zero_grad()
            labels = model(data)
            loss = loss_function(labels, data['label'])
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if batch_count % 100 == 0:
                dev_loss = evaluate(model, eval_data_loader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    best_model = model
                    improve = "*"
                    last_improve_batch_count = batch_count
                else:
                    improve = ""

                end_time = time.time()
                msg = "Iter: {0:>6},  Train Loss: {1:>.6f}, Val Loss: {2:>.6f}, {3:>.2f}, {4}"
                print(
                    msg.format(
                        batch_count,
                        loss.item(),
                        dev_loss,
                        end_time - start_time,
                        improve,
                    )
                )
            batch_count += 1
            if batch_count - last_improve_batch_count > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                save_model(config, best_model)
                break
    save_model(config, best_model)


