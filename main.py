from utils.download_dataset import download_and_extract_dataset
from dataset.data_preparation import data_preparation
from models.custom_net import CustomNet
from train import train
from eval import validate
import torch
from torch import nn
# 1. Download dataset
if __name__ == "__main__":
    download_and_extract_dataset()

    # 2. Preparazione dataloader
    train_loader, val_loader = data_preparation(batch_size=64)

    # 3. Modello + loss + optimizer
    model = CustomNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 4. Training
    best_acc = 0
    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer)
        val_accuracy = validate(model, val_loader, criterion)
        best_acc = max(best_acc, val_accuracy)

    print(f'Best validation accuracy: {best_acc:.2f}%')
