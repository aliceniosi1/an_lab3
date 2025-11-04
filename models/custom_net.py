# models/custom_net.py
import torch.nn as nn

class CustomNet(nn.Module):
    def __init__(self, num_classes=200):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 112 * 112, num_classes)

    def forward(self, x):
        x = self.pool(self.conv1(x).relu())
        x = self.pool(self.conv2(x).relu())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
