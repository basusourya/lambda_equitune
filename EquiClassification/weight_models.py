import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class WeightNet(nn.Module):
    def __init__(self, model_name='alexnet'):
        super().__init__()
        if model_name == 'alexnet':
            self.fc1 = nn.Linear(9216, 100)
        elif model_name == 'resnet':
            self.fc1 = nn.Linear(512, 100)
        elif model_name == 'vgg':
            self.fc1 = nn.Linear(25088, 100)
        elif model_name=='densenet':
            self.fc1 = nn.Linear(50176, 100)
        else:
            NotImplementedError("model name is not implemented for this algorithm")
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        self.dp1 = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x  #+ torch.randn(size=x.shape, device=x.device) # add Gaussian to avoid overfitting
        x = self.dp1(self.relu(self.fc1(x)))
        x = self.dp1(self.fc2(x))
        x = self.fc3(x)
        return x  # dim [B, 1]


if __name__ == "__main__":
    x = torch.randn(size=(4, 3, 224, 224))

    net = WeightNet()

    out = net(x)
    print(f"out.shape: {out.shape}")