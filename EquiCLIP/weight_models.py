import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class WeightNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.model_name == "RN50":
            self.fc1 = nn.Linear(1024, 100)
        else:
            self.fc1 = nn.Linear(512, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        self.dp1 = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # check the impact of the gaussian noise even on CNNs
        # x = x + torch.randn(size=x.shape, device=x.device) # add Gaussian to avoid overfitting
        x = self.dp1(self.relu(self.fc1(x)))
        x = self.dp1(self.fc2(x))
        x = self.fc3(x)
        # x = torch.abs(x) + 0.5
        # x = torch.exp(-0.0 * x)
        # x = torch.exp(-0.1 * self.k * x)
        return x  # dim [B, 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument('--model_name', default="RN50", type=str)
    args = parser.parse_args()
    x = torch.randn(size=(4, 3, 224, 224))

    net = WeightNet(args)

    out = net(x)
    print(f"out.shape: {out.shape}")