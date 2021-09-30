import torch
from torch import nn
from torch.utils.data import DataLoader

class SarcNN(nn.Module):
    def __init__(self):
        super(SarcNN, self).__init__()
        # nn w/ 3 hidden layers
        self.linear_relu_stack = nn.Sequential(
            # TODO: how to make this structure better?
            nn.Linear(100, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 2),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out

model = SarcNN()
print(model)