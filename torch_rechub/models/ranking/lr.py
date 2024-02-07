import torch
import torch.nn as nn


class LR(nn.Module):
    """Logistic Regression Module. It is the one Non-linear
    transformation for input feature.
    z = sigmoid(wx + b) = 1 / (1 + e^(wx+b))

    Args:
        input_dim (int): input size of Linear module.

    Shape:
        - Input: `(batch_size, input_dim)`
        - Output: `(batch_size, 1)`
    """

    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))
