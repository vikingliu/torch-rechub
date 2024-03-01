import torch
from torch import nn
import torch.nn.functional as F

from ...basic.layers import GraphAttentionLayer


class GAT(nn.Module):

    def __init__(self,
                 in_features,
                 n_hidden,
                 n_heads,
                 num_classes,
                 concat=False,
                 dropout=0.4,
                 leaky_relu_slope=0.2):
        super(GAT, self).__init__()

        # Define the Graph Attention layers
        self.gat1 = GraphAttentionLayer(
            in_features=in_features, out_features=n_hidden, n_heads=n_heads,
            concat=concat, dropout=dropout, leaky_relu_slope=leaky_relu_slope
        )

        self.gat2 = GraphAttentionLayer(
            in_features=n_hidden, out_features=num_classes, n_heads=1,
            concat=False, dropout=dropout, leaky_relu_slope=leaky_relu_slope
        )

    def forward(self, input_tensor: torch.Tensor, adj_mat: torch.Tensor):
        # Apply the first Graph Attention layer
        x = self.gat1(input_tensor, adj_mat)
        x = F.elu(x)  # Apply ELU activation function to the output of the first layer

        # Apply the second Graph Attention layer
        x = self.gat2(x, adj_mat)

        return F.softmax(x, dim=1)  # Apply softmax activation function
