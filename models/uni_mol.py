from torch import nn
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch_geometric.nn import global_mean_pool


class uni_mol2_model(torch.nn.Module):

    # Too large output_features: Make it 2 or 4.
    def __init__(self, input_features):
        super(uni_mol2_model, self).__init__()
        self.lin = Linear(input_features, 64)

    def forward(self, data_x):

        x = self.lin(data_x)
        return x

