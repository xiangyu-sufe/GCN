from typing import List


import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, last= False, dropout=0.5):
        super(GCNLayer, self).__init__()
        self.last = last
        self.gcn = GCNConv(in_channels, out_channels)
        self.relu = nn.ReLU()
        if not self.last:
            self.layernorm1 = nn.LayerNorm(out_channels)
            self.dropout = nn.Dropout(dropout)
        
            self.layernorm2 = nn.LayerNorm(out_channels)
    
    def forward(self, x, edge_index):
        # Apply GCN
        x_gcn = self.gcn(x, edge_index)
        
        # Apply ReLU
        if not self.last:
            # Apply LayerNorm
            x_gcn = self.relu(x_gcn)
            x_gcn = self.layernorm1(x_gcn)
            
            # Apply Dropout
            x_gcn = self.dropout(x_gcn)
        
            # Add residual connection
            x_gcn = x + x_gcn
            
            # Apply second LayerNorm
            x_gcn = self.layernorm2(x_gcn)
            
        return x_gcn


class FlexGCN(nn.Module):
    
    def __init__(self, input:int, num_layers:int) -> None:
        """
        inplanes 代表每层的输出
        """
        super(FlexGCN, self).__init__()
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers-1):
            self.gcn_layers.append(GCNLayer(input, input))
        self.gcn_layers.append(GCNLayer(input, 1, last=True))
        
    def forward(self, x:Tensor, edge):
        for layer in self.gcn_layers:
            x = layer(x, edge)
            
        return x
    
if __name__ == "__main__":

    input = torch.randn((3561,23))
    edge_index = torch.tensor([[],[]], dtype=torch.long)
    net = FlexGCN(input=23, num_layers=2)
    x=net(input, edge_index)
    print(x.abs().max())
    