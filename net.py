from typing import List


import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid


class FlexGCN(nn.Module):
    
    def __init__(self, input:int, inplanes:List[int], num_layers:int) -> None:
        """
        inplanes 代表每层的输出
        """
        super(FlexGCN, self).__init__()
        assert len(inplanes) == num_layers
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList() 
        for i in range(num_layers):
            output = inplanes[i]
            self.gcn_layers.append(GCNConv(input, output))
            input = output
        
        
    def forward(self, x:Tensor, edge):
        for i in range(self.num_layers):
            x_raw = x.clone()
            x = self.gcn_layers[i](x, edge)
            x = F.relu(x)
            x = F.layer_norm(x)
            x = F.dropout(x)
            x = x + x_raw
        # ? FNN    
        return x
    