import torch.nn as nn 

from mmdet.models.builder import HEADS

@HEADS.register_module()
class V2lTranformHead(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=1024, out_dim=768, n_layers=1):
        self.n_layers = n_layers
        
        self.linears = nn.ModuleList()
        if n_layers == 1:
            self.linears.append(nn.Linear(in_dim, out_dim))
        else:
            for i in range(n_layers):
                if i == 0:
                    self.linears.append(nn.Linear(in_dim, hidden_dim))
                elif i > 0 and i < n_layers - 1:
                    self.linears.append(nn.Linear(hidden_dim, hidden_dim))
                else:
                    self.linears.append(nn.Linear(hidden_dim, out_dim))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.linears[0](x)
        return out