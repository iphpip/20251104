import torch.nn as nn
from typing import List

class MLPProjectionHead(nn.Module):
    """MLP投影头"""
    
    def __init__(self, input_dim: int = 128, hidden_dims: List[int] = [256, 256],
                 output_dim: int = 64):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)