import torch
import torch.nn as nn
from typing import List

class TabularModel(nn.Module):
    """Base model for tabular data"""
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 dropout_rate: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        return self.model(x)