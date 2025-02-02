import torch
import torch.nn as nn
from .config import DAMLConfig

class DAMLEncoder(nn.Module):
    """Encoder network for DAML"""
    def __init__(self, config: DAMLConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(config.input_dim, config.hidden_dim),
            *[nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(config.num_layers - 2)],
            nn.Linear(config.hidden_dim, config.hidden_dim)
        ])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.dropout(self.activation(layer(x)))
        return self.layers[-1](x)
