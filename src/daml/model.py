import torch
import torch.nn as nn
from .config import DAMLConfig
from .encoder import DAMLEncoder
from .decoder import DAMLDecoder

class DAMLModel(nn.Module):
    """Complete DAML model combining encoder and decoder"""
    def __init__(self, config: DAMLConfig):
        super().__init__()
        self.encoder = DAMLEncoder(config)
        self.decoder = DAMLDecoder(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        encoded = [self.encoder(x[:, t]) for t in range(seq_len)]
        encoded = torch.stack(encoded, dim=1)
        
        decoded = [self.decoder(encoded[:, t]) for t in range(seq_len)]
        return torch.stack(decoded, dim=1)
