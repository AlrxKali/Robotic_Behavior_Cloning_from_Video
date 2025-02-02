import torch
from dataclasses import dataclass

@dataclass
class DAMLConfig:
    """Configuration settings for the DAML model"""
    input_dim: int = 8  # Input feature dimensions (keypoints)
    hidden_dim: int = 128  # Hidden layer size
    output_dim: int = 8  # Output dimensions (robot joint angles)
    num_layers: int = 3  # Number of network layers
    learning_rate: float = 0.001  # Learning rate for optimization
    batch_size: int = 32  # Batch size for training
    num_epochs: int = 100  # Number of training epochs
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Training device
