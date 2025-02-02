import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List

class DemonstrationDataset(Dataset):
    """Dataset for demonstration trajectory sequences"""
    def __init__(self, demonstrations: List[np.ndarray], sequence_length: int = 30):
        self.demonstrations = demonstrations
        self.sequence_length = sequence_length
        self.sequences = self._prepare_sequences()
    
    def _prepare_sequences(self) -> List[torch.Tensor]:
        sequences = []
        for demo in self.demonstrations:
            for i in range(len(demo) - self.sequence_length + 1):
                sequence = demo[i:i + self.sequence_length]
                sequences.append(torch.FloatTensor(sequence))
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]
