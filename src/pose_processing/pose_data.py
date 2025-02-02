import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PoseData:
    """Data class to store pose information"""
    timestamp: float
    keypoints: np.ndarray  # Shape: [N_keypoints, 3] (x, y, confidence)
    frame_id: int
    metadata: Dict[str, Any]
