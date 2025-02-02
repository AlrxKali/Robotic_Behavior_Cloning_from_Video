import numpy as np
from typing import Dict, List
from .pose_data import PoseData

class ViewCombiner:
    """Combines pose data from multiple views into a single pose representation"""
    def __init__(self, view_weights: Dict[str, float] = None):
        self.previous_poses = {}
        self.smoothing_window = 5
        self.view_weights = view_weights or {
            'front': 1.0,
            'right': 0.8,
            'left': 0.8,
            'upper': 0.7,
            'diagonal': 0.9
        }

    def compute_view_confidence(self, pose_data: PoseData, angle: str) -> float:
        """Computes a confidence score for a given camera view"""
        visibility_score = np.mean(pose_data.keypoints[:, 2])
        
        if angle in self.previous_poses:
            prev_positions = self.previous_poses[angle][-self.smoothing_window:]
            current_pos = pose_data.keypoints[:, :2]

            if prev_positions:
                movements = np.array([np.linalg.norm(current_pos - prev) for prev in prev_positions])
                tracking_stability = 1.0 / (1.0 + np.std(movements))
            else:
                tracking_stability = 1.0
        else:
            tracking_stability = 1.0
            self.previous_poses[angle] = []
        
        self.previous_poses[angle].append(pose_data.keypoints[:, :2])
        if len(self.previous_poses[angle]) > self.smoothing_window:
            self.previous_poses[angle].pop(0)
        
        relative_position = self.view_weights.get(angle, 0.7)
        return visibility_score * tracking_stability * relative_position

    def combine_poses(self, multi_view_poses: Dict[str, List[PoseData]], frame_idx: int) -> np.ndarray:
        """Combines multiple view poses into a single estimate"""
        frame_poses = {}
        confidences = {}

        for angle, poses in multi_view_poses.items():
            if frame_idx < len(poses):
                pose = poses[frame_idx]
                frame_poses[angle] = pose
                confidences[angle] = self.compute_view_confidence(pose, angle)

        if not frame_poses:
            return None

        total_weight = sum(confidences.values())
        weights = {k: v / total_weight for k, v in confidences.items()} if total_weight > 0 else {}

        combined_keypoints = np.zeros_like(next(iter(frame_poses.values())).keypoints)

        for angle, pose in frame_poses.items():
            combined_keypoints += pose.keypoints * weights.get(angle, 0)

        return combined_keypoints
