import numpy as np
from .pose_data import PoseData

class PoseProcessor:
    """Abstract base class for pose processing steps"""
    def process(self, pose_data: PoseData) -> PoseData:
        raise NotImplementedError

class OutlierRemoval(PoseProcessor):
    """Removes keypoints with low confidence or sudden changes"""
    def __init__(self, confidence_threshold: float = 0.5, position_threshold: float = 50.0):
        self.confidence_threshold = confidence_threshold
        self.position_threshold = position_threshold
        self.previous_keypoints = None

    def process(self, pose_data: PoseData) -> PoseData:
        mask = pose_data.keypoints[:, 2] >= self.confidence_threshold
        
        if self.previous_keypoints is not None:
            diff = np.linalg.norm(pose_data.keypoints[:, :2] - self.previous_keypoints, axis=1)
            mask &= diff < self.position_threshold
        
        self.previous_keypoints = pose_data.keypoints[:, :2].copy()
        pose_data.keypoints[~mask] = np.nan
        return pose_data

class CoordinateNormalizer(PoseProcessor):
    """Normalizes pose coordinates based on shoulder position"""
    def process(self, pose_data: PoseData) -> PoseData:
        shoulder_coords = pose_data.keypoints[11:13].mean(axis=0)[:2]  # Avg of left & right shoulders
        pose_data.keypoints[:, :2] -= shoulder_coords
        pose_data.metadata['normalization_reference'] = shoulder_coords.tolist()
        return pose_data

class InterpolationHandler(PoseProcessor):
    """Handles missing keypoints by interpolating from previous frames"""
    def __init__(self, max_frames: int = 5):
        self.max_frames = max_frames
        self.buffer = []

    def process(self, pose_data: PoseData) -> PoseData:
        self.buffer.append(pose_data)
        if len(self.buffer) > self.max_frames:
            self.buffer.pop(0)

        if np.any(np.isnan(pose_data.keypoints)):
            pose_data = self._interpolate_missing(pose_data)
        return pose_data

    def _interpolate_missing(self, pose_data: PoseData) -> PoseData:
        if len(self.buffer) < 2:
            return pose_data

        for i in range(pose_data.keypoints.shape[0]):
            if np.isnan(pose_data.keypoints[i, 0]):
                valid_poses = [b.keypoints[i] for b in self.buffer if not np.isnan(b.keypoints[i, 0])]
                if valid_poses:
                    pose_data.keypoints[i] = valid_poses[-1]
        return pose_data
