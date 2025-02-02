import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List
from pose_processing.pose_data import PoseData

class PoseVisualizer:
    """Class for visualizing human pose data in 2D and 3D"""
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.connections = self.mp_pose.POSE_CONNECTIONS

    def visualize_multi_view(self, frames: Dict[str, np.ndarray], poses: Dict[str, PoseData], output_path: str):
        """Visualizes poses from multiple camera views."""
        n_views = len(frames)
        fig, axes = plt.subplots(1, n_views, figsize=(6 * n_views, 6))
        if n_views == 1:
            axes = [axes]

        for ax, (angle, frame) in zip(axes, frames.items()):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(frame_rgb)

            if angle in poses:
                self._draw_pose_on_image(ax, poses[angle])

            ax.set_title(f"View: {angle}")
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def visualize_3d_pose(self, pose_3d: np.ndarray, output_path: str):
        """Visualizes a 3D pose."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], c='b', marker='o')

        for connection in self.connections:
            start_idx, end_idx = connection
            x = [pose_3d[start_idx, 0], pose_3d[end_idx, 0]]
            y = [pose_3d[start_idx, 1], pose_3d[end_idx, 1]]
            z = [pose_3d[start_idx, 2], pose_3d[end_idx, 2]]
            ax.plot(x, y, z, 'r-')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Pose Reconstruction')

        plt.savefig(output_path)
        plt.close()

    def create_pose_animation(self, poses_sequence: List[np.ndarray], output_path: str, fps: int = 30):
        """Creates an animated sequence of 3D poses."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            ax.clear()
            pose = poses_sequence[frame]

            ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c='b', marker='o')

            for connection in self.connections:
                start_idx, end_idx = connection
                x = [pose[start_idx, 0], pose[end_idx, 0]]
                y = [pose[start_idx, 1], pose[end_idx, 1]]
                z = [pose[start_idx, 2], pose[end_idx, 2]]
                ax.plot(x, y, z, 'r-')

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Frame {frame}')

        anim = animation.FuncAnimation(fig, update, frames=len(poses_sequence), interval=1000 / fps)

        if output_path.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
        else:
            anim.save(output_path, writer='ffmpeg', fps=fps)

        plt.close()

    def _draw_pose_on_image(self, ax, pose_data: PoseData):
        """Draws pose keypoints and connections on a 2D image."""
        keypoints = pose_data.keypoints
        visible_points = keypoints[:, 2] > 0.5

        ax.scatter(keypoints[visible_points, 0], keypoints[visible_points, 1], c='r', s=20)

        for connection in self.connections:
            start_idx, end_idx = connection

            if keypoints[start_idx, 2] > 0.5 and keypoints[end_idx, 2] > 0.5:
                x = [keypoints[start_idx, 0], keypoints[end_idx, 0]]
                y = [keypoints[start_idx, 1], keypoints[end_idx, 1]]
                ax.plot(x, y, 'g-', linewidth=2, alpha=0.7)
