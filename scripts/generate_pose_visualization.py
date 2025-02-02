import argparse
import numpy as np
from src.visualization.pose_visualizer import PoseVisualizer

def visualize_pose(pose_file: str, output_path: str):
    """Generates a 3D pose visualization from a processed pose file."""
    pose_data = np.load(pose_file, allow_pickle=True)
    visualizer = PoseVisualizer()
    visualizer.visualize_3d_pose(pose_data[0], output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D pose visualization.")
    parser.add_argument("--pose", type=str, required=True, help="Path to processed pose file (.npy).")
    parser.add_argument("--output", type=str, required=True, help="Path to save visualization.")

    args = parser.parse_args()
    visualize_pose(args.pose, args.output)
