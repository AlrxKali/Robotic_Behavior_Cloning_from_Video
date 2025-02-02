# 📦 Visualization

## Overview
This module provides tools for **2D & 3D pose visualization** and **pose animation**.

## Key Components
- **`pose_visualizer.py`** - Renders pose keypoints on images and generates animations.

## Usage
```python
from visualization.pose_visualizer import PoseVisualizer
visualizer = PoseVisualizer()
visualizer.visualize_3d_pose(pose_data, "output.png")