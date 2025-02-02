# 📦 Pose Processing

## Overview
This module extracts, filters, and processes **human pose data** from videos using **MediaPipe**.

## Key Components
- **`pose_data.py`** - Defines the `PoseData` structure.
- **`filters.py`** - Implements pose correction techniques (outlier removal, normalization, interpolation).
- **`pose_extractor.py`** - Extracts keypoints from video.
- **`view_combiner.py`** - Fuses multi-view poses.

## Usage
```bash
python scripts/process_demonstrations.py --input demonstrations/ --output processed_demonstrations/
