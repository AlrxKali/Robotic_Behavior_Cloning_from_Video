# 📦 Training Pipeline

## Overview
This module converts processed pose data into **robot joint movements** and trains the DAML model.

## Key Components
- **`train_pipeline.py`** - Handles demonstration data processing, inverse kinematics, and model training.

## Training Workflow
1. **Extract pose data** using `pose_extractor.py`.
2. **Convert poses to joint angles** using `train_pipeline.py`.
3. **Train the model** using `train_pipeline.py`.

## Usage
```bash
python scripts/train_model.py

