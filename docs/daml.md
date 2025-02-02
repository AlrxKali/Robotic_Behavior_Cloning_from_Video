# 📦 DAML Framework

## Overview
The **Demonstration-Aware Machine Learning (DAML) Framework** is a deep learning model designed to learn robotic trajectories from human motion data. It consists of an **encoder-decoder architecture** implemented in **PyTorch**.

## Key Components
- **`config.py`** - Defines model hyperparameters.
- **`dataset.py`** - Loads demonstration trajectory data.
- **`encoder.py`** - Encodes input poses.
- **`decoder.py`** - Decodes learned representations into robot joint movements.
- **`model.py`** - Combines encoder and decoder into a full model.
- **`trainer.py`** - Trains the DAML model.

## Usage
```bash
python scripts/train_model.py


**For more details, refer to the main README.md.**

