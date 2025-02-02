# 🤖 Behavior cloning framework based on Demonstration-Aware Machine Learning (DAML) System

**Author:** Alejandro Alemany

## 🚀 Overview
The **Demonstration-Aware Machine Learning (DAML) Framework** is designed for **human motion capture, processing, and robot trajectory learning**. The system processes **multi-view pose data**, applies **machine learning techniques**, and integrates with **robot simulation environments** for training robotic control models.

This project includes:
- **Pose extraction & filtering** using **MediaPipe**
- **Multi-view pose fusion** for robust motion tracking
- **Deep learning-based trajectory learning** using **PyTorch**
- **Robot control simulation** in **MuJoCo**
- **Scalable training pipeline** with modular components

---


---

## 📦 Installation

### 🔹 Prerequisites
Ensure you have the following installed:
- Python 3.9
- PyTorch
- NumPy, OpenCV, Matplotlib
- MuJoCo (for simulation)
- MediaPipe (for pose extraction)

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔹 Setting Up MuJoCo
Download MuJoCo from mujoco.org
Extract and install dependencies (mujoco-py for Python integration)
Set up MUJOCO_PATH in your environment

## 🛠 Usage
### 🔹 1. Process Demonstration Data
Extracts poses from videos and saves processed data:

```
python scripts/process_demonstrations.py --input demonstrations/ --output processed_demonstrations/
```
### 🔹 2. Train the Model
Trains the DAML model on processed pose data:
```
python scripts/train_model.py
```

### 🔹 3. Test Robot Control
Runs the trained model inside a MuJoCo simulation:
```
python scripts/test_robot.py
```

## 📖 Documentation
Each module has a dedicated documentation file:

- 📌 [DAML Framework](docs/daml.md)
- 🎯 [Pose Processing](docs/pose_processing.md)
- 🎨 [Visualization](docs/visualization.md)
- 🏋️ [Training Pipeline](docs/training.md)
- 🛠 [Testing & Simulation](docs/testing.md)
- 🔧 [Utilities](docs/utils.md)


🔗 References
MuJoCo: https://mujoco.org/
PyTorch: https://pytorch.org/
MediaPipe: https://developers.google.com/mediapipe

---