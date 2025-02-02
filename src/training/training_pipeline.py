import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
import mujoco
from pose_processing.pose_data import PoseData
from pose_processing.pose_extractor import PoseExtractor
from pose_processing.view_combiner import ViewCombiner
from daml.config import DAMLConfig
from daml.model import DAMLModel

class RobotTrainingPipeline:
    """Pipeline for processing demonstration data and training the DAML model."""
    def __init__(self, model_save_dir: str = "models"):
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        self.pose_extractor = PoseExtractor()
        self.view_combiner = ViewCombiner()

        # Initialize MuJoCo model for inverse kinematics (IK)
        self.mj_model = mujoco.MjModel.from_xml_path("panda_mujoco/world.xml")
        self.mj_data = mujoco.MjData(self.mj_model)

    def process_demonstrations(self, demo_dir: str, output_dir: str) -> List[Dict]:
        """Processes demonstration videos, extracts pose data, and combines views."""
        demo_path = Path(demo_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        training_data = []

        for demo_folder in demo_path.glob("demo_*"):
            demo_output_dir = output_path / demo_folder.name
            demo_output_dir.mkdir(exist_ok=True)

            for variation_dir in demo_folder.glob("variation_*"):
                print(f"Processing {variation_dir.name}...")

                videos = list(variation_dir.glob("*.MOV"))
                if not videos:
                    print(f"Warning: No video files found in {variation_dir}")
                    continue

                pose_data = {}
                for video_path in videos:
                    angle = video_path.stem
                    print(f"Extracting poses from {angle} view...")
                    pose_data[angle] = self.pose_extractor.process_video(str(video_path))

                if not pose_data:
                    print(f"Warning: No valid pose data for {variation_dir.name}")
                    continue

                # Combine multi-view poses
                combined_poses = []
                for frame_idx in range(min(len(p) for p in pose_data.values())):
                    combined_pose = self.view_combiner.combine_poses(pose_data, frame_idx)
                    if combined_pose is not None:
                        combined_poses.append(combined_pose)

                if not combined_poses:
                    print(f"Warning: No poses could be combined for {variation_dir.name}")
                    continue

                # Save processed data
                np.save(demo_output_dir / f"{variation_dir.name}_poses.npy", combined_poses)
                training_data.append({'poses': combined_poses, 'metadata': {'variation': variation_dir.name}})

        return training_data

    def convert_poses_to_joints(self, poses: np.ndarray) -> np.ndarray:
        """Converts human pose data to robot joint angles using inverse kinematics."""
        joint_angles = []

        for pose in poses:
            try:
                position = pose[:3].astype(np.float64)  # Extract position (x, y, z)
                orientation = pose[3:7].astype(np.float64) if len(pose) >= 7 else np.array([1.0, 0.0, 0.0, 0.0])

                angles = self.solve_inverse_kinematics(position, orientation)
                joint_angles.append(angles)

            except Exception as e:
                print(f"Error converting pose to joints: {e}")
                joint_angles.append(np.zeros(7))

        return np.array(joint_angles)

    def solve_inverse_kinematics(self, target_pos: np.ndarray, target_quat: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Solves inverse kinematics for the robot arm."""
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        if np.any(np.isnan(target_pos)) or np.any(np.isnan(target_quat)):
            return np.zeros(7)

        for _ in range(max_iter):
            jacp = np.zeros((3, self.mj_model.nv))
            jacr = np.zeros((3, self.mj_model.nv))
            bodyid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")

            current_pos = self.mj_data.body("panda_hand").xpos
            current_quat = self.mj_data.body("panda_hand").xquat

            pos_error = target_pos - current_pos
            rot_error = np.zeros(3, dtype=np.float64)
            mujoco.mju_subQuat(rot_error, current_quat, target_quat)

            if np.linalg.norm(pos_error) < 1e-3 and np.linalg.norm(rot_error) < 1e-3:
                break

            mujoco.mj_jacBody(self.mj_model, self.mj_data, jacp, jacr, bodyid)
            J = np.concatenate((jacp, jacr))
            error = np.concatenate([pos_error, rot_error])

            qvel = np.linalg.pinv(J) @ error
            self.mj_data.qvel[:7] = qvel[:7]
            mujoco.mj_step(self.mj_model, self.mj_data)

        return self.mj_data.qpos[:7]

    def train_model(self, training_data: List[Dict]):
        """Trains the DAML model using processed demonstration data."""
        joint_sequences = []
        for data in training_data:
            joint_seq = self.convert_poses_to_joints(np.array(data['poses']))
            joint_sequences.append(joint_seq)

        # Define model configuration
        config = DAMLConfig(
            input_dim=joint_sequences[0].shape[1],
            hidden_dim=128,
            output_dim=7,
            num_layers=3
        )

        model = DAMLModel(config)
        model_save_path = self.model_save_dir / "franka_model.pth"

        torch.save({'model_state_dict': model.state_dict(), 'config': config}, model_save_path)
        print(f"Model saved to {model_save_path}")

        return model

def main():
    """Main function to run the training pipeline."""
    try:
        pipeline = RobotTrainingPipeline()
        training_data = pipeline.process_demonstrations("demonstrations", "processed_demonstrations")

        if training_data:
            pipeline.train_model(training_data)
            print("\nTraining complete!")
        else:
            print("\nNo training data was collected!")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
