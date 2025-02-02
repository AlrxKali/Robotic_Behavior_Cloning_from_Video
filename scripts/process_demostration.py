import argparse
from pathlib import Path
from pose_processing.pose_extractor import PoseExtractor
from pose_processing.view_combiner import ViewCombiner
import numpy as np

def process_demonstrations(input_dir: str, output_dir: str):
    """Processes demonstration videos to extract pose data and combine multi-view poses."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    pose_extractor = PoseExtractor()
    view_combiner = ViewCombiner()
    
    for demo_folder in input_path.glob("demo_*"):
        demo_output = output_path / demo_folder.name
        demo_output.mkdir(exist_ok=True)

        for variation_folder in demo_folder.glob("variation_*"):
            print(f"Processing {variation_folder.name}...")
            
            videos = list(variation_folder.glob("*.MOV"))
            if not videos:
                print(f"Warning: No videos found in {variation_folder}")
                continue
            
            pose_data = {}
            for video in videos:
                angle = video.stem
                print(f"Extracting poses from {angle} view...")
                pose_data[angle] = pose_extractor.process_video(str(video))

            if not pose_data:
                continue

            combined_poses = []
            for frame_idx in range(min(len(p) for p in pose_data.values())):
                combined_pose = view_combiner.combine_poses(pose_data, frame_idx)
                if combined_pose is not None:
                    combined_poses.append(combined_pose)

            if combined_poses:
                np.save(demo_output / f"{variation_folder.name}_poses.npy", combined_poses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process demonstration videos for pose extraction.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw demonstrations.")
    parser.add_argument("--output", type=str, required=True, help="Path to save processed demonstrations.")

    args = parser.parse_args()
    process_demonstrations(args.input, args.output)
