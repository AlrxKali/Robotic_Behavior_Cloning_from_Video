import cv2
import mediapipe as mp
import numpy as np
import time
from typing import List
from .pose_data import PoseData
from .filters import OutlierRemoval, CoordinateNormalizer, InterpolationHandler

class PoseExtractor:
    """Extracts human pose keypoints from video using MediaPipe"""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        self.processors = [
            OutlierRemoval(),
            CoordinateNormalizer(),
            InterpolationHandler()
        ]

    def process_video(self, video_path: str) -> List[PoseData]:
        poses = []
        cap = cv2.VideoCapture(video_path)
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0], lm.visibility]
                                      for lm in results.pose_landmarks.landmark])

                pose_data = PoseData(
                    timestamp=time.time(),
                    keypoints=landmarks,
                    frame_id=frame_id,
                    metadata={'frame_shape': frame.shape}
                )

                # Apply processing pipeline
                for processor in self.processors:
                    pose_data = processor.process(pose_data)

                poses.append(pose_data)

            frame_id += 1

        cap.release()
        return poses
