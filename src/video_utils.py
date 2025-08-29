import cv2
import numpy as np
import os
from typing import Tuple
import subprocess

class FrameExtractor:
    def __init__(self, video_path: str, output_dir: str, list_of_frames: list[int]):
        self.video_path = video_path
        self.output_dir = output_dir
        self.list_of_frames = list_of_frames

    def extract_frames(self):
        """
        Extract frames from the video at the given list of frames.
        """
        cap = cv2.VideoCapture(self.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_numbers = np.array(self.list_of_frames)
        frame_numbers = frame_numbers[frame_numbers < frame_count] # remove frames that are out of range
        frame_numbers = frame_numbers[frame_numbers >= 0] # remove negative frames
        
        for frame_number in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(f"{self.output_dir}/frame_{frame_number}.jpg", frame)
        cap.release()

class Video:
    def __init__(self, video_path: str):
        self.video_path = video_path

    def get_frame_count(self):
        cap = cv2.VideoCapture(self.video_path)
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_fps(self):
        cap = cv2.VideoCapture(self.video_path)
        return cap.get(cv2.CAP_PROP_FPS)
    
    def get_frames(self, list_of_frames: list[int]):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        for frame_number in list_of_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    
    def get_resolution(self) -> Tuple[int, int]:
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return width, height

def get_unique_video_name(folder_path: str) -> str:
    """
    Get the video name in the folder (mp4, MP4)
    """
    video_name = None
    for file in os.listdir(folder_path):
        if file.endswith(('.mp4', '.MP4')):
            video_name = file
            break
    return video_name
    
    