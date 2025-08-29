from video_utils import Video, get_unique_video_name
import yaml
from pathlib import Path
from typing import Dict
import json
import cv2
import numpy as np
from typing import Tuple, List
from loguru import logger
import argparse

def load_checkerboard_specs(specs_file: str) -> Dict:
    """Load checkerboard specifications from JSON or YAML file."""
    specs_path = Path(specs_file)
    if specs_path.suffix.lower() in ['.json']:
        with open(specs_path, 'r') as f:
            return json.load(f)
    elif specs_path.suffix.lower() in ['.yml', '.yaml']:
        with open(specs_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {specs_path.suffix}")

def find_corners_in_images(image_numbers_list: List[int], video: Video, pattern_size: Tuple[int, int], 
                          square_size: float) -> Tuple[List, List, List]:
    """Find checkerboard corners in defined images in the video.
    Args:
        image_numbers_list: List of image numbers to process.
        video: Video object.
        pattern_size: Tuple of (width, height) of the checkerboard pattern.
        square_size: Size of the square in the checkerboard pattern.
    Returns:
        object_points: List of 3D points in real world space.
        image_points: List of 2D points in image plane.
        successful_images: List of image numbers that were successfully processed.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    object_points = []  # 3D points in real world space
    image_points = []   # 2D points in image plane
    successful_images = []
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ..., (8,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    frames = video.get_frames(image_numbers_list)
    for frame_number, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            object_points.append(objp)
            image_points.append(corners2)
            successful_images.append(frame_number)
            logger.info(f"Found corners in {frame_number} - {frame_number}/{len(image_numbers_list)} images processed")
        else:
            logger.warning(f"No corners found in {frame_number} - {frame_number}/{len(image_numbers_list)} images processed")
    
    logger.info(f"Successfully processed {len(successful_images)}/{len(image_numbers_list)} images")
    return object_points, image_points, successful_images


def calibrate_camera(object_points: List, image_points: List, 
                    image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run camera calibration using OpenCV's calibrateCamera."""
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], 
                                        camera_matrix, dist_coeffs)
        error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    logger.info(f"Calibration error: {mean_error/len(object_points):.4f} pixels")
    return camera_matrix, dist_coeffs, mean_error/len(object_points)

class IntrinsicCalibration:
    def __init__(self, video_path: str, checkerboard_specs_path: str, save_path: str):
        logger.debug(f"Initializing IntrinsicCalibration with video_path: {video_path}, checkerboard_specs_path: {checkerboard_specs_path}, save_path: {save_path}")
        self.video_path = video_path
        self.checkerboard_specs_path = checkerboard_specs_path
        self.save_path = save_path
        self.video = Video(video_path)
        self.checkerboard_specs = load_checkerboard_specs(checkerboard_specs_path)
        logger.debug(f"Checkerboard specs: {self.checkerboard_specs}")
        self.image_numbers_list = self.get_image_numbers_list()
        logger.debug(f"Image numbers list: {self.image_numbers_list}")
    
    def calibrate(self):
        object_points, image_points, successful_images = find_corners_in_images(self.image_numbers_list, self.video, (self.checkerboard_specs["inner_corners_x"], self.checkerboard_specs["inner_corners_y"]), self.checkerboard_specs["square_size_mm"])
        camera_matrix, dist_coeffs, reprojection_error = calibrate_camera(object_points, image_points, self.video.get_resolution())
        self.save_calibration(camera_matrix, dist_coeffs, reprojection_error)
        logger.debug(f"Calibration results saved to {self.save_path}")

    
    def get_image_numbers_list(self) -> List[int]:
        """Get the list of image numbers from the video."""
        return list(range(self.video.get_frame_count()))[::24]
    
    def save_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, reprojection_error: float):
        """Save the calibration results to a file."""
        with open(self.save_path, 'w') as f:
            json.dump({"camera_matrix": camera_matrix.tolist(), "dist_coeffs": dist_coeffs.tolist(), "reprojection_error": reprojection_error}, f, indent=2)
            
def main():
    parser = argparse.ArgumentParser(description="Intrinsic calibration")
    parser.add_argument("calibration_folder_path", type=str, help="Path to the calibration folder")
    args = parser.parse_args()
    video_path = Path(args.calibration_folder_path) / get_unique_video_name(args.calibration_folder_path)
    checkerboard_specs_path = Path(args.calibration_folder_path) / "checkerboard_specs.yml"
    save_path = Path(args.calibration_folder_path) / "calibration_results.json"
   
    intrinsic_calibration = IntrinsicCalibration(video_path, checkerboard_specs_path, save_path)
    intrinsic_calibration.calibrate()

if __name__ == "__main__":
    main()