from stereo_data_folder_structure import load_scene_folder_structure
from intrinsics_calibration import  IntrinsicCalibration
from extrinsics_calibration import StereoTagDetector, calibrate_stereo_many, get_summary
import os
from video_utils import get_unique_video_name
import yaml
import json
from utils import load_parameters
from video_utils import Video
import numpy as np
from loguru import logger
import cv2
from typing import List, Tuple, Dict, Any   
import json
with open("config.json", "r") as f:
    config = json.load(f)
TEMPORAL_CALIB_STEP_SEC = config["temporal_calib_step_sec"]


class Scene:
    def __init__(self, scene_name: str):
        self.scene_name = scene_name
        self.scene_folder_structure = load_scene_folder_structure(scene_name)
        self.parameters = self.get_parameters()
        self.cameras = ["camera_1", "camera_2"]
        
    def calibrate_all_intrinsics(self):
        for camera_name in self.cameras:
            self.calibrate_intrinsics(camera_name)
        
    def calibrate_intrinsics(self, camera_name: str):
        print(f"Calibrating intrinsics for {camera_name}")
        calibration_folder_path = os.path.join(self.scene_folder_structure.folder_path, self.scene_folder_structure.get_calibration_intrinsics_folder_name())
        calibration_camera_path = os.path.join(calibration_folder_path, camera_name)
        checkerboard_specs_path = os.path.join(calibration_camera_path, "checkerboard_specs.yml")
        video_name = get_unique_video_name(calibration_camera_path)
        video_path = os.path.join(calibration_camera_path, video_name)
        save_path = os.path.join(calibration_folder_path, camera_name, f"intrinsics.json")
        intrinsic_calibration = IntrinsicCalibration(video_path, checkerboard_specs_path, save_path)
        camera_matrix, dist_coeffs, reprojection_error = intrinsic_calibration.calibrate(save_images=False)
        print(f"Intrinsic calibration saved to {save_path}")
        return camera_matrix, dist_coeffs, reprojection_error
    
    def get_calibration_intrinsics(self):
        intrinsic_calibration = {}
        # if already calibrated, load the intrinsics.json file, otherwise calibrate and save the intrinsics.json file
        for camera_name in self.cameras:
            if os.path.exists(os.path.join(self.scene_folder_structure.folder_path, self.scene_folder_structure.get_calibration_intrinsics_folder_name(), camera_name, "intrinsics.json")):
                with open(os.path.join(self.scene_folder_structure.folder_path, self.scene_folder_structure.get_calibration_intrinsics_folder_name(), camera_name, "intrinsics.json"), "r") as f:
                    intrinsic_calibration[camera_name] = json.load(f)
            else:
                camera_matrix, dist_coeffs, reprojection_error = self.calibrate_intrinsics(camera_name)
                intrinsics_dict = {
                    "camera_matrix": camera_matrix.tolist(),
                    "dist_coeffs": dist_coeffs.tolist(),
                    "reprojection_error": reprojection_error
                }
                intrinsic_calibration[camera_name] = intrinsics_dict
                with open(os.path.join(self.scene_folder_structure.folder_path, self.scene_folder_structure.get_calibration_intrinsics_folder_name(), camera_name, "intrinsics.json"), "w") as f:
                    json.dump(intrinsics_dict, f, indent=2)
        return intrinsic_calibration
    
    def get_parameters(self):
        with open(os.path.join(self.scene_folder_structure.folder_path, self.scene_name, "parameters.yml"), "r") as f:
            parameters = yaml.safe_load(f)
        return parameters

class ExtrinsicCalibration:
    def __init__(self, scene: Scene):
        self.scene = scene
        camera_1_path = os.path.join(self.scene.scene_folder_structure.folder_path, self.scene.scene_name, "camera_1")
        camera_2_path = os.path.join(self.scene.scene_folder_structure.folder_path, self.scene.scene_name, "camera_2")
        
        video_1_name = get_unique_video_name(camera_1_path)
        video_2_name = get_unique_video_name(camera_2_path)
        
        self.video_1 = Video(os.path.join(camera_1_path, video_1_name))
        self.video_2 = Video(os.path.join(camera_2_path, video_2_name))
        self.save_path = os.path.join(self.scene.scene_folder_structure.folder_path, self.scene.scene_name, "extrinsics.json")
        self.extrinsics_calibration_pattern_specs = self.get_extrinsics_calibration_pattern_specs()
        self.sync_frame_offset = self.get_sync_frame_offset()
        self.stereo_tag_detector = StereoTagDetector(config_path=os.path.join(self.scene.scene_folder_structure.folder_path, self.scene.scene_name, "extrinsics_calibration_pattern_specs.yml"))
        self.image_numbers_list = self.get_image_numbers_list()
        self.check_video_fps()
        
    def get_extrinsics_calibration_pattern_specs(self):
        return load_parameters(os.path.join(self.scene.scene_folder_structure.folder_path, self.scene.scene_name, "extrinsics_calibration_pattern_specs.yml"))
    
    def get_sync_frame_offset(self):
        parameters = self.scene.parameters
        camera_1_sync_event_list_F = parameters["camera_1"]["sync_event_time_F"]
        camera_2_sync_event_list_F = parameters["camera_2"]["sync_event_time_F"]
        
        diff = np.array(camera_1_sync_event_list_F) - np.array(camera_2_sync_event_list_F)
        if not np.all(diff == diff[0]):
            logger.warning(f"frame offset diff is not constant: {diff}")
        diff_mean = np.mean(diff)
        return int(diff_mean)
    
    def get_image_numbers_list(self):
        # Convert fps to int for slice step
        step = int(self.video_1.get_fps()*TEMPORAL_CALIB_STEP_SEC)
        camera_1_image_numbers_list = list(range(self.scene.parameters["camera_1"]["sync_calib_start_frame"], self.scene.parameters["camera_1"]["sync_calib_end_frame"], step))
        #for camera_2, substract the sync_frame_offset
        camera_2_image_numbers_list = [frame - self.sync_frame_offset for frame in camera_1_image_numbers_list]
        return camera_1_image_numbers_list, camera_2_image_numbers_list
    
    def check_video_fps(self):
        assert self.video_1.get_fps() == self.video_2.get_fps(), "Video fps are not the same"
        logger.debug(f"Video fps: {self.video_1.get_fps()}")
        logger.debug(f"Video fps: {self.video_2.get_fps()}")
    
    def check_resolution(self):
        assert self.video_1.get_resolution() == self.video_2.get_resolution(), "Video resolutions are not the same"
        logger.debug(f"Video resolution: {self.video_1.get_resolution()}")
        logger.debug(f"Video resolution: {self.video_2.get_resolution()}")
    
    def get_intrinsics(self):
        """Get the intrinsic calibration parameters for both cameras."""
        intrinsics = self.scene.get_calibration_intrinsics()
        
        # Convert from lists back to numpy arrays
        self.camera_matrix1 = np.array(intrinsics["camera_1"]["camera_matrix"])
        self.dist_coeffs1 = np.array(intrinsics["camera_1"]["dist_coeffs"])
        self.camera_matrix2 = np.array(intrinsics["camera_2"]["camera_matrix"])
        self.dist_coeffs2 = np.array(intrinsics["camera_2"]["dist_coeffs"])
        
        logger.debug(f"Loaded intrinsics for camera_1: reprojection_error={intrinsics['camera_1']['reprojection_error']}")
        logger.debug(f"Loaded intrinsics for camera_2: reprojection_error={intrinsics['camera_2']['reprojection_error']}")
    
    def calibrate_extrinsics(self):
        # Load intrinsic calibration parameters
        self.get_intrinsics()
        
        camera_1_image_numbers_list, camera_2_image_numbers_list = self.get_image_numbers_list()
        successful_pairs = []
        object_points_list = []
        image_points1_list = []
        image_points2_list = []
        image_size = None
        
        for camera_1_image_number, camera_2_image_number in zip(camera_1_image_numbers_list, camera_2_image_numbers_list):
            camera_1_frames = self.video_1.get_frames([camera_1_image_number])
            camera_2_frames = self.video_2.get_frames([camera_2_image_number])
            
            if not camera_1_frames or not camera_2_frames:
                continue
                
            camera_1_image = camera_1_frames[0]
            camera_2_image = camera_2_frames[0]
            p3d, p2d1, p2d2 = self.stereo_tag_detector.get_correspondences(camera_1_image, camera_2_image)
            if p3d is None or p2d1 is None or p2d2 is None:
                continue
            
            successful_pairs.append((camera_1_image_number, camera_2_image_number))
            
            object_points_list.append(p3d)
            image_points1_list.append(p2d1)
            image_points2_list.append(p2d2)
            if image_size is None:
                image_size = (camera_1_image.shape[1], camera_1_image.shape[0])
        
        results = calibrate_stereo_many(
            object_points_list,
            image_points1_list,
            image_points2_list,
            self.camera_matrix1,
            self.dist_coeffs1,
            self.camera_matrix2,
            self.dist_coeffs2,
            image_size
        )
        
        self.save_successful_pairs(successful_pairs)
        self.save_extrinsics(results)
        return results
    
    def load_extrinsics(self):
        path = os.path.join(self.scene.scene_folder_structure.folder_path, self.scene.scene_name, "extrinsics_calibration.json")
        with open(path, "r") as f:
            return json.load(f)
    
    def save_extrinsics(self, results: Dict[str, Any]):
        path = os.path.join(self.scene.scene_folder_structure.folder_path, self.scene.scene_name, "extrinsics_calibration.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
    
    def save_successful_pairs(self, successful_pairs: List[Tuple[int, int]]):
        # save to a folder called successful_pairs to 360p
        #successful_pairs/pair_i 
        #successful_pairs/pairs.json

        for i, (camera_1_image_number, camera_2_image_number) in enumerate(successful_pairs):
            camera_1_frames = self.video_1.get_frames([camera_1_image_number])
            camera_2_frames = self.video_2.get_frames([camera_2_image_number])
            # mkdir successful_pairs if not exists
            path = os.path.join(self.scene.scene_folder_structure.folder_path, self.scene.scene_name, "successful_pairs")
            os.makedirs(path, exist_ok=True)
            if camera_1_frames and camera_2_frames:
                camera_1_image = camera_1_frames[0]
                camera_2_image = camera_2_frames[0]
                
                # Resize images to 360p (640x360)
                camera_1_image_360p = cv2.resize(camera_1_image, (640, 360))
                camera_2_image_360p = cv2.resize(camera_2_image, (640, 360))
                
                cv2.imwrite(os.path.join(path, f"pair_{i}_camera1.png"), camera_1_image_360p)
                cv2.imwrite(os.path.join(path, f"pair_{i}_camera2.png"), camera_2_image_360p)

    def get_extrinsics_summary(self):
        extrinsics = self.load_extrinsics()
        return get_summary(extrinsics)
    
    def save_extrinsics_summary(self):
        measured_baseline = self.scene.parameters["measured_baseline_m"]
        summary = self.get_extrinsics_summary()
        summary += f"\nMeasured Baseline: {measured_baseline:.3f} meters"
        with open(os.path.join(self.scene.scene_folder_structure.folder_path, self.scene.scene_name, "extrinsics_summary.txt"), "w") as f:
            f.write(summary)