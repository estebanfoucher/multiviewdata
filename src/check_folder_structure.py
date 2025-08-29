import os
import json

class StereoDataFolderStructure:
    """
    Structure of the stereo data folder:
    - stereo_data_folder
        - scene_<number>
            - camera_1
                - video.mp4
            - camera_2
                - video.mp4
            - config.yml
            - calibration_extrinsics_pattern_specs.yml
        - calibration_intrinsics_<number>
            - camera_1
                - video.mp4
            - camera_2
                - video.mp4

    """
    
    def __init__(self, folder_path):
        self.folder_path = folder_path
    
    def check_folder_exists(self):
        if not os.path.exists(self.folder_path):
            raise ValueError(f"Folder {self.folder_path} does not exist")

    def check_subfolders_names(self):
        admitted_prefixes = ["scene", "calibration_intrinsics"]
        ignored = [".DS_Store"]
        
        for subfolder in os.listdir(self.folder_path):
            if subfolder in ignored:
                continue
            if not any(subfolder.startswith(prefix) for prefix in admitted_prefixes):
                raise ValueError(f"Subfolder {subfolder} has an invalid name")
    
    def check_scene_folder_folder_structure(self, folder_path):
        """
        Check that the folder has two subfolders named camera_1 and camera_2 each with a video.mp4 file
        Check that the folder has a parameters.yml file
        Check that the folder has a calibration_extrinsics_pattern_specs.yml file
        """        
        expected_files = ["camera_1", "camera_2", "parameters.yml", "calibration_extrinsics_pattern_specs.yml"]
        actual_files = [f for f in os.listdir(folder_path) if f not in [".DS_Store"]]
        
        # Check that all expected files/folders are present
        for expected in expected_files:
            if expected not in actual_files:
                raise ValueError(f"Expected '{expected}' in {folder_path}, but found: {actual_files}")
        
        # check that the camera_1 and camera_2 folders have a video.mp4 file
        assert self.check_folder_has_one_video_file(os.path.join(folder_path, "camera_1"))
        assert self.check_folder_has_one_video_file(os.path.join(folder_path, "camera_2"))
        
        # check that the parameters.yml file exists
        assert os.path.isfile(os.path.join(folder_path, "parameters.yml"))
        
        # check that the calibration_extrinsics_pattern_specs.yml file exists
        assert os.path.isfile(os.path.join(folder_path, "calibration_extrinsics_pattern_specs.yml"))
        
    def check_calibration_intrinsics_folder_structure(self, folder_path):
        """
        Check that the folder has two subfolders named camera_1 and camera_2 each with a video.mp4 file
        """
        expected_folders = ["camera_1", "camera_2"]
        actual_files = [f for f in os.listdir(folder_path) if f not in [".DS_Store"]]
        
        # Check that all expected folders are present
        for expected in expected_folders:
            if expected not in actual_files:
                raise ValueError(f"Expected folder '{expected}' in {folder_path}, but found: {actual_files}")
        
        # check that the camera_1 and camera_2 folders have a video.mp4 file
        assert self.check_folder_has_one_video_file(os.path.join(folder_path, "camera_1"))
        assert self.check_folder_has_one_video_file(os.path.join(folder_path, "camera_2"))
        
        # assert each camera folder has a pattern_specs.yml file
        assert os.path.isfile(os.path.join(folder_path, "camera_1", "pattern_specs.yml")), f"Pattern specs file not found in {os.path.join(folder_path, 'camera_1')}"
        assert os.path.isfile(os.path.join(folder_path, "camera_2", "pattern_specs.yml")), f"Pattern specs file not found in {os.path.join(folder_path, 'camera_2')}"
    
    def check_folder_has_one_video_file(self, folder_path):
        video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4") or f.endswith(".MP4")]
        if len(video_files) != 1:
            raise ValueError(f"Folder {folder_path} has {len(video_files)} video files, expected 1")
        return video_files[0]

    def check_folder_structure(self):
        self.check_folder_exists()
        self.check_subfolders_names()
        
        for subfolder in os.listdir(self.folder_path):
            if subfolder in [".DS_Store"]:
                continue
            if subfolder.startswith("scene"):
                self.check_scene_folder_folder_structure(os.path.join(self.folder_path, subfolder))
            else:
                self.check_calibration_intrinsics_folder_structure(os.path.join(self.folder_path, subfolder))

def test_check_folder_structure():
    with open("config.json", "r") as f:
        config = json.load(f)
    STEREO_DATA_FOLDER_NAME = config["stereo_data_folder_name"]
    stereo_data_folder_structure = StereoDataFolderStructure(STEREO_DATA_FOLDER_NAME)
    stereo_data_folder_structure.check_folder_structure()

if __name__ == "__main__":
    test_check_folder_structure()
    print("data folder structure valid")