This repo aims to check preprocess multi-view data.

Aims :
- organize data
- get cameras intrinsincs and extrinsics
- syncronize videos

Your folder structure should match the following :

- stereo_data_folder/
  - scene_x/
    - camera_1/
      - video.mp4
    - camera_2/
      - video.mp4
    - parameters.yml
    - calibration_extrinsics_pattern_specs.yml
  - calibration_intrinsics_x/
    - camera_1/
      - video.mp4
      - pattern_specs.yml
    - camera_2/
      - video.mp4
      - pattern_specs.yml


1. Put your data folder name in config.json and check your structure by running 

`bash scripts/check_folder_structure.sh`

