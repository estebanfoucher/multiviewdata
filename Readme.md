# Multi-View Data Preprocessing

This repo aims to preprocess multi-view data.

## Aims
- Organize data
- Get cameras intrinsics and extrinsics
- Synchronize videos

## Installation

1. Create virtual environment:
   ```bash
   virtualenv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Your folder structure should match the following:

```
stereo_data_folder/
├── scene_x/
│   ├── camera_1/
│   │   └── video.mp4
│   ├── camera_2/
│   │   └── video.mp4
│   ├── parameters.yml
│   └── extrinsics_calibration_pattern_specs.yml
└── calibration_intrinsics_x/
    ├── camera_1/
    │   ├── video.mp4
    │   └── checkerboard_specs.yml
    └── camera_2/
        ├── video.mp4
        └── checkerboard_specs.yml
```

1. Put your data folder name in `config.json`
2. Check your structure by running:
   ```bash
   bash scripts/check_folder_structure.sh
   ```
3. Run the main script:
   ```bash
   python src/main.py
   ```

