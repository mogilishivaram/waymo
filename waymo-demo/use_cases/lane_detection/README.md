# Lane detection and Waymo lane investigation

This use case contains a classical lane-candidate baseline and exploratory
tools for understanding the lane information available in the Waymo record.
It does **not** contain a trained SCNN model, verified lane annotations, or a
lane-segmentation benchmark.

## Current components

### 1. OpenCV lane baseline

Run from the project root:

```powershell
python use_cases\lane_detection\scripts\detect_lanes.py
```

This reads `data\images\front_*.jpg`, applies white/yellow color masks,
Canny edges, a road-region mask, morphology, and probabilistic Hough lines.
It writes:

- `outputs\lane_detection\lane_detection_baseline.mp4`
- `outputs\lane_detection\lane_detection_preview.jpg`
- seven first-frame diagnostics in
  `outputs\lane_detection\diagnostics`

The result is a visual candidate overlay. It is not a semantic lane mask and
does not claim that every detected line is a real lane marking.

### 2. Color-mask candidate generation

```powershell
python use_cases\lane_detection\scripts\generate_lane_mask_candidates.py
```

The generator searches `datasets\lanes\images` and `input` for original
camera images. It ignores generated folders and generated suffixes to avoid
reprocessing its own outputs. It applies a conservative road ROI, HSV/LAB
thresholds, morphology, and connected-component filtering.

For each source image it writes:

- `datasets\lanes\mask_candidates\candidate_XXXXX_<source>_mask.png`
- `datasets\lanes\mask_yellow\candidate_XXXXX_<source>_yellow.png`
- `datasets\lanes\mask_white\candidate_XXXXX_<source>_white.png`
- `datasets\lanes\mask_overlays\candidate_XXXXX_<source>_overlay.jpg`

White candidates are shown in magenta and yellow candidates in yellow in the
review overlays. These are weak candidate masks only. They must be reviewed
and corrected before being placed in `datasets\lanes\annotations`.

Earlier threshold experiments may also be present in
`mask_candidates_v2`, `mask_yellow_v2`, `mask_white_v2`, and
`mask_overlays_v2`. The current script does not write to those versioned
folders.

### 3. Waymo record and projection diagnostics

The following scripts use the preserved TFRecord under `input`:

```powershell
python use_cases\lane_detection\scripts\inspect_waymo_lane_labels.py
python use_cases\lane_detection\scripts\inspect_waymo_map_features.py
python use_cases\lane_detection\scripts\inspect_projected_feature_ids.py
python use_cases\lane_detection\scripts\check_camera_timing.py
python use_cases\lane_detection\scripts\check_camera_calibration.py
python use_cases\lane_detection\scripts\debug_map_projection.py
python use_cases\lane_detection\scripts\project_waymo_map_to_camera.py
python use_cases\lane_detection\scripts\project_waymo_map_rolling_shutter.py
python use_cases\lane_detection\scripts\inspect_projected_lidar_labels.py
```

Their purposes are:

- `inspect_waymo_lane_labels.py`: inspect lane-related labels in frames.
- `inspect_waymo_map_features.py`: inspect map features in the record.
- `inspect_projected_feature_ids.py`: inspect projected map-feature IDs.
- `check_camera_timing.py`: inspect frame and camera timing.
- `check_camera_calibration.py`: print camera dimensions, intrinsics, and
  extrinsics.
- `debug_map_projection.py`: run a focused projection/debug inspection.
- `project_waymo_map_to_camera.py`: project map road-line features into the
  camera view.
- `project_waymo_map_rolling_shutter.py`: repeat projection with rolling-
  shutter timing and write a report.
- `inspect_projected_lidar_labels.py`: inspect projected LiDAR labels.

These scripts write exploratory PNG/report artifacts under
`outputs\lane_detection`. Treat the results as diagnostics, not verified
annotations. Projection quality must be checked visually against the camera
image.

## Data layout

```text
datasets\lanes\
├── images\          Original camera images
├── annotations\     Reserved for reviewed annotations
├── mask_candidates\ Combined candidate masks
├── mask_yellow\     Yellow candidate masks
├── mask_white\      White candidate masks
└── mask_overlays\   Review overlays
```

## Planned work

The selected future model direction remains SCNN or another trained
lane-segmentation model. Before training, the project needs a defined
annotation schema, reviewed labels, a train/validation split, lane geometry
metrics, and a reproducible evaluation command.
