# Waymo Road-Scene Perception Demo

This repository contains a Windows/Python computer-vision demo built from
Waymo road-scene data. It brings together three road-scene tasks:

1. **Traffic-light detection and state classification** using an Ultralytics
   YOLO model and a Streamlit image-upload application.
2. **Vehicle detection and tracking** using the prepared Waymo vehicle data
   and Ultralytics tracking scripts.
3. **Lane-candidate detection** using OpenCV color thresholds, edge detection,
   and probabilistic Hough lines.
4. **Waymo lane-data investigation** using TFRecord camera calibration,
   timing, map-feature, LiDAR-label, and camera-projection diagnostics.

The repository also contains the original Waymo extraction, prepared datasets,
trained weights, generated videos, and diagnostic images. Large data and
model files are intentionally kept in their existing locations; this README
documents how they are connected rather than copying them.

## What is implemented

| Component | Entry point | Current behavior |
| --- | --- | --- |
| Traffic-light image demo | `app.py` | Upload an image, run YOLO inference, display boxes and classes, and download an annotated PNG. |
| Traffic-light debugger | `debug_traffic.py` | Open a single image in a Tkinter dialog, run the traffic-light YOLO model, and print the detections without the Streamlit UI. |
| Combined road-video demo | `process_road_video.py` | Select a video, run traffic-light YOLO, vehicle YOLO, and OpenCV lane overlays, then write a new MP4 beside the input. |
| Integrated scene demo | `use_cases/object_tracking/scripts/integrated_scene.py` | Runs vehicle tracking and traffic-light detection together on extracted front-camera frames and writes `outputs/combined_scene_demo.mp4`. |
| Vehicle tracking | `use_cases/object_tracking/scripts/track_vehicles.py` | Runs vehicle detection/tracking on extracted front-camera images and writes a tracking video/result set. |
| Vehicle prediction | `use_cases/object_tracking/scripts/predict_vehicle.py` | Runs saved vehicle weights on images and writes detection images. |
| Lane baseline | `use_cases/lane_detection/scripts/detect_lanes.py` | Produces a classical OpenCV lane-candidate baseline and intermediate diagnostics; it is not a trained SCNN model. |
| Lane mask candidates | `use_cases/lane_detection/scripts/generate_lane_mask_candidates.py` | Generates combined, yellow, white, and review-overlay candidate images from original camera photographs. |
| Waymo lane inspection | `use_cases/lane_detection/scripts/*.py` | Inspects the Waymo TFRecord and investigates map-feature, LiDAR-label, timing, calibration, and camera projection. |
| Traffic-light dataset utilities | `scripts/*.py` | Inspect, clean, validate, annotate, convert, and evaluate traffic-light data. |
| Shared helper utilities | `shared/*.py` | Provides small reusable helpers for locating images, creating OpenCV video writers, and drawing labels in notebooks/scripts. |

The selected SCNN design is documented in
`use_cases/lane_detection/README.md`, but SCNN training/inference is not
implemented in this checkout. Do not interpret the OpenCV lane lines as
ground-truth lane segmentation.

## Support modules and legacy staging areas

A few repository folders are not main pipelines but still participate in the
project layout:

- `shared/video_utils.py`: helper functions for discovering image files and
  creating OpenCV video writers.
- `shared/visualization.py`: small utility for drawing OpenCV text labels on
  frames.
- `traffic_light_detection/`: a root-level staging area with `data`, `models`,
  `outputs`, and `scripts` directories. It is present for compatibility with the
  older project organization, but the active app and current demo logic are
  centered on `app.py`, `runs/`, and `use_cases/`.

## Requirements

- Windows 10/11
- Python 3.10 or newer
- PowerShell
- A working C/C++ build environment may be required by some scientific
  packages if a compatible wheel is not available.
- GPU is optional. CPU inference works but is slower. CUDA is used
  automatically by `app.py` when PyTorch reports that CUDA is available.

The dependency list is in `requirements.txt`:

- `ultralytics`: YOLO loading, inference, training, and tracking
- `opencv-python`: image/video I/O and classical lane processing
- `numpy`: array and numerical operations
- `torch`: PyTorch runtime and CUDA detection
- `tensorflow`: Waymo tooling compatibility
- `waymo-open-dataset-tf-2-11-0`: Waymo TFRecord utilities
- `scikit-learn`: traffic-light evaluation utilities
- `pyyaml`: YAML configuration and dataset metadata

## Start locally on Windows

Run every command below in **PowerShell**. The first command changes into the
project directory; all relative paths in the rest of this README assume that
directory.

```powershell
cd D:\shivaram\waymo-demo
```

### 1. Create or reuse a virtual environment

This project already contains `.venv` and `.venv-waymo` directories on the
current machine. A new environment is safer when setting up the project on
another machine:

```powershell
cd D:\shivaram\waymo-demo
py -3.11 -m venv .venv-local
```

What each line does:

- `cd D:\shivaram\waymo-demo` selects the repository root.
- `py -3.11` asks the Windows Python launcher for Python 3.11.
- `-m venv .venv-local` creates an isolated environment in `.venv-local`.

Activate it:

```powershell
.\.venv-local\Scripts\Activate.ps1
```

If PowerShell blocks local scripts for the current user, allow signed/local
scripts once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv-local\Scripts\Activate.ps1
```

The prompt should show `(.venv-local)` after activation. To leave the
environment later, run:

```powershell
deactivate
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

- The first line updates `pip` inside the active environment.
- The second line installs every package listed in `requirements.txt`.
- Do not install packages globally when the virtual environment is active.

The Waymo package and TensorFlow can be large. If installation fails because
of an unavailable wheel, use a Python version supported by the package
combination or use the already prepared `.venv-waymo` environment.

### 3. Verify the checkout without changing it

```powershell
python scripts\verify_project.py
```

This read-only check validates the expected project layout and reports missing
or pending assets. It does not download data, train a model, modify labels, or
create output videos. It checks the preserved Waymo TFRecord, extracted image,
vehicle dataset, object-tracking metadata, model weights, and object-tracking
demo. It reports traffic-light and lane image/annotation directories
separately because those are optional or still being developed.

### 4. Start the local Streamlit host

```powershell
python -m streamlit run app.py
```

Expected result:

```text
Local URL: http://localhost:8501
```

Open **http://localhost:8501** in a browser. The `app.py` command performs
the following steps:

1. Resolves the repository root from the location of `app.py`.
2. Loads
   `runs\detect\runs\detect\traffic_light_detection_v2\weights\best.pt`.
3. Stops with a visible error if that model file is missing or cannot load.
4. Shows a traffic-light image uploader for JPG, JPEG, and PNG files.
5. Displays the original image and its dimensions.
6. Lets the user choose a confidence threshold from `0.05` through `0.95`.
7. Lets the user choose an inference size of `640`, `800`, or `960`.
8. Uses CUDA device `0` when `torch.cuda.is_available()` is true; otherwise
   it uses the CPU.
9. Converts the uploaded RGB image to BGR before Ultralytics inference.
10. Runs YOLO inference when **Detect traffic lights** is clicked.
11. Converts the plotted BGR result back to RGB for browser display.
12. Shows the number of detections, image size, threshold, device, model
    path, class mapping, confidence, class ID, and `[x1,y1,x2,y2]` boxes.
13. Provides **Download annotated image**, which downloads
    `traffic_light_predictions.png`.

To stop the local server, return to the PowerShell window and press
`Ctrl+C`. To use another port:

```powershell
python -m streamlit run app.py --server.port 8502
```

Then open **http://localhost:8502**.

## Run the combined road-video demo

This is a desktop/PowerShell command, not a Streamlit page:

```powershell
python process_road_video.py
```

The script:

1. Checks the traffic-light model at
   `runs\detect\runs\detect\traffic_light_detection_v2\weights\best.pt`.
2. Checks the vehicle model at
   `use_cases\object_tracking\models\waymo_vehicle_baseline-3\weights\best.pt`.
3. Opens a Windows file picker for an input video.
4. Reads MP4, AVI, MOV, or MKV frames with OpenCV.
5. Applies the lane-candidate overlay to each frame.
6. Runs the vehicle model with confidence `0.30` and image size `640`.
7. Runs the traffic-light model with confidence `0.25` and image size `640`.
8. Draws both YOLO prediction layers on top of the lane overlay.
9. Writes `<input-name>_road_predictions.mp4` next to the selected input.
10. Releases the video reader and writer even if processing raises an error.

The lane layer is intentionally labeled either `LANE CANDIDATES` or
`LANES NOT CLEARLY DETECTED`. It uses an ROI, white/yellow color masks,
Canny edges, morphology, and Hough lines; it does not load an SCNN model.

## Lane-data workflow

The lane work currently has three complementary parts:

1. `detect_lanes.py` creates a video baseline from the extracted front-camera
   images in `data\images`.
2. `generate_lane_mask_candidates.py` creates weak color-based candidate masks
   from original camera photographs. These masks are inputs for review or
   future annotation; they are not verified labels.
3. The Waymo inspection/projection scripts investigate whether map and LiDAR
   information can provide a stronger lane-label source for the selected
   camera and timestamp.

### Generate lane mask candidates

Run from the project root:

```powershell
python use_cases\lane_detection\scripts\generate_lane_mask_candidates.py
```

The generator searches these locations for original images:

- `datasets\lanes\images`
- `input`

It ignores generated mask folders and generated filenames so that rerunning
the command does not recursively treat its own outputs as new source images.
For each source image it:

1. Builds a conservative lower-image road region of interest.
2. Finds white/light-gray candidates with HSV and LAB brightness thresholds.
3. Finds yellow candidates with HSV thresholds.
4. Removes small components and applies a small morphological opening.
5. Writes a combined binary mask.
6. Writes separate yellow and white binary masks.
7. Writes a review overlay where white candidates are magenta and yellow
   candidates are yellow.

Outputs are written to:

```text
datasets\lanes\mask_candidates\
datasets\lanes\mask_yellow\
datasets\lanes\mask_white\
datasets\lanes\mask_overlays\
```

The checkout may also contain `mask_candidates_v2`, `mask_yellow_v2`,
`mask_white_v2`, and `mask_overlays_v2` from earlier threshold experiments.
The current generator writes only to the four directories above and does not
overwrite or consume the versioned experiment folders.

The output filename starts with `candidate_00000_` (and so on) to avoid
collisions when source folders contain duplicate image names. The script
prints the number processed, skipped, and each output directory. If no source
images are found, it exits with an explanatory message and tells you to place
original photographs in `datasets\lanes\images`.

### Inspect Waymo lane metadata and camera geometry

These commands read the Waymo TFRecord configured in the scripts:

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

Use these scripts in the following order when diagnosing projection:

1. `inspect_waymo_lane_labels.py` inspects available lane-related frame
   labels.
2. `inspect_waymo_map_features.py` inspects map features in the record.
3. `check_camera_timing.py` checks frame/image timestamps and timing data.
4. `check_camera_calibration.py` prints front-camera dimensions, intrinsics,
   and extrinsics.
5. `debug_map_projection.py` performs a focused projection/debug run.
6. `project_waymo_map_to_camera.py` projects map road-line features into the
   camera image using the standard frame transform.
7. `project_waymo_map_rolling_shutter.py` repeats projection while accounting
   for rolling-shutter timing and writes a report.
8. `inspect_projected_feature_ids.py` and
   `inspect_projected_lidar_labels.py` inspect the resulting projected
   feature/label overlays.

These utilities are exploratory diagnostics. Their plotted points and lines
must be checked against the camera image before being used as annotations.
They do not train a lane model or establish a benchmark metric.

## Run the individual pipelines

Run these from `D:\shivaram\waymo-demo`.

### Object tracking

```powershell
python use_cases\object_tracking\scripts\integrated_scene.py
python use_cases\object_tracking\scripts\track_vehicles.py
python use_cases\object_tracking\scripts\predict_vehicle.py
```

`integrated_scene.py` creates a combined vehicle + traffic-light overlay video
from the extracted front-camera frames and writes
`outputs\combined_scene_demo.mp4`. `track_vehicles.py` uses the extracted
front-camera images in `data\images`, the prepared vehicle model/data
configuration, and the tracking API. `predict_vehicle.py` creates saved
prediction images for inspection. The existing object-tracking documentation
is in `use_cases\object_tracking\README.md`.

Training is explicit and can require a GPU:

```powershell
python use_cases\object_tracking\scripts\train_vehicle.py
```

Do not run training merely to use the existing demo; trained weights are
already preserved under
`use_cases\object_tracking\models\waymo_vehicle_baseline-3\weights`.

### Lane baseline

```powershell
python use_cases\lane_detection\scripts\detect_lanes.py
```

The script reads `data\images\front_*.jpg`, writes the baseline video to
`outputs\lane_detection\lane_detection_baseline.mp4`, saves a first-frame
preview to `outputs\lane_detection\lane_detection_preview.jpg`, and writes
seven diagnostic images under `outputs\lane_detection\diagnostics`.

The lane source folder currently contains `datasets\lanes\images` and the
generated candidate folders. The candidate folders are:

```text
datasets\lanes\
├── images\          Original camera photographs
├── annotations\     Reserved for reviewed/verified annotations
├── mask_candidates\ Combined binary candidates
├── mask_yellow\     Yellow-paint binary candidates
├── mask_white\      White/light-gray binary candidates
└── mask_overlays\   Human-review overlays
```

### Traffic-light inspection and evaluation utilities

These commands are useful when working on the traffic-light dataset:

```powershell
python scripts\inspect_box_schema.py
python scripts\inspect_labels.py
python scripts\check_boxes.py
python scripts\validate_traffic_light_dataset.py
python scripts\diagnose_traffic_colors.py
python scripts\evaluate_traffic_colors.py
```

The remaining scripts are specialized utilities:

- `extract_waymo.py`: extracts Waymo TFRecord camera data.
- `convert_waymo_yolo.py`: converts Waymo annotations to YOLO-style labels.
- `build_clean_traffic_light_dataset.py`: builds the cleaned traffic-light
  dataset structure.
- `review_traffic_light_classes.py`: reviews class assignments.
- `annotate_one_image.py`: interactively draws one image's traffic-light
  boxes; keys are documented by the script at startup.
- `annotate_traffic_light_review.py`: supports the traffic-light review
  workflow.

Inspect a script's header and source before running a data-changing utility.
Some tools write labels or derived datasets and are not read-only.

## Directory map

```text
waymo-demo/
├── app.py                         Streamlit traffic-light image app
├── process_road_video.py          Combined video inference and overlays
├── debug_traffic.py               Tkinter traffic-light image debugger
├── requirements.txt               Python dependencies
├── config/
│   └── project.yaml               Canonical dataset, model, and output paths
├── scripts/                       Extraction, conversion, validation, review
├── data/
│   ├── images/                    Extracted front-camera JPG frames
│   ├── yolo_vehicle/              Prepared vehicle YOLO data
│   ├── labels/                    Reserved/raw label location
│   ├── raw/                       Reserved/raw data location
│   └── panoptic/                  Reserved panoptic-data location
├── datasets/
│   ├── object_tracking/           Object-tracking dataset metadata
│   ├── traffic_lights/            Traffic-light downloads and cleaned data
│   ├── lanes/                     Lane images, candidate masks, annotations
│   └── waymo_raw/                 Raw Waymo dataset location
├── input/                         Waymo TFRecord input files
├── use_cases/
│   ├── object_tracking/           Vehicle scripts, weights, results, README
│   ├── traffic_light_detection/   Traffic-light notes and result locations
│   └── lane_detection/            Lane baseline script and documentation
├── runs/                          YOLO training/evaluation runs and weights
├── outputs/                       Generated demos and lane projection diagnostics
├── models/ and weights/           Additional model artifacts
├── tools/                         Vendored Waymo Open Dataset source
├── .venv/ and .venv-waymo/        Local environments; not application code
└── yolo11n.pt                     Root-level YOLO model artifact
```

## Configuration and important paths

`config\project.yaml` is the canonical path map:

- `datasets.waymo_raw` -> `input`
- `datasets.extracted_images` -> `data/images`
- `datasets.object_tracking` -> `data/yolo_vehicle`
- `datasets.traffic_lights` -> `datasets/traffic_lights`
- `datasets.lanes` -> `datasets/lanes`
- object-tracking output -> `outputs/object_tracking_demo.mp4`
- traffic-light output target -> `outputs/traffic_light_demo.mp4`
- lane output target -> `outputs/lane_detection_demo.mp4`

Some older scripts retain their own historical paths, which is why the actual
generated files may use names such as
`outputs\lane_detection\lane_detection_baseline.mp4` or
`outputs\combined_scene_demo.mp4`. Check the script being run when exact
output naming matters.

## Existing outputs and artifacts

The checkout currently includes generated examples such as:

- `outputs\combined_scene_demo.mp4`
- `outputs\object_tracking_demo.mp4`
- `outputs\lane_detection\lane_detection_baseline.mp4`
- lane diagnostic JPG files
- YOLO training metrics, confusion matrices, and `results.csv` files
- traffic-light `best.pt` weights under `runs\detect`

Outputs are evidence of previous runs, not a guarantee that a new run will
produce identical results. Video codecs, Python package versions, CPU/GPU,
confidence values, and model files can change the result.

## Troubleshooting

### `Model file not found`

Run from the repository root and confirm the expected file exists:

```powershell
Test-Path .\runs\detect\runs\detect\traffic_light_detection_v2\weights\best.pt
Test-Path .\use_cases\object_tracking\models\waymo_vehicle_baseline-3\weights\best.pt
```

The commands should print `True`. If they print `False`, restore the required
trained weights or update the model path in the relevant script.

### Streamlit opens but inference fails

Confirm that the environment is active and that the model can load:

```powershell
python -c "import torch, ultralytics; print(torch.__version__); print(ultralytics.__version__); print(torch.cuda.is_available())"
```

Also try a clear JPG/PNG traffic-light image and lower the confidence
threshold in the sidebar.

### MP4 cannot be written

`process_road_video.py` uses the OpenCV `mp4v` codec. Confirm that the input
video opens, that the destination is writable, and that no previous output
file is locked by a media player. The script raises an explicit error if the
reader or writer cannot be opened.

### No lane lines appear

This is expected on frames without visible lane markings. The lane code is a
candidate detector sensitive to camera angle, lighting, road texture, and
occlusion; it is not a trained or benchmarked lane-segmentation model.

## Data and model limitations

- Waymo data and derived datasets may have their own licenses, terms, and
  usage restrictions. Confirm those terms before redistribution.
- Model confidence is not a correctness guarantee.
- Existing demonstrations are not a substitute for quantitative evaluation.
- Traffic-light and lane outputs should be visually reviewed before being
  used for research conclusions.
- Do not commit credentials, private data, or newly generated large artifacts.

## License and attribution

This repository includes or references the Waymo Open Dataset tooling and
third-party Python packages. Follow the license and attribution requirements
of each upstream project and of the Waymo Open Dataset before publishing or
redistributing this code, data, models, or generated media.
