# Traffic-light detection and classification

- **Algorithm:** A separate region-localization stage followed by a
  Random Forest classifier with `red`, `yellow`, and `green` labels.
- **Dataset:** S2TLD is staged under `datasets/traffic_lights/coco-s2tld`.
  Run `python scripts/validate_traffic_light_dataset.py` before training.
- **Status:** Dataset validation is available, but the numeric class mapping is
  not source-manifest-confirmed. `datasets/traffic_lights/data.draft.yaml`
  is intentionally labeled draft and must not be treated as final.
- **Architecture:** Keep localization and classification separate. Use the
  detector boxes to produce crops, then train the required Random Forest on
  those state-labeled crops. YOLO boxes alone do not provide a trained
  classifier or guarantee clean crops.

Next steps are to confirm the original numeric class order from the source
annotation convention, implement crop extraction and Random Forest training,
evaluate localization and classification separately, and generate
`outputs/traffic_light_demo.mp4`.
