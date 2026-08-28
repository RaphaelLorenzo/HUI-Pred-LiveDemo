# HUI-Pred-LiveDemo
Live end-to-end demonstration of Human-Robot interaction anticipation

## Installation

- Locally using `conda` with pre-recorded data (images, videos, episodes...)
```
conda create --name huilive python=3.10
conda activate huilive
pip install -r requirements.txt
python folder_demo_yolo26.py
```

- Locally using `docker` (with a ROS2 node subscribing to )
```
cd docker_base
docker build -t huipreddemo_base:latest .
cd ../
sh docker_base/run_container.sh
python ros2_node_yolo26.py -bp -ip ./checkpoints/converted_mb_FineTuned_SingleImage_05_03_26_best_adaptative_f1/ --overlay_mode nodrawing
```

- On jetson using `docker` (with a ROS2 node subscribing to )
```
docker build -t huipreddemo_jetson:latest .
sh run_docker.sh
python ros2_node_yolo26.py -bp -ip ./checkpoints/converted_mb_FineTuned_SingleImage_05_03_26_best_adaptative_f1/ --overlay_mode nodrawing
```

## List of sequences per location for testing
When using `-ed` option for input

```yaml
    - "Room005": ['rosbag2_2025_07_10-10_29_13', 'rosbag2_2025_07_10-15_47_18'],
    - "EntranceCBack": ['rosbag2_2025_07_07-10_24_20', 'rosbag2_2025_07_16-13_12_03', 'rosbag2_2025_07_16-14_07_49', 'rosbag2_2025_07_16-15_50_45', 'rosbag2_2025_07_23-11_01_56', 'rosbag2_2025_07_23-12_18_45', 'rosbag2_2025_07_23-13_17_40', 'rosbag2_2025_07_23-14_30_55', 'rosbag2_2025_07_29-10_23_14', 'rosbag2_2025_07_29-13_17_18', 'rosbag2_2025_07_29-14_09_46', 'rosbag2_2025_10_15-12_02_00', 'rosbag2_2025_10_15-13_11_27', 'rosbag2_2025_10_15-14_02_29', 'rosbag2_2025_10_15-14_30_03', 'rosbag2_2025_10_16-09_27_48', 'rosbag2_2025_10_16-12_47_57'],
    - "Bulle12X": ['rosbag2_2025_10_20-10_13_32', 'rosbag2_2025_10_20-11_51_02', 'rosbag2_2025_10_20-16_02_46'],
    - "MainEntrance": ['rosbag2_2025_07_11-10_28_07', 'rosbag2_2025_07_11-11_18_00', 'rosbag2_2025_07_18-10_37_07', 'rosbag2_2025_07_25-10_52_32', 'rosbag2_2025_07_25-14_14_16', 'rosbag2_2025_10_09-08_52_20'],
    - "CoffeeB": ['rosbag2_2025_07_11-13_27_26', 'rosbag2_2025_07_11-14_54_55', 'rosbag2_2025_07_17-11_28_34', 'rosbag2_2025_07_17-12_52_12', 'rosbag2_2025_07_24-10_41_01', 'rosbag2_2025_07_24-12_14_56', 'rosbag2_2025_07_24-13_33_54', 'rosbag2_2025_07_24-14_33_36'],
    - "Room104": ['rosbag2_2025_07_07-11_16_10', 'rosbag2_2025_07_07-12_38_45', 'rosbag2_2025_07_07-15_33_32', 'rosbag2_2025_07_15-12_39_21', 'rosbag2_2025_07_15-13_41_01', 'rosbag2_2025_07_15-14_48_22', 'rosbag2_2025_07_21-10_22_11', 'rosbag2_2025_07_21-11_56_40', 'rosbag2_2025_07_21-13_09_22', 'rosbag2_2025_07_21-14_11_37', 'rosbag2_2025_07_21-15_15_07', 'rosbag2_2025_07_28-10_18_10', 'rosbag2_2025_07_28-11_25_33', 'rosbag2_2025_07_28-13_05_46', 'rosbag2_2025_07_28-14_19_07', 'rosbag2_2025_10_17-13_19_29', 'rosbag2_2025_10_17-14_28_15', 'rosbag2_2025_10_17-15_11_00', 'rosbag2_2025_10_17-16_47_09'],
    - "MainHallway": ['rosbag2_2025_10_09-10_23_38', 'rosbag2_2025_10_15-09_37_49', 'rosbag2_2025_10_15-11_03_05'],
    - "Cafeteria": ['rosbag2_2025_07_07-10_49_31', 'rosbag2_2025_07_22-09_38_18', 'rosbag2_2025_07_22-10_59_25', 'rosbag2_2025_07_22-12_18_30', 'rosbag2_2025_07_22-13_30_39'],
    - "EntranceCFacing": ['rosbag2_2025_10_07-15_03_48', 'rosbag2_2025_10_07-16_21_39', 'rosbag2_2025_10_09-17_37_23', 'rosbag2_2025_10_09-18_50_21', 'rosbag2_2025_10_15-12_27_14', 'rosbag2_2025_10_16-11_29_56'],
    - "AstorPlace": ["2022_09_21_astor_place_landfill","2022_09_21_astor_place_recycle","2022_09_26_astor_place_landfill","2022_09_26_astor_place_recycle","2022_09_28_astor_place_landfill","2022_09_28_astor_place_recycle","2022_10_06_astor_place_landfill","2022_10_06_astor_place_recycle","2022_10_12_astor_place_landfill_0","2022_10_12_astor_place_landfill_1","2022_10_12_astor_place_recycle_0","2022_10_12_astor_place_recycle_1"],
    - "AlbeeSquare": ["2023_07_06_albee_square_landfill_0","2023_07_06_albee_square_landfill_1","2023_07_06_albee_square_recycle_0","2023_07_06_albee_square_recycle_1","2023_07_07_albee_square_landfill_0","2023_07_07_albee_square_landfill_1","2023_07_07_albee_square_recycle_0","2023_07_07_albee_square_recycle_1","2023_07_11_albee_square_landfill_0","2023_07_11_albee_square_landfill_1","2023_07_11_albee_square_recycle_0","2023_07_11_albee_square_recycle_1","2023_07_12_albee_square_landfill_0","2023_07_12_albee_square_landfill_1","2023_07_12_albee_square_recycle_0","2023_07_12_albee_square_recycle_1","2023_07_14_albee_square_landfill","2023_07_14_albee_square_recycle"]
```

Validation/test sequences are in `EntranceCFacing`, `Room104` and `AlbeeSquare`

## ROS2 node topics

### Input

- `/rgb/color/image_raw/compressed`  
  **Type:** `sensor_msgs/msg/CompressedImage`  
  **Description:** Main RGB camera stream (compressed image).

- `/camera/aligned_depth_to_color/image_raw/compressed`  
  **Type:** `sensor_msgs/msg/CompressedImage`  
  **Description:** (Optional, if using depth) Depth image aligned to the color stream (compressed image).

- `/rgb/color/camera_info`  
  **Type:** `sensor_msgs/msg/CameraInfo`  
  **Description:** (Required when using depth) Camera intrinsics used to project the torso pixel and median torso depth into a 3D camera-frame position.

### Output

- `/huipred/overlay/compressed`  
  **Type:** `sensor_msgs/msg/CompressedImage`  
  **Description:** The camera image overlaid with YOLO pose, predicted interaction, and other visualizations according to the overlay mode.

- `/huipred/tracks`  
  **Type:** `std_msgs/msg/Float32MultiArray`  
  **Description:** Flat array of per-person tracking records for the current frame. Each detected person occupies **70 consecutive floats**. For person index `p` (0-based), use base offset `p * 70`.

  **Payload length:** `70 floats × number_of_tracks`

  **Offsets (relative to each person's base `p * 70`):**

  | Offset | Field | Description |
  |--------|-------|-------------|
  | 0 | `image_height` | Output image height in pixels |
  | 1 | `image_width` | Output image width in pixels |
  | 2 | `track_id` | YOLO/ByteTrack track identifier |
  | 3 | `x1` | Bounding box left (pixels) |
  | 4 | `y1` | Bounding box top (pixels) |
  | 5 | `x2` | Bounding box right (pixels) |
  | 6 | `y2` | Bounding box bottom (pixels) |
  | 7 | `box_confidence` | YOLO box confidence score |
  | 8 | `depth` | Median torso depth in meters from 10 depth samples (5 per torso diagonal). `-1.0` if depth enabled but unavailable, `-2.0` if depth disabled |
  | 9 | `torso_u` | Torso pixel u (median of valid sample locations). `-1.0` if unavailable |
  | 10 | `torso_v` | Torso pixel v (median of valid sample locations). `-1.0` if unavailable |
  | 11 | `torso_x` | 3D torso x in camera frame (meters). `-1.0` if unavailable |
  | 12 | `torso_y` | 3D torso y in camera frame (meters). `-1.0` if unavailable |
  | 13 | `torso_z` | 3D torso z in camera frame (meters). `-1.0` if unavailable |
  | 14 | `ip_output` | Latest interaction prediction for this track |
  | 15 | `ip_output_filtered` | Filtered/averaged interaction prediction |
  | 16 | `yolo_inference_time_s` | YOLO inference time for this frame (seconds) |
  | 17 | `ip_inference_time_s` | IP inference time for this frame (seconds) |
  | 18 | `ip_model_index` | Loaded IP checkpoint index, or `-1` if unknown |
  | 19–69 | `skeleton` | 17 COCO joints × 3 values each: `[x, y, conf]` normalized to `[0, 1]` for x/y. Joint `i` starts at offset `19 + i * 3` |

  **Example:** To read person 2's bounding box and depth from `data`: `base = 2 * 70`, then `x1 = data[base + 3]`, `depth = data[base + 8]`.

- `/huipred/tracks_detections2d`  
  **Type:** `vision_msgs/msg/Detection2DArray`  
  **Description:** One `Detection2D` per tracked person with valid left/right shoulder keypoints for the current frame, published alongside `/huipred/tracks` when estimation mode is not `none_based`. `header.stamp` matches the input RGB image timestamp. `header.frame_id` is the camera optical frame (`camera_info` when available, otherwise `camera_optical_frame`).

  **Validity:** A person is omitted when either shoulder keypoint is missing or below the keypoint confidence threshold.

  **Per-detection fields:**

  | Field | Description |
  |-------|-------------|
  | `id` | YOLO/ByteTrack track identifier (string) |
  | `bbox.center` | Bounding box center in pixels: `x = (x1+x2)/2`, `y = (y1+y2)/2` |
  | `bbox.center.theta` | Yaw in the image plane derived from the shoulder line (radians) |
  | `bbox.size_x`, `bbox.size_y` | Bounding box width and height in pixels |
  | `results[0].hypothesis.class_id` | `"person"` |
  | `results[0].hypothesis.score` | YOLO box confidence |
  | `results[0].pose.pose.position` | 3D torso position in the camera frame (meters), or shoulder midpoint fallback. `(-1, -1, -1)` if unavailable |
  | `results[0].pose.pose.orientation` | Yaw-only orientation (pitch=roll=0) from the shoulder line in the camera horizontal plane |

- `/huipred/torso_markers`  
  **Type:** `visualization_msgs/msg/MarkerArray`  
  **Description:** Oriented `CUBE` markers at each valid person pose (same shoulder validity rules as `/huipred/tracks_detections2d`). Cube dimensions are 0.6 m wide, 0.3 m deep, and 1.8 m high (local X/Y/Z), aligned with body yaw. Color encodes IP score (gray when unavailable, red→green for 0→1). Timestamps match the input RGB image.

- `/huipred/torso_poses`  
  **Type:** `geometry_msgs/msg/PoseArray`  
  **Description:** 3D torso positions for the current frame, in the camera optical frame (`header.frame_id` comes from `camera_info` when available, otherwise `camera_optical_frame`). Only **valid** projections are included (one `Pose` per person with a successful depth + camera_info projection). Orientation is identity (`w=1`); only `position.x/y/z` is meaningful.


## Remarks
For compatibily reasons the checkpoints containing config and `state_dict` must be separated using `convert_checkpoints.py` before running using the jetson docker.