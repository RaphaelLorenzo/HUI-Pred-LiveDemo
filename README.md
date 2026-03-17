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

- On jetson using `docker` (with a ROS2 node subscribing to )
```
docker build -t huipreddemo_jetson:latest .
sh run_docker.sh
python ros2_node_yolo26.py
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

- `/camera/color/image_raw/compressed`  
  **Type:** `sensor_msgs/msg/CompressedImage`  
  **Description:** Main RGB camera stream (compressed image).

- `/camera/aligned_depth_to_color/image_raw/compressed`  
  **Type:** `sensor_msgs/msg/CompressedImage`  
  **Description:** (Optional, if using depth) Depth image aligned to the color stream (compressed image).

### Output

- `/huilive_overlay/compressed`  
  **Type:** `sensor_msgs/msg/CompressedImage`  
  **Description:** The camera image overlaid with YOLO pose, predicted interaction, and other visualizations according to the overlay mode.

- `/huilive_tracks`  
  **Type:** `std_msgs/msg/Float32MultiArray`  
  **Description:** Sequence of per-person tracking information for each detection in the current frame:  
    `[image_height, image_width, x1, y1, x2, y2, box_confidence, depth, ip_output, ip_output_filtered]` (repeated for each track/person).  

  - `x1, y1, x2, y2`: Bounding box coordinates (float)  
  - `box_confidence`: YOLO box confidence score  
  - `depth`: Estimated depth (or -1.0 if unavailable)  
  - `ip_output`: Latest interaction prediction output for this track  
  - `ip_output_filtered`: Filtered/averaged interaction prediction for temporal stability


## Remarks
For compatibily reasons the checkpoints containing config and `state_dict` must be separated using `convert_checkpoints.py` before running using the jetson docker.