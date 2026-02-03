import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import os

# COCO pose skeleton connections (17 keypoints format)
COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], 
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], 
    [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
]


def get_track_color(track_id: int) -> tuple:
    """Generate a consistent color for a track ID using golden ratio."""
    hue = (track_id * 0.618033988749895) % 1.0
    rgb = cv2.cvtColor(np.array([[[int(hue * 180), 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def estimate_torso_depth(keypoints: np.ndarray, scores: np.ndarray, depth_image: np.ndarray, kp_thresh: float) -> float | None:
    """
    Estimate depth at the center of the torso using diagonal sampling.
    
    Uses 6 points: 3 sampled at 0.25, 0.5, 0.75 on diagonal left_shoulder->right_hip
    and 3 on diagonal right_shoulder->left_hip. Takes median of valid samples.
    A diagonal is only considered if both its keypoints are valid (above threshold).
    
    COCO keypoint indices:
        5: left_shoulder, 6: right_shoulder, 11: left_hip, 12: right_hip
    
    Returns:
        Median depth value in the depth image units, or None if no valid samples.
    """
    # Keypoint indices
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
    LEFT_HIP, RIGHT_HIP = 11, 12
    
    sample_ratios = [0.25, 0.5, 0.75]
    depth_samples = []
    h, w = depth_image.shape[:2]
    
    # Diagonal 1: left_shoulder (5) -> right_hip (12)
    if scores[LEFT_SHOULDER] > kp_thresh and scores[RIGHT_HIP] > kp_thresh:
        pt1 = keypoints[LEFT_SHOULDER]
        pt2 = keypoints[RIGHT_HIP]
        for ratio in sample_ratios:
            x = int(pt1[0] + ratio * (pt2[0] - pt1[0]))
            y = int(pt1[1] + ratio * (pt2[1] - pt1[1]))
            if 0 <= x < w and 0 <= y < h:
                depth_samples.append(depth_image[y, x])
    
    # Diagonal 2: right_shoulder (6) -> left_hip (11)
    if scores[RIGHT_SHOULDER] > kp_thresh and scores[LEFT_HIP] > kp_thresh:
        pt1 = keypoints[RIGHT_SHOULDER]
        pt2 = keypoints[LEFT_HIP]
        for ratio in sample_ratios:
            x = int(pt1[0] + ratio * (pt2[0] - pt1[0]))
            y = int(pt1[1] + ratio * (pt2[1] - pt1[1]))
            if 0 <= x < w and 0 <= y < h:
                depth_samples.append(depth_image[y, x])
    
    if len(depth_samples) == 0:
        return -1000.0
    
    return float(np.median(depth_samples))


# Load model
pose_model = YOLO("checkpoints/yolo26n-pose.pt")


def process_folder(args: argparse.Namespace) -> dict:
    """Process all images in folder with detection, tracking and pose estimation."""
    track_history = {}
    image_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(args.directory)) for f in fn if f.endswith(".jpg") or f.endswith(".png")]
    image_paths.sort()
    
    depth_paths = None
    if args.depth is not None:
        depth_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(args.depth)) for f in fn if f.endswith(".jpg") or f.endswith(".png")]
        depth_paths.sort()
        assert len(image_paths) == len(depth_paths), "Number of images and depth images must be the same"
        
    for frame_idx, image_path in enumerate(image_paths):
        t_frame_start = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        
        # Load depth image if provided
        depth_image = None
        if depth_paths is not None:
            depth_image = cv2.imread(depth_paths[frame_idx], cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                print(f"Warning: Could not load depth image {depth_paths[frame_idx]}")
        
        # Detection + tracking + pose with YOLO pose model
        t_infer_start = time.perf_counter()
        results = pose_model.track(image, persist=True, verbose=False)[0]
        t_infer = time.perf_counter() - t_infer_start
        
        # Skip frame if no detections
        if results.boxes.id is None:
            continue
        
        track_ids = results.boxes.id.int().cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        keypoints_all = results.keypoints.xy.cpu().numpy()
        scores_all = results.keypoints.conf.cpu().numpy()
        
        # Compute depth for each person if depth image is available
        depths = []
        if depth_image is not None:
            for i in range(len(track_ids)):
                depth_val = estimate_torso_depth(
                    keypoints_all[i], scores_all[i], depth_image, args.kp_thresh
                ) / 1000.0 # in m
                depths.append(depth_val)
        else:
            depths = [None] * len(track_ids)
        
        # Update track history
        for i, track_id in enumerate(track_ids):
            track_id = int(track_id)
            if track_id not in track_history:
                track_history[track_id] = {"detections": [], "poses": []}
            
            detection_data = {
                "frame": frame_idx,
                "bbox": boxes[i].tolist(),
                "conf": float(confs[i]),
            }
            if depths[i] is not None:
                detection_data["depth"] = depths[i]
            
            track_history[track_id]["detections"].append(detection_data)
            track_history[track_id]["poses"].append({
                "frame": frame_idx,
                "keypoints": keypoints_all[i].tolist(),
                "scores": scores_all[i].tolist(),
            })

        # Display
        if args.display:
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            for i, track_id in enumerate(track_ids):
                color = get_track_color(int(track_id))
                x1, y1, x2, y2 = boxes[i].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Build label with ID and depth (if available)
                label = f"ID:{track_id}"
                if depths[i] is not None:
                    label += f" D:{depths[i]:.1f}"
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                keypoints = keypoints_all[i]
                scores = scores_all[i]
                # Draw skeleton
                for j, k in COCO_SKELETON:
                    if scores[j] > args.kp_thresh and scores[k] > args.kp_thresh:
                        pt1 = tuple(keypoints[j].astype(int))
                        pt2 = tuple(keypoints[k].astype(int))
                        cv2.line(frame, pt1, pt2, color, 2)
                # Draw keypoints
                for kp, score in zip(keypoints, scores):
                    if score > args.kp_thresh:
                        cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1)
            cv2.imshow("Pose", frame)
            if cv2.waitKey(1) == ord("q"):
                break

        t_frame = time.perf_counter() - t_frame_start
        print(f"Frame {frame_idx}/{len(image_paths)} | {len(track_ids)} tracks | "
              f"infer: {t_infer*1000:.1f}ms, total: {t_frame*1000:.1f}ms")

    if args.display:
        cv2.destroyAllWindows()
    return track_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", "-d", type=str, help="Path to folder containing images")
    parser.add_argument("--depth", type=str, default=None, help="Path to folder containing depth images (optional)")
    parser.add_argument("--display", action="store_true", default=False, help="Display results with cv2")
    parser.add_argument("--kp_thresh", type=float, default=0.3, help="Keypoint confidence threshold")
    args = parser.parse_args()
    
    history = process_folder(args)
    print(f"Processed {len(history)} unique tracks")
    for track_id, data in history.items():
        print(f"  Track {track_id}: {len(data['detections'])} frames")
