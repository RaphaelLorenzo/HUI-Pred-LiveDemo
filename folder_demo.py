import time
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, VitPoseForPoseEstimation
import os

VITPOSE_KEYPOINTS_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", 
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
    "left_wrist", "right_wrist", "left_hip", "right_hip", 
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# COCO pose skeleton connections (17 keypoints format)
VITPOSE_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], 
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], 
    [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
]

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_track_color(track_id: int) -> tuple:
    """Generate a consistent color for a track ID using golden ratio."""
    hue = (track_id * 0.618033988749895) % 1.0
    rgb = cv2.cvtColor(np.array([[[int(hue * 180), 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


# Load models
detector = YOLO("checkpoints/yolo26n.pt")
if not os.path.exists("checkpoints/vitpose") or not os.path.exists("checkpoints/vitpose/config.json") or not os.path.exists("checkpoints/vitpose/model.safetensors") or not os.path.exists("checkpoints/vitpose/preprocessor_config.json"):
    print("Downloading vitpose model...")
    pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)
    pose_processor.save_pretrained("checkpoints/vitpose")
    pose_model.save_pretrained("checkpoints/vitpose")
    print("Vitpose model downloaded and saved to checkpoints/vitpose. Please run the script again.")
    exit()
else:
    print("Loading vitpose model from local cache...")
    pose_processor = AutoProcessor.from_pretrained("checkpoints/vitpose")
    pose_model = VitPoseForPoseEstimation.from_pretrained("checkpoints/vitpose", device_map=device)


def process_folder(folder_path: str, display: bool = False, kp_thresh: float = 0.3) -> dict:
    """Process all images in folder with detection, tracking and pose estimation."""
    track_history = {}
    image_paths = sorted(Path(folder_path).glob("*.[jpJP][pnPN][gG]*"))
    
    for frame_idx, image_path in enumerate(image_paths):
        t_frame_start = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        
        # Detection + tracking with ultralytics (persist=True maintains track IDs)
        t_detect_start = time.perf_counter()
        results = detector.track(image, persist=True, classes=[0], verbose=False)[0]
        t_detect = time.perf_counter() - t_detect_start
        
        # Skip frame if no detections
        if results.boxes.id is None:
            continue
        
        track_ids = results.boxes.id.int().cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
    
        # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        # Convert boxes to list format for ViTPose [[x1, y1, w, h], ...]
        person_boxes = boxes.tolist()
        
        # Pose preprocessing
        t_preprocess_start = time.perf_counter()
        inputs = pose_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)
        t_preprocess = time.perf_counter() - t_preprocess_start
        
        # Pose inference
        t_inference_start = time.perf_counter()
        with torch.no_grad():
            outputs = pose_model(**inputs)
        t_inference = time.perf_counter() - t_inference_start
        
        # Pose postprocessing
        t_postprocess_start = time.perf_counter()
        pose_results = pose_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])[0]
        t_postprocess = time.perf_counter() - t_postprocess_start
        
        # Update track history
        for i, track_id in enumerate(track_ids):
            track_id = int(track_id)
            if track_id not in track_history:
                track_history[track_id] = {"detections": [], "poses": []}
            
            track_history[track_id]["detections"].append({
                "frame": frame_idx,
                "bbox": boxes[i].tolist(),
                "conf": float(confs[i]),
            })
            track_history[track_id]["poses"].append({
                "frame": frame_idx,
                "keypoints": pose_results[i]["keypoints"].numpy().tolist(),
                "scores": pose_results[i]["scores"].numpy().tolist(),
            })

        # Display
        if display:
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            for i, track_id in enumerate(track_ids):
                color = get_track_color(int(track_id))
                x1, y1, w, h = boxes[i].astype(int)
                x2 = x1 + w
                y2 = y1 + h
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                keypoints = pose_results[i]["keypoints"].numpy()
                scores = pose_results[i]["scores"].numpy()
                # Draw skeleton
                for j, k in VITPOSE_SKELETON:
                    if scores[j] > kp_thresh and scores[k] > kp_thresh:
                        pt1 = tuple(keypoints[j].astype(int))
                        pt2 = tuple(keypoints[k].astype(int))
                        cv2.line(frame, pt1, pt2, color, 2)
                # Draw keypoints
                for kp, score in zip(keypoints, scores):
                    if score > kp_thresh:
                        cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1)
            cv2.imshow("Pose", frame)
            if cv2.waitKey(1) == ord("q"):
                break

        t_frame = time.perf_counter() - t_frame_start
        print(f"Frame {frame_idx}/{len(image_paths)} | {len(track_ids)} tracks | "
              f"detect: {t_detect*1000:.1f}ms, preproc: {t_preprocess*1000:.1f}ms, "
              f"infer: {t_inference*1000:.1f}ms, postproc: {t_postprocess*1000:.1f}ms, "
              f"total: {t_frame*1000:.1f}ms")

    if display:
        cv2.destroyAllWindows()
    return track_history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Path to folder containing images")
    parser.add_argument("--display", action="store_true", default=False, help="Display results with cv2")
    parser.add_argument("--kp_thresh", type=float, default=0.3, help="Keypoint confidence threshold")
    args = parser.parse_args()
    
    history = process_folder(args.folder, display=args.display, kp_thresh=args.kp_thresh)
    print(f"Processed {len(history)} unique tracks")
    for track_id, data in history.items():
        print(f"  Track {track_id}: {len(data['detections'])} frames")