import time
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

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


# Load model
pose_model = YOLO("checkpoints/yolo26n-pose.pt")


def process_folder(folder_path: str, display: bool = False, kp_thresh: float = 0.3) -> dict:
    """Process all images in folder with detection, tracking and pose estimation."""
    track_history = {}
    image_paths = sorted(Path(folder_path).glob("*.[jpJP][pnPN][gG]*"))
    
    for frame_idx, image_path in enumerate(image_paths):
        t_frame_start = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        
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
                "keypoints": keypoints_all[i].tolist(),
                "scores": scores_all[i].tolist(),
            })

        # Display
        if display:
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            for i, track_id in enumerate(track_ids):
                color = get_track_color(int(track_id))
                x1, y1, x2, y2 = boxes[i].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                keypoints = keypoints_all[i]
                scores = scores_all[i]
                # Draw skeleton
                for j, k in COCO_SKELETON:
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
              f"infer: {t_infer*1000:.1f}ms, total: {t_frame*1000:.1f}ms")

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
