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


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)


# Load models
pose_model = YOLO("checkpoints/yolo26n-pose.pt")
seg_model = YOLO("checkpoints/yolo26n-seg.pt")


def process_folder(folder_path: str, display: bool = False, kp_thresh: float = 0.3) -> dict:
    """Process all images in folder with detection, tracking and pose estimation."""
    track_history = {}
    image_paths = sorted(Path(folder_path).glob("*.[jpJP][pnPN][gG]*"))
    
    for frame_idx, image_path in enumerate(image_paths):
        t_frame_start = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        
        # Detection + tracking + pose with YOLO pose model
        t_pose_start = time.perf_counter()
        pose_results = pose_model.track(image, persist=True, verbose=False)[0]
        t_pose = time.perf_counter() - t_pose_start
        
        # Skip frame if no detections
        if pose_results.boxes.id is None:
            continue
        
        track_ids = pose_results.boxes.id.int().cpu().numpy()
        boxes = pose_results.boxes.xyxy.cpu().numpy()
        confs = pose_results.boxes.conf.cpu().numpy()
        keypoints_all = pose_results.keypoints.xy.cpu().numpy()
        scores_all = pose_results.keypoints.conf.cpu().numpy()
        
        # Segmentation with YOLO seg model
        t_seg_start = time.perf_counter()
        seg_results = seg_model(image, classes=[0], verbose=False)[0]
        t_seg = time.perf_counter() - t_seg_start
        
        # Match seg masks to tracked boxes using IoU
        seg_boxes = seg_results.boxes.xyxy.cpu().numpy() if seg_results.boxes is not None else np.array([])
        seg_masks = seg_results.masks.xy if seg_results.masks is not None else []
        matched_masks = [None] * len(track_ids)
        for si, seg_box in enumerate(seg_boxes):
            best_iou, best_idx = 0, -1
            for pi, pose_box in enumerate(boxes):
                iou = box_iou(seg_box, pose_box)
                if iou > best_iou:
                    best_iou, best_idx = iou, pi
            if best_iou > 0.5 and best_idx >= 0:
                matched_masks[best_idx] = seg_masks[si]
        
        # Update track history
        for i, track_id in enumerate(track_ids):
            track_id = int(track_id)
            if track_id not in track_history:
                track_history[track_id] = {"detections": [], "poses": [], "masks": []}
            
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
            track_history[track_id]["masks"].append({
                "frame": frame_idx,
                "polygon": matched_masks[i].tolist() if matched_masks[i] is not None else None,
            })

        # Display
        if display:
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            overlay = frame.copy()
            for i, track_id in enumerate(track_ids):
                color = get_track_color(int(track_id))
                # Draw mask
                if matched_masks[i] is not None:
                    pts = matched_masks[i].astype(np.int32)
                    cv2.fillPoly(overlay, [pts], color)
                # Draw box
                x1, y1, x2, y2 = boxes[i].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # Draw skeleton
                keypoints = keypoints_all[i]
                scores = scores_all[i]
                for j, k in COCO_SKELETON:
                    if scores[j] > kp_thresh and scores[k] > kp_thresh:
                        pt1 = tuple(keypoints[j].astype(int))
                        pt2 = tuple(keypoints[k].astype(int))
                        cv2.line(frame, pt1, pt2, color, 2)
                # Draw keypoints
                for kp, score in zip(keypoints, scores):
                    if score > kp_thresh:
                        cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            cv2.imshow("Pose", frame)
            if cv2.waitKey(1) == ord("q"):
                break

        t_frame = time.perf_counter() - t_frame_start
        print(f"Frame {frame_idx}/{len(image_paths)} | {len(track_ids)} tracks | "
              f"pose: {t_pose*1000:.1f}ms, seg: {t_seg*1000:.1f}ms, total: {t_frame*1000:.1f}ms")

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
