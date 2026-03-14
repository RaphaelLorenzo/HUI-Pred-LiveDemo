#!/usr/bin/env python3
"""
Anonymize faces in a video using YOLO26x-pose.
Detects poses, blurs the face region (nose, eyes, ears) for each person, saves with _anonymized suffix.
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

# Import ultralytics before torch (project convention)
from ultralytics import YOLO


# COCO 17 keypoints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
FACE_KEYPOINT_INDICES = [0, 1, 2, 3, 4]


def face_bbox_from_keypoints(keypoints: np.ndarray, scores: np.ndarray, kp_thresh: float, padding: float):
    """
    Compute a face bounding box from COCO face keypoints (0-4).
    keypoints: (17, 2), scores: (17,)
    Returns (x1, y1, x2, y2) in int, or None if insufficient valid points.
    """
    pts = []
    for i in FACE_KEYPOINT_INDICES:
        if scores[i] > kp_thresh:
            pts.append(keypoints[i])
    if len(pts) < 2:
        return None
    pts = np.array(pts)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    w, h = x2 - x1, y2 - y1
    # Padding (expand box)
    pad_x = max(w * padding, 20)
    pad_y = max(h * padding, 20)
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = int(x2 + pad_x)
    y2 = int(y2 + pad_y)
    return (x1, y1, x2, y2)


def blur_face_roi(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, ksize: int = 31):
    """Blur the rectangle [x1,y1,x2,y2] in-place. ksize must be odd."""
    if ksize % 2 == 0:
        ksize += 1
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
    frame[y1:y2, x1:x2] = blurred


def anonymize_video(
    video_path: str,
    model_path: str = "checkpoints/yolo26x-pose.pt",
    kp_thresh: float = 0.3,
    face_padding: float = 0.5,
    blur_ksize: int = 31,
):
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    stem, ext = video_path.stem, video_path.suffix
    out_path = video_path.parent / f"{stem}_anonymized{ext}"

    pose_model = YOLO(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # YOLO expects BGR or RGB; we keep BGR for cv2
        results = pose_model(frame, verbose=False)[0]
        if results.keypoints is not None and results.keypoints.xy is not None:
            kp_xy = results.keypoints.xy.cpu().numpy()
            kp_conf = results.keypoints.conf.cpu().numpy()
            for i in range(len(kp_xy)):
                bbox = face_bbox_from_keypoints(
                    kp_xy[i], kp_conf[i], kp_thresh, face_padding
                )
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    x2 = min(x2, frame.shape[1])
                    y2 = min(y2, frame.shape[0])
                    blur_face_roi(frame, x1, y1, x2, y2, ksize=blur_ksize)
        out.write(frame)
        n += 1
        if total > 0 and n % 100 == 0:
            print(f"Processed {n}/{total} frames")

    cap.release()
    out.release()
    print(f"Saved: {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Anonymize faces in a video using YOLO26x-pose (blur face region)."
    )
    parser.add_argument(
        "--video", "-i",
        type=str,
        help="Path to input video",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="checkpoints/yolo26x-pose.pt",
        help="Path to YOLO pose model",
    )
    parser.add_argument(
        "--kp_thresh",
        type=float,
        default=0.3,
        help="Keypoint confidence threshold for face keypoints",
    )
    parser.add_argument(
        "--face_padding",
        type=float,
        default=0.5,
        help="Padding around face bbox as fraction of face size",
    )
    parser.add_argument(
        "--blur",
        type=int,
        default=31,
        help="Gaussian blur kernel size (odd)",
    )
    args = parser.parse_args()
    anonymize_video(
        args.video,
        model_path=args.model,
        kp_thresh=args.kp_thresh,
        face_padding=args.face_padding,
        blur_ksize=args.blur if args.blur % 2 else args.blur + 1,
    )


if __name__ == "__main__":
    main()
