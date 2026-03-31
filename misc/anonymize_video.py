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


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    denom = area1 + area2 - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


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


def blur_roi(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, ksize: int):
    """Blur the rectangle [x1,y1,x2,y2] in-place. ksize must be odd."""
    if ksize % 2 == 0:
        ksize += 1
    h, w = frame.shape[:2]
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
    frame[y1:y2, x1:x2] = blurred


def blur_polygon_mask(frame: np.ndarray, polygon_xy: np.ndarray, ksize: int):
    """Blur pixels inside polygon mask (in-place). polygon_xy is Nx2 float/ints in image coords."""
    if polygon_xy is None:
        return
    pts = np.asarray(polygon_xy, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        return
    if ksize % 2 == 0:
        ksize += 1
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    frame[mask == 255] = blurred[mask == 255]


def shoulder_y_from_keypoints(keypoints: np.ndarray, scores: np.ndarray, kp_thresh: float):
    """Return y (float) of the shoulder line if available, else None."""
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
    ys = []
    if scores[LEFT_SHOULDER] > kp_thresh:
        ys.append(float(keypoints[LEFT_SHOULDER][1]))
    if scores[RIGHT_SHOULDER] > kp_thresh:
        ys.append(float(keypoints[RIGHT_SHOULDER][1]))
    if not ys:
        return None
    return float(min(ys))


def anonymize_video(
    video_path: str,
    model_path: str = "checkpoints/yolo26x-pose.pt",
    seg_model_path: str | None = "checkpoints/yolo26n-seg.pt",
    conf_thresh: float = 0.3,
    kp_thresh: float = 0.3,
    face_padding: float = 0.5,
    face_blur_ksize: int = 31,
    head_blur_ksize: int = 51,
    body_blur_ksize: int = 31,
    shoulder_fallback_ratio: float = 0.30,
    skip: int = 1,
):
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    stem, ext = video_path.stem, video_path.suffix
    out_path = video_path.parent / f"{stem}_anonymized{ext}"

    pose_model = YOLO(model_path)
    seg_model = YOLO(seg_model_path) if seg_model_path else None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    skip = max(1, int(skip))
    fps_out = max(1e-6, float(fps_in) / float(skip))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps_out, (w, h))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = 0
    frame_idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_idx % skip != 0:
            frame_idx += 1
            continue
        ret, frame = cap.retrieve()
        frame_idx += 1
        if not ret:
            break
        # YOLO expects BGR or RGB; we keep BGR for cv2
        results = pose_model(frame, conf=conf_thresh, verbose=False)[0]

        kp_xy = None
        kp_conf = None
        pose_boxes = None
        if results.keypoints is not None and results.keypoints.xy is not None:
            kp_xy = results.keypoints.xy.cpu().numpy()
            kp_conf = results.keypoints.conf.cpu().numpy()
        if results.boxes is not None and results.boxes.xyxy is not None:
            pose_boxes = results.boxes.xyxy.cpu().numpy()

        # Full-body blur using segmentation masks (kernel size 31)
        if seg_model is not None and pose_boxes is not None and len(pose_boxes) > 0:
            seg_results = seg_model(frame, classes=[0], conf=conf_thresh, verbose=False)[0]
            seg_boxes = (
                seg_results.boxes.xyxy.cpu().numpy()
                if seg_results.boxes is not None and seg_results.boxes.xyxy is not None
                else np.array([])
            )
            seg_polys = seg_results.masks.xy if seg_results.masks is not None else []
            if len(seg_boxes) > 0 and len(seg_polys) > 0:
                matched_polys = [None] * len(pose_boxes)
                for si, seg_box in enumerate(seg_boxes):
                    best_iou, best_pi = 0.0, -1
                    for pi, pose_box in enumerate(pose_boxes):
                        iou = box_iou(seg_box, pose_box)
                        if iou > best_iou:
                            best_iou, best_pi = iou, pi
                    if best_iou > 0.5 and best_pi >= 0:
                        matched_polys[best_pi] = seg_polys[si]
                for poly in matched_polys:
                    if poly is not None:
                        blur_polygon_mask(frame, poly, ksize=body_blur_ksize)

        # Stronger head/upper anonymization: blur from shoulder line to top of the person box (kernel size 51)
        if pose_boxes is not None and len(pose_boxes) > 0:
            for i, box in enumerate(pose_boxes):
                x1, y1, x2, y2 = box.tolist()
                y_shoulder = None
                if kp_xy is not None and kp_conf is not None and i < len(kp_xy):
                    y_shoulder = shoulder_y_from_keypoints(kp_xy[i], kp_conf[i], kp_thresh)
                if y_shoulder is None:
                    y_shoulder = y1 + shoulder_fallback_ratio * (y2 - y1)
                blur_roi(frame, x1, y1, x2, y_shoulder, ksize=head_blur_ksize)

        # Face blur (kept, but now in addition to above)
        if kp_xy is not None and kp_conf is not None:
            for i in range(len(kp_xy)):
                bbox = face_bbox_from_keypoints(kp_xy[i], kp_conf[i], kp_thresh, face_padding)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    blur_roi(frame, x1, y1, x2, y2, ksize=face_blur_ksize)
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
        description="Anonymize people in a video using YOLO26 pose + optional YOLO26 segmentation."
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
        "--seg_model",
        type=str,
        default="checkpoints/yolo26n-seg.pt",
        help="Path to YOLO segmentation model (set empty to disable)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold for pose + segmentation detections",
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
        "--face_blur",
        type=int,
        default=81,
        help="Gaussian blur kernel size for face (odd)",
    )
    parser.add_argument(
        "--head_blur",
        type=int,
        default=81,
        help="Gaussian blur kernel size for shoulder->top region (odd)",
    )
    parser.add_argument(
        "--body_blur",
        type=int,
        default=51,
        help="Gaussian blur kernel size for segmentation body mask (odd)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Write 1 out of every N frames (output FPS will be divided by N)",
    )
    args = parser.parse_args()
    anonymize_video(
        args.video,
        model_path=args.model,
        seg_model_path=args.seg_model if args.seg_model else None,
        conf_thresh=args.conf,
        kp_thresh=args.kp_thresh,
        face_padding=args.face_padding,
        face_blur_ksize=args.face_blur if args.face_blur % 2 else args.face_blur + 1,
        head_blur_ksize=args.head_blur if args.head_blur % 2 else args.head_blur + 1,
        body_blur_ksize=args.body_blur if args.body_blur % 2 else args.body_blur + 1,
        skip=args.skip,
    )


if __name__ == "__main__":
    main()
