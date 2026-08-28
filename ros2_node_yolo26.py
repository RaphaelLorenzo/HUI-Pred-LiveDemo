#!/usr/bin/env python3.10
"""
ROS2 node for real-time human-robot interaction prediction using YOLO pose estimation.

Subscribes to CompressedImage topics for RGB and depth, runs YOLO pose detection
+ tracking, and publishes an annotated overlay image with bounding boxes,
track IDs, skeleton, and predicted interaction scores on /huipred/overlay/compressed.

In Docker (ROS Humble uses Python 3.10), run with:
    python3.10 ros2_node_yolo26.py --rgb_topic /camera/color/image_raw/compressed --depth_topic /camera/depth/image_raw/compressed --interaction_prediction_checkpoint checkpoints/mb_FineTuned_28_02_26_best_ap.pth
"""

import argparse
import threading
import types
import time
import cv2
import os
from PIL import Image
from functools import partial

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image as RosImage
from std_msgs.msg import Float32MultiArray, String
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray
import message_filters

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None


# WARNING : this is very weird : but if ultralytics is imported after torch, the YOLO model does not work (no detections !)
from ultralytics import YOLO
# pose_model = YOLO("checkpoints/yolo26x-pose.pt")
# results_test = pose_model("frame_000146.jpg", verbose=True)[0]
# print(f"Results test: {results_test.boxes.id}")
# exit()

# IMPORTANT : import torch after ultralytics !
import torch
import torch.nn as nn

from utils.geometry_utils import rot_matrix_torch
from utils.visualization import get_track_color

from predictors.mlp import MLPInteractionPredictor
from predictors.lstm import LSTMInteractionPredictor
from predictors.MotionBERT.lib.model.DSTformer import DSTformer
from predictors.MotionBERT.lib.model.model_action import ActionNet
from predictors.STG_NF.model_pose import STG_NF
from predictors.STGCN.net.st_gcn import Model as STGCN
from predictors.SkateFormer.model.SkateFormer import SkateFormer

import sys
import numpy as np
from utils.other_utils import read_yaml_to_dic

# ---------------------------------------------------------------------------
# Behavior prototype thresholds
# ---------------------------------------------------------------------------
INITIAL_INTERACTION_PREDICTION_THRESHOLD = 0.2
INITIAL_MIN_FRAMES_ENGAGED = 2
BREAKUP_FRAMES_DISENGAGED = 10
MIN_EYES_SCALE_THRESHOLD = 0.5
MAX_EYES_SCALE_THRESHOLD = 0.99

IP_MODELS_NAME_TO_INDEX = {
    "converted_mb_FineTuned_28_02_26_best_ap": 0,
    "converted_mb_FineTuned_SingleImage_05_03_26_best_adaptative_f1": 1,
    "converted_mb_FineTuned_SingleImage_NoLegs_19_03_26_best_adaptative_f1": 2,
    "converted_mb_FineTuned_SingleImage_ReprojectRecenter_19_03_26_best_adaptative_f1": 3,
    "converted_mb_FineTuned_SingleImageV2_05_03_26_best_adaptative_f1": 4,
    "converted_mb_FineTuned_Sub3_05_03_26_best_adaptative_f1": 5,
    "converted_mb_FineTuned_Sub3_Randomized_05_03_26_best_adaptative_f1": 6,
    "converted_mb_FineTuned_Sub3_Randomized_05_03_26_best_ap": 7,
    "converted_mb_lite_FineTuned_SingleImage_05_03_26_best_adaptative_f1": 8,
}

COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
    [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
]

PERSON_CUBE_DEPTH_M = 0.3
PERSON_CUBE_WIDTH_M = 0.6
PERSON_CUBE_HEIGHT_M = 1.8
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

METADATA_CKPT_COLUMNS = [
    "recording", "episode", "image_height", "image_width",
    "unique_track_identifier", "track_id", "image_file", "image_index",
    "validity", "current_segment", "total_segments", "position_in_segment",
    "length_of_current_segment", "timestamp", "timestamp_sec",
    "timestamp_track", "engagement", "time_to_first_interaction", "mask_rle",
]

# ---------------------------------------------------------------------------
# Helper functions (ported from folder_demo_yolo26.py)
# ---------------------------------------------------------------------------

def estimate_torso_depth_and_center(
    keypoints: np.ndarray,
    scores: np.ndarray,
    depth_image: np.ndarray,
    kp_thresh: float,
    logger,
):
    """
    Estimate torso center pixel and depth from diagonal depth sampling.

    Samples 5 points on each torso diagonal, keeps valid depth values only, and
    returns the median depth together with the median sampled pixel location.
    """
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
    LEFT_HIP, RIGHT_HIP = 11, 12
    sample_ratios = np.linspace(1.0 / 6.0, 5.0 / 6.0, 5)
    depth_samples = []
    sample_points = []
    h, w = depth_image.shape[:2]

    if scores[LEFT_SHOULDER] > kp_thresh and scores[RIGHT_HIP] > kp_thresh:
        pt1, pt2 = keypoints[LEFT_SHOULDER], keypoints[RIGHT_HIP]
        for r in sample_ratios:
            x, y = int(pt1[0] + r * (pt2[0] - pt1[0])), int(pt1[1] + r * (pt2[1] - pt1[1]))
            if 0 <= x < w and 0 <= y < h:
                depth_value = float(depth_image[y, x])
                if np.isfinite(depth_value) and depth_value > 0:
                    depth_samples.append(depth_value)
                    sample_points.append((float(x), float(y)))
    else:
        logger.info(f"Left shoulder score: {scores[LEFT_SHOULDER]} | Right hip score: {scores[RIGHT_HIP]} | Insufficient")
                

    if scores[RIGHT_SHOULDER] > kp_thresh and scores[LEFT_HIP] > kp_thresh:
        pt1, pt2 = keypoints[RIGHT_SHOULDER], keypoints[LEFT_HIP]
        for r in sample_ratios:
            x, y = int(pt1[0] + r * (pt2[0] - pt1[0])), int(pt1[1] + r * (pt2[1] - pt1[1]))
            if 0 <= x < w and 0 <= y < h:
                depth_value = float(depth_image[y, x])
                if np.isfinite(depth_value) and depth_value > 0:
                    depth_samples.append(depth_value)
                    sample_points.append((float(x), float(y)))
    else:
        logger.info(f"Right shoulder score: {scores[RIGHT_SHOULDER]} | Left hip score: {scores[LEFT_HIP]} | Insufficient")

    if len(depth_samples) < 3:
        logger.warn("Not enough depth samples to estimate torso center and depth")
        return None, None

    sample_points_array = np.array(sample_points, dtype=np.float32)
    torso_center_uv = np.median(sample_points_array, axis=0)
    torso_depth = float(np.median(np.array(depth_samples, dtype=np.float32)))
    return torso_depth, torso_center_uv


def project_pixel_to_3d(
    pixel_xy: np.ndarray,
    depth_m: float,
    camera_matrix: np.ndarray,
):
    """Project one image pixel with depth into camera-frame 3D coordinates."""
    if pixel_xy is None or depth_m is None or depth_m <= 0 or camera_matrix is None:
        return None

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    if fx == 0.0 or fy == 0.0:
        return None

    u = float(pixel_xy[0])
    v = float(pixel_xy[1])
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float32)


def _sample_depth_m_at_pixel(
    pixel_uv: np.ndarray,
    depth_image: np.ndarray | None,
    depth_scale: float,
    fallback_depth_m: float | None,
) -> float | None:
    """Return depth in meters at a pixel, or fallback depth when unavailable."""
    if depth_image is not None:
        h, w = depth_image.shape[:2]
        x, y = int(pixel_uv[0]), int(pixel_uv[1])
        if 0 <= x < w and 0 <= y < h:
            depth_raw = float(depth_image[y, x])
            if np.isfinite(depth_raw) and depth_raw > 0:
                return depth_raw / depth_scale
    return fallback_depth_m


def _quaternion_from_rotation_matrix(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) from a 3x3 rotation matrix (local-to-parent)."""
    m = np.asarray(rot, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return (
            float((m[2, 1] - m[1, 2]) / s),
            float((m[0, 2] - m[2, 0]) / s),
            float((m[1, 0] - m[0, 1]) / s),
            float(0.25 * s),
        )
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return (
            float(0.25 * s),
            float((m[0, 1] + m[1, 0]) / s),
            float((m[0, 2] + m[2, 0]) / s),
            float((m[2, 1] - m[1, 2]) / s),
        )
    if m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return (
            float((m[0, 1] + m[1, 0]) / s),
            float(0.25 * s),
            float((m[1, 2] + m[2, 1]) / s),
            float((m[0, 2] - m[2, 0]) / s),
        )
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return (
        float((m[0, 2] + m[2, 0]) / s),
        float((m[1, 2] + m[2, 1]) / s),
        float(0.25 * s),
        float((m[1, 0] - m[0, 1]) / s),
    )


def _body_orientation_from_shoulders(
    left_3d: np.ndarray | None,
    right_3d: np.ndarray | None,
    shoulder_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return orthonormal width, forward, and up axes for a yaw-only body frame."""
    up_cam = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    if left_3d is not None and right_3d is not None:
        width_dir = left_3d - right_3d
        width_dir[1] = 0.0
        width_norm = float(np.linalg.norm(width_dir))
        if width_norm < 1e-6:
            return None
        width_dir /= width_norm
        forward_dir = np.cross(width_dir, up_cam)
    else:
        facing_2d = np.array([-shoulder_2d[1], shoulder_2d[0]], dtype=np.float32)
        forward_dir = np.array([facing_2d[0], 0.0, facing_2d[1]], dtype=np.float32)
        forward_norm = float(np.linalg.norm(forward_dir))
        if forward_norm < 1e-6:
            return None
        forward_dir /= forward_norm
        width_dir = np.cross(forward_dir, up_cam)

    width_norm = float(np.linalg.norm(width_dir))
    if width_norm < 1e-6:
        return None
    width_dir /= width_norm
    forward_dir[1] = 0.0
    forward_norm = float(np.linalg.norm(forward_dir))
    if forward_norm < 1e-6:
        return None
    forward_dir /= forward_norm
    return width_dir, forward_dir, up_cam


def _quaternion_from_body_axes(
    width_dir: np.ndarray,
    forward_dir: np.ndarray,
    up_dir: np.ndarray,
) -> tuple[float, float, float, float]:
    """Quaternion aligning local X=width, Y=depth/forward, Z=height/up."""
    rot = np.stack(
        [
            np.asarray(width_dir, dtype=np.float32),
            np.asarray(forward_dir, dtype=np.float32),
            np.asarray(up_dir, dtype=np.float32),
        ],
        axis=1,
    )
    return _quaternion_from_rotation_matrix(rot)


def compute_person_pose_from_shoulders(
    keypoints: np.ndarray,
    scores: np.ndarray,
    depth_image: np.ndarray | None,
    camera_matrix: np.ndarray | None,
    depth_scale: float,
    kp_thresh: float,
    torso_center_uv: np.ndarray | None,
    torso_depth_m: float | None,
) -> dict | None:
    """Estimate yaw-only person pose from both shoulders (pitch=roll=0).

    Returns None when either shoulder is missing/unconfident.
    """
    if scores[LEFT_SHOULDER] <= kp_thresh or scores[RIGHT_SHOULDER] <= kp_thresh:
        return None

    left_uv = np.asarray(keypoints[LEFT_SHOULDER], dtype=np.float32)
    right_uv = np.asarray(keypoints[RIGHT_SHOULDER], dtype=np.float32)
    shoulder_2d = right_uv - left_uv
    if float(np.linalg.norm(shoulder_2d)) < 1e-6:
        return None

    left_depth_m = _sample_depth_m_at_pixel(left_uv, depth_image, depth_scale, torso_depth_m)
    right_depth_m = _sample_depth_m_at_pixel(right_uv, depth_image, depth_scale, torso_depth_m)
    left_3d = (
        project_pixel_to_3d(left_uv, left_depth_m, camera_matrix)
        if left_depth_m is not None and camera_matrix is not None
        else None
    )
    right_3d = (
        project_pixel_to_3d(right_uv, right_depth_m, camera_matrix)
        if right_depth_m is not None and camera_matrix is not None
        else None
    )

    axes = _body_orientation_from_shoulders(left_3d, right_3d, shoulder_2d)
    if axes is None:
        return None
    width_dir, forward_dir, up_dir = axes

    position = None
    if torso_center_uv is not None and torso_depth_m is not None and camera_matrix is not None:
        position = project_pixel_to_3d(torso_center_uv, torso_depth_m, camera_matrix)
    elif left_3d is not None and right_3d is not None:
        position = 0.5 * (left_3d + right_3d)

    quat = _quaternion_from_body_axes(width_dir, forward_dir, up_dir)
    bbox_theta = float(np.arctan2(-shoulder_2d[1], shoulder_2d[0]) + (np.pi / 2.0))
    return {
        "position": position,
        "orientation": quat,
        "bbox_theta": bbox_theta,
    }


def load_model_from_config(config: dict, device: torch.device):
    include_columns = config["include_columns"]
    data_columns = [c for c in include_columns if c not in METADATA_CKPT_COLUMNS]
    input_dim = len(data_columns)

    if "select_input_range" in config:
        assert config["select_input_range"] == [0, -1], "select_input_range must be [0, -1] for live interaction prediction"
    
    sequence_length = config["input_length_in_frames"] // config["subsample_frames"]
    
    print(f"Input length in frames: {config['input_length_in_frames']} | Sequence length: {sequence_length} | Subsampling frames: {config['subsample_frames']}")

    # Subsampling is handled by running the pipeline at target_fps = source_fps / subsample_frames
    # (frame dropping in the ROS node); no need to restrict model loading here.

    mt = config["force_model_type"]

    if mt == "mlp":
        model = MLPInteractionPredictor(
            input_dim=input_dim, sequence_length=sequence_length,
            hidden_dims=config["hidden_dims"], dropout=config["dropout"],
        ).to(device)
    elif mt == "lstm":
        model = LSTMInteractionPredictor(
            input_dim=input_dim, sequence_length=sequence_length,
            hidden_dim=config["lstm_hidden_dim"], num_layers=config["lstm_num_layers"],
            dropout=config["lstm_dropout"], bidirectional=False,
        ).to(device)
    elif mt.lower().startswith("motionbert") or mt.lower().startswith("mb"):
        lite = "_lite" in mt.lower()
        backbone = DSTformer(
            dim_in=3, dim_out=3, dim_feat=256 if lite else 512, dim_rep=512,
            depth=5, num_heads=8, mlp_ratio=4 if lite else 2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), maxlen=243,
            num_joints=17, desired_return="representation",
        )
        model = ActionNet(
            backbone=backbone, dim_rep=512, num_classes=1,
            dropout_ratio=config["mb_head_dropout"],
            version="class", hidden_dim=config["mb_head_hidden_dim"], num_joints=17,
        ).to(device)
    elif mt == "stg_nf":
        model = STG_NF(
            device=device, pose_shape=(2, sequence_length, 18),
            hidden_channels=config["stg_nf_hidden_channels"],
            K=config["stg_nf_K"], L=config["stg_nf_L"], R=config["stg_nf_R"],
            actnorm_scale=config["stg_nf_actnorm_scale"],
            flow_permutation="permute", flow_coupling="affine",
            LU_decomposed=True, learn_top=False,
            edge_importance=config["stg_nf_edge_importance"],
            temporal_kernel_size=None, strategy="uniform",
            max_hops=config["stg_nf_max_hops"],
        ).to(device)
    elif mt == "stgcn":
        model = STGCN(
            in_channels=config["stgcn_in_channels"], num_class=1,
            graph_args={"layout": config["stgcn_layout"], "strategy": "spatial"},
            edge_importance_weighting=config["stgcn_edge_importance_weighting"],
        ).to(device)
    elif mt == "skateformer":
        assert sequence_length % 8 == 0
        Tdim = sequence_length // 8
        ncols = len(config["include_columns"])
        if ncols == 74:
            num_j, tss = 20, [(Tdim, 4), (Tdim, 5), (Tdim, 4), (Tdim, 5)]
        elif ncols in (263, 212):
            num_j, tss = 24, [(Tdim, 8), (Tdim, 12), (Tdim, 8), (Tdim, 12)]
        else:
            raise ValueError(f"Invalid input columns count: {ncols}")
        model = SkateFormer(
            in_channels=config["skateformer_in_channels"],
            depths=(2, 2, 2, 2), channels=(96, 192, 192, 192),
            num_classes=1, embed_dim=96, num_people=1, num_points=num_j,
            kernel_size=7, num_heads=32, attn_drop=0.5, head_drop=0.0,
            rel=True, drop_path=0.2,
            type_1_size=tss[0], type_2_size=tss[1],
            type_3_size=tss[2], type_4_size=tss[3],
            mlp_ratio=1.0, index_t=True,
        ).to(device)
    else:
        raise ValueError(f"Invalid model type: {mt}")

    model.eval()
    return model


def coco2h36m(x: torch.Tensor):
    """Convert COCO 17-keypoint format to H36M 17-keypoint format.  x: (B, T, 17, C)"""
    y = torch.zeros(x.shape, device=x.device)
    y[:, :, 0, :] = (x[:, :, 11, :] + x[:, :, 12, :]) * 0.5
    y[:, :, 1, :] = x[:, :, 12, :]
    y[:, :, 2, :] = x[:, :, 14, :]
    y[:, :, 3, :] = x[:, :, 16, :]
    y[:, :, 4, :] = x[:, :, 11, :]
    y[:, :, 5, :] = x[:, :, 13, :]
    y[:, :, 6, :] = x[:, :, 15, :]
    y[:, :, 8, :] = (x[:, :, 5, :] + x[:, :, 6, :]) * 0.5
    y[:, :, 7, :] = (y[:, :, 0, :] + y[:, :, 8, :]) * 0.5
    y[:, :, 9, :] = x[:, :, 0, :]
    y[:, :, 10, :] = (x[:, :, 1, :] + x[:, :, 2, :]) * 0.5
    y[:, :, 11, :] = x[:, :, 5, :]
    y[:, :, 12, :] = x[:, :, 7, :]
    y[:, :, 13, :] = x[:, :, 9, :]
    y[:, :, 14, :] = x[:, :, 6, :]
    y[:, :, 15, :] = x[:, :, 8, :]
    y[:, :, 16, :] = x[:, :, 10, :]
    return y


def backproject_points_to_equirect(
    points: torch.Tensor,
    persp_size: tuple,
    eq_size: tuple = (1920, 3840),
    fov: float = 70.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    device: torch.device = None,
):
    """Backproject 2D perspective points to equirectangular coordinates."""
    if device is None:
        device = points.device
    out_w, out_h = persp_size
    eq_h, eq_w = eq_size
    fov_rad = torch.deg2rad(torch.tensor(fov, dtype=torch.float32, device=device))
    aspect = out_h / out_w
    fov_y = 2 * torch.arctan(torch.tan(fov_rad / 2) * aspect)
    original_shape = points.shape[:-1]
    points_flat = points.reshape(-1, 2)
    tan_fov_x = torch.tan(fov_rad / 2)
    tan_fov_y = torch.tan(fov_y / 2)
    x_norm = (points_flat[:, 0] / out_w - 0.5) * 2 * tan_fov_x
    y_norm = -(points_flat[:, 1] / out_h - 0.5) * 2 * tan_fov_y
    z = torch.ones_like(x_norm)
    directions = torch.stack([x_norm, y_norm, z], dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    R = rot_matrix_torch(yaw, pitch, roll, device)
    dirs_rot = directions @ R.T
    lon = torch.atan2(dirs_rot[:, 0], dirs_rot[:, 2])
    lat = torch.asin(torch.clamp(dirs_rot[:, 1], -1.0, 1.0))
    eq_x = (lon / (2 * torch.pi) + 0.5) * eq_w
    eq_y = (0.5 - lat / torch.pi) * eq_h
    eq_points = torch.stack([eq_x, eq_y], dim=-1)
    return eq_points.reshape(*original_shape, 2)


def action_net_inference(
    args, model: ActionNet, config: dict,
    current_tracks_history: dict, device: torch.device,
    image_size: tuple, backprojection: bool = False,
):
    """Run ActionNet interaction prediction on current tracks.

    Expects track history at target_fps (source_fps / subsample_frames), so we take
    the last sequence_length = input_length_in_frames // subsample_frames frames.
    """
    assert config["mb_input_norm"] == "vid"
    min_valid_keypoints = config["min_keypoints_filter"]
    subsample_frames = config["subsample_frames"]
    if "select_input_range" in config:
        assert config["select_input_range"] == [0, -1], "select_input_range must be [0, -1] for live interaction prediction"

    input_length_in_frames = config["input_length_in_frames"]
    sequence_length = input_length_in_frames // subsample_frames
    # min_frames_inference = max(1, input_length_in_frames // 2)  # ActionNet: allow half window
    min_frames_inference = sequence_length # cannot allow different length because of stacking
    max_index_gap_allowed = 2

    eq_w, eq_h = (3840, 1920) if backprojection else image_size
    whtensor = torch.tensor([eq_w, eq_h], device=device)

    return_dict = {tid: "NC" for tid in current_tracks_history}
    valid_ids, valid_input_tensors = [], []

    for track_id, track_history in current_tracks_history.items():
        indexes = track_history["indexes"]
        input_tensors = track_history["ip_input_tensor"]
        if len(indexes) < min_frames_inference:
            return_dict[track_id] = f"not_enough_frames_{len(indexes)}"
            continue
        last_idx = np.array(indexes[-sequence_length:])
        if np.any(np.diff(last_idx) > max_index_gap_allowed):
            return_dict[track_id] = "index_gap_too_large"
            continue
        last_t = torch.stack(input_tensors[-sequence_length:], dim=0)
        if torch.any((last_t[:, :, 2] > args.kp_thresh).sum(dim=1) < min_valid_keypoints):
            return_dict[track_id] = "not_enough_valid_joints"
            continue
        valid_ids.append(track_id)
        valid_input_tensors.append(last_t)

    if not valid_ids:
        return return_dict

    valid_input_tensors = torch.stack(valid_input_tensors, dim=0)  # B, T, 17, 3

    if backprojection:
        persp_w, persp_h = image_size
        xy = valid_input_tensors[..., :2]
        scores = valid_input_tensors[..., 2:3]
        eq_coords = backproject_points_to_equirect(
            xy, (persp_w, persp_h), (eq_h, eq_w),
            fov=70.0, yaw=0.0, pitch=0.0, roll=0.0, device=device,
        )
        valid_input_tensors = torch.cat([eq_coords, scores], dim=-1)

    valid_input_tensors = coco2h36m(valid_input_tensors)
    scale = min(eq_w, eq_h) / 2.0
    valid_input_tensors[..., :2] = (valid_input_tensors[..., :2] - whtensor / 2.0) / scale
    valid_input_tensors = valid_input_tensors.unsqueeze(1)  # B, M, T, 17, 3
    valid_input_tensors = valid_input_tensors.to(device)

    # valid_input_tensors = valid_input_tensors.half()
    # with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
    with torch.no_grad():
        out = model(valid_input_tensors)
        probs = torch.sigmoid(out.squeeze())
        if probs.ndim == 0:
            probs = probs.unsqueeze(0)
        for i, tid in enumerate(valid_ids):
            return_dict[tid] = probs[i].item()

    return return_dict


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class HUIPredNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("hui_pred_node")

        rgb_topic = args.rgb_topic
        depth_topic = args.depth_topic or ""
        self.depth_topic = depth_topic
        self.compressed_depth = (
            depth_topic.endswith("/compressed")
            or depth_topic.endswith("/compressedDepth")
            or "/compressedDepth/" in depth_topic
        )
        yolo_path = args.yolo_model_path
        ip_ckpt = args.interaction_prediction_checkpoint or ""
        self._use_depth = depth_topic != ""
        self.camera_info_topic = args.camera_info_topic or ""
        self._camera_matrix = None
        self._camera_frame_id = "camera_optical_frame"

        # Args-like namespace consumed by helper functions
        self._args = types.SimpleNamespace(
            debug=args.debug,
            kp_thresh=args.kp_thresh,
            depth_scale=args.depth_scale,
            backprojection=args.backprojection,
            interaction_prediction_checkpoint=ip_ckpt if ip_ckpt else None,
        )

        # -- Load YOLO pose model --
        self.get_logger().info(f"Loading YOLO pose model from {yolo_path}")
        self.pose_model = YOLO(yolo_path)
        
        # -- Load interaction-prediction model (optional) --
        self.ip_model = None
        self.ip_model_index = -1
        self.ip_config = None
        if ip_ckpt:
            ip_ckpt_str = os.path.basename(ip_ckpt)
            if ip_ckpt_str in IP_MODELS_NAME_TO_INDEX:
                self.ip_model_index = IP_MODELS_NAME_TO_INDEX[ip_ckpt_str]
            else:
                self.get_logger().warning(f"IP model name {ip_ckpt_str} not found in IP_MODELS_NAME_TO_INDEX, using index -1")
                self.ip_model_index = -1
                
            assert("converted_" in ip_ckpt), "IP checkpoint must be a converted checkpoint, use the convert_checkpoints.py script to convert the checkpoint (split state dict and config)"
            self.get_logger().info(f"Loading IP checkpoint from {ip_ckpt}")
            
            # first load the config
            assert(os.path.exists(os.path.join(ip_ckpt, "meta.yaml"))), "Meta file not found"
            meta = read_yaml_to_dic(os.path.join(ip_ckpt, "meta.yaml"))
            self.ip_config = meta["config"]

            self.get_logger().info(
                f"IP model type: {self.ip_config['force_model_type']} | "
                f"AUC {meta.get('val_auc', 0):.4f} | AP {meta.get('val_ap', 0):.4f}"
            )

            # then load the state dict
            assert(os.path.exists(os.path.join(ip_ckpt, "model_state_dict.pt"))), "Model state dict not found"
            model_state_dict = torch.load(os.path.join(ip_ckpt, "model_state_dict.pt"), map_location="cpu")

            self.ip_model = load_model_from_config(self.ip_config, device="cuda")
            self.ip_model.load_state_dict(model_state_dict, strict=True)
            # self.ip_model.half()
            
            
            
            if not isinstance(self.ip_model, ActionNet):
                raise NotImplementedError(
                    f"{type(self.ip_model).__name__} not yet supported for live IP"
                )
            self.get_logger().info("IP model loaded successfully")

            # Target FPS = source_fps / subsample_frames; run YOLO+IP only at that rate
            subsample_frames = self.ip_config["subsample_frames"]
            target_fps = args.source_fps / subsample_frames
            self._min_interval = 1.0 / target_fps if target_fps > 0 else 0.0
            
            # add a tolereance
            self._min_interval = 0.9 * self._min_interval
            
            self.get_logger().info(
                f"Subsample frames={subsample_frames} -> target FPS={target_fps:.1f} (min_interval={self._min_interval*1000:.0f}ms)"
            )
        else:
            self._min_interval = None
            
        self._last_processed_rgb_time = 0.0
        self._last_received_rgb_time = 0.0
        
        # -- Current estimation mode --
        self.current_estimation_mode = args.default_estimation_mode
        if self.current_estimation_mode == "depth_based":
            assert(depth_topic != ""), "Depth topic is required for depth-based IP estimation"
        
        # -- IP estimation depth range --
        self.ip_estimation_depth_min = args.ip_estimation_depth_min
        self.ip_estimation_depth_max = args.ip_estimation_depth_max

        # -- IP estimation box height range (relative to image height) --
        self.ip_estimation_box_min = args.ip_estimation_box_min
        self.ip_estimation_box_max = args.ip_estimation_box_max
        assert 0.1 <= self.ip_estimation_box_min <= 1.0, "ip_estimation_box_min must be between 0.1 and 1.0"
        assert 0.1 <= self.ip_estimation_box_max <= 1.0, "ip_estimation_box_max must be between 0.1 and 1.0"
        assert self.ip_estimation_box_min < self.ip_estimation_box_max, "ip_estimation_box_min must be less than ip_estimation_box_max"

        # -- State --
        self.track_history: dict = {}
        self.frame_idx = 0
        self.filter_length = args.filter_length
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._process_lock = threading.Lock()
        self._cv_bridge = CvBridge() if CvBridge is not None else None
        if self._use_depth and self.compressed_depth and self._cv_bridge is None:
            self.get_logger().warning(
                "cv_bridge is unavailable; compressed depth decoding falls back to OpenCV PNG decode only"
            )

        # -- Publisher: /huipred/overlay (annotated image with detections) --
        self._pub_overlay = self.create_publisher(
            CompressedImage, "/huipred/overlay/compressed", 10
        )

        # -- Publisher: /huipred/tracks (flat float array; metadata + track payload) --
        self._pub_tracks = self.create_publisher(
            Float32MultiArray, "/huipred/tracks", 10
        )

        # -- Publisher: /huipred/torso_markers (oriented person cubes colored by IP score) --
        self._pub_torso_markers = self.create_publisher(
            MarkerArray, "/huipred/torso_markers", 10
        )

        # -- Publisher: /huipred/tracks_detections2d (2D bboxes + 3D torso poses) --
        self._pub_tracks_detections2d = self.create_publisher(
            Detection2DArray, "/huipred/tracks_detections2d", 10
        )
        self._overlay_mode = args.overlay_mode
        self.get_logger().info(f"Overlay mode: {self._overlay_mode}")

        # -- Subscriber: dynamic estimation mode switch --
        # Expected values in msg.data:
        # - "ip_inference"
        # - "depth_based"
        # - "box_based"
        self._estimation_mode_sub = self.create_subscription(
            String,
            args.estimation_mode_topic,
            self._estimation_mode_cb,
            10,
        )
        self.get_logger().info(
            f"Estimation mode topic: {args.estimation_mode_topic} (default={self.current_estimation_mode})"
        )

        # -- Subscribers --
        if self._use_depth:
            depth_mode = "CompressedImage" if self.compressed_depth else "Image"
            self.get_logger().info(
                f"Subscribing (synced): RGB={rgb_topic}  Depth={depth_topic} ({depth_mode})"
            )
            if self.camera_info_topic == "":
                raise ValueError("camera_info_topic is required when depth is enabled")
            self._sub_camera_info = self.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self._camera_info_cb,
                10,
            )
            self._sub_rgb = message_filters.Subscriber(self, CompressedImage, rgb_topic)
            if self.compressed_depth:
                self._sub_depth = message_filters.Subscriber(self, CompressedImage, depth_topic)
            else:
                self._sub_depth = message_filters.Subscriber(self, RosImage, depth_topic)
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._sub_rgb, self._sub_depth], queue_size=50, slop=0.5,
            )
            self._sync.registerCallback(self._synced_cb)
        else:
            self.get_logger().info(f"Subscribing (RGB only): {rgb_topic}")
            self._sub_rgb = self.create_subscription(
                CompressedImage, rgb_topic, self._rgb_only_cb, 1,
            )

        self.get_logger().info("HUI-Pred node ready")

    # ----- message decoding ---------------------------------------------------

    @staticmethod
    def _decode_compressed_rgb(msg: CompressedImage):
        """Decode a CompressedImage to a BGR numpy array."""
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    @staticmethod
    def _extract_png_payload(raw: bytes):
        """Return the byte slice containing a PNG payload, if any."""
        png_magic = b"\x89PNG\r\n\x1a\n"
        png_offset = raw.find(png_magic)
        if png_offset >= 0:
            return raw[png_offset:]

        if len(raw) > 12:
            # ROS compressedDepth transport prepends a 12-byte config header.
            payload = raw[12:]
            if payload.startswith(png_magic):
                return payload
        return None

    def _decode_compressed_depth(self, msg: CompressedImage):
        """Decode a CompressedImage carrying depth data.

        Prefers cv_bridge for RealSense/ROS compressedDepth messages, then falls
        back to OpenCV PNG decoding.
        """
        raw = bytes(msg.data)
        fmt = (msg.format or "").lower()
        if len(raw) == 0:
            self.get_logger().warn(
                f"Compressed depth message is empty (format='{msg.format}')"
            )
            return None

        if self._cv_bridge is not None:
            try:
                depth = self._cv_bridge.compressed_imgmsg_to_cv2(
                    msg, desired_encoding="passthrough"
                )
                if depth is not None and depth.size > 0:
                    if depth.ndim == 3:
                        depth = depth[:, :, 0]
                    return depth
            except Exception as e:
                self.get_logger().warning(
                    f"cv_bridge compressed depth decode failed (format='{msg.format}', "
                    f"bytes={len(raw)}): {e}"
                )

        png_payload = self._extract_png_payload(raw)
        if png_payload is None:
            self.get_logger().warn(
                "Compressed depth payload is not PNG (possibly RVL). "
                f"Use the raw depth topic instead of '/compressed' "
                f"(format='{msg.format}', bytes={len(raw)})."
            )
            return None

        buf = np.frombuffer(png_payload, dtype=np.uint8)
        if buf.size == 0:
            self.get_logger().warn(
                f"Compressed depth PNG payload is empty (format='{msg.format}')"
            )
            return None

        depth = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if depth is None:
            self.get_logger().warn(
                f"OpenCV failed to decode compressed depth PNG "
                f"(format='{msg.format}', bytes={len(raw)})"
            )
            return None
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return depth

    @staticmethod
    def _decode_depth_image(msg: RosImage, logger):
        """Decode a raw sensor_msgs/Image depth message to numpy array.

        Supports common depth encodings:
        - 16UC1 / mono16 -> uint16 depth
        - 32FC1 -> float32 depth
        """
        if msg.height == 0 or msg.width == 0:
            logger.warn(
                f"Depth image has invalid shape {msg.width}x{msg.height} "
                f"(encoding='{msg.encoding}')"
            )
            return None

        enc = (msg.encoding or "").lower()

        if enc in ("16uc1", "mono16"):
            dtype = np.dtype(np.uint16)
        elif enc == "32fc1":
            dtype = np.dtype(np.float32)
        else:
            logger.warn(f"Unsupported depth encoding: '{msg.encoding}'")
            return None

        if msg.is_bigendian:
            dtype = dtype.newbyteorder(">")
        else:
            dtype = dtype.newbyteorder("<")

        bytes_per_pixel = dtype.itemsize
        row_stride_bytes = msg.step if msg.step > 0 else msg.width * bytes_per_pixel
        expected_bytes = row_stride_bytes * msg.height
        if len(msg.data) < expected_bytes:
            logger.warn(
                f"Depth image data too short: got {len(msg.data)} bytes, "
                f"expected at least {expected_bytes} "
                f"({msg.width}x{msg.height}, step={msg.step}, encoding='{msg.encoding}')"
            )
            return None

        try:
            arr = np.frombuffer(msg.data, dtype=dtype)
            row_stride_elems = row_stride_bytes // bytes_per_pixel
            arr = arr.reshape((msg.height, row_stride_elems))[:, :msg.width]
            return np.ascontiguousarray(arr)
        except Exception as e:
            logger.warn(
                f"Failed to decode depth image ({msg.width}x{msg.height}, "
                f"step={msg.step}, encoding='{msg.encoding}'): {e}"
            )
            return None

    def _describe_depth_message(self, depth_msg):
        """Summarize a depth message for decode-failure logs."""
        if isinstance(depth_msg, CompressedImage):
            return (
                f"type=CompressedImage format='{depth_msg.format}' "
                f"bytes={len(depth_msg.data)}"
            )
        return (
            f"type=Image encoding='{depth_msg.encoding}' "
            f"size={depth_msg.width}x{depth_msg.height} step={depth_msg.step} "
            f"bytes={len(depth_msg.data)}"
        )

    def _log_depth_decode_failure(self, depth_msg):
        mode = "compressed" if self.compressed_depth else "raw"
        self.get_logger().warn(
            f"Failed to decode depth message ({mode}, topic='{self.depth_topic}', "
            f"{self._describe_depth_message(depth_msg)}). "
            "If using RealSense, prefer the raw topic "
            "'.../aligned_depth_to_color/image_raw' (no /compressedDepth suffix)."
        )

    def _camera_info_cb(self, msg: CameraInfo):
        """Cache the latest camera intrinsic matrix for 3D projection."""
        try:
            self._camera_matrix = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            self._camera_frame_id = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f"Failed to parse camera info: {e}")

    # ----- overlay / eye animation builders -----------------------------------

    def _build_overlay_image(
        self,
        bgr: np.ndarray,
        current_track_ids: list,
        boxes: np.ndarray,
        keypoints_all: np.ndarray,
        scores_all: np.ndarray,
    ) -> np.ndarray:
        """Build the camera image with bbox, skeleton, and IP score overlay."""
        overlay = bgr.copy()
        args = self._args
        for i, tid_raw in enumerate(current_track_ids):
            tid = int(tid_raw)
            label = f"ID:{tid}"
            ip_score = None
            if tid in self.track_history and self.track_history[tid]["ip_output"]:
                v = self.track_history[tid]["ip_output"][-1]
                if isinstance(v, (int, float)):
                    label += f" IP:{v:.2f}"
                    ip_score = v
                else:
                    label += f" {v}"
                    ip_score = 0.0
            color = (0, 0, 255)
            if ip_score is not None:
                v = max(0.0, min(1.0, float(ip_score)))
                red = int(255 * (1.0 - v))
                green = int(255 * v)
                color = (0, green, red)
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 8)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            font_scale = 3.5
            thickness = 10
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), base_line = cv2.getTextSize(label, font, font_scale, thickness)
            text_org = (center_x - text_width // 2, center_y + text_height // 2)
            cv2.putText(overlay, label, text_org, font, font_scale, color, thickness)
            kp = keypoints_all[i]
            sc = scores_all[i]
            for j, k in COCO_SKELETON:
                if sc[j] > args.kp_thresh and sc[k] > args.kp_thresh:
                    pt1 = tuple(kp[j].astype(int))
                    pt2 = tuple(kp[k].astype(int))
                    cv2.line(overlay, pt1, pt2, color, 2)
            for kp_pt, score in zip(kp, sc):
                if score > args.kp_thresh:
                    cv2.circle(overlay, (int(kp_pt[0]), int(kp_pt[1])), 4, color, -1)
        return overlay

    @staticmethod
    def _latest_ip_score(track_history: dict) -> float | None:
        """Return the latest valid IP score for marker coloring, or None if not computed."""
        if not track_history.get("ip_output"):
            return None
        value = track_history["ip_output"][-1]
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        return None

    @staticmethod
    def _ip_score_to_marker_rgba(ip_score: float | None) -> tuple[float, float, float, float]:
        """Gray when IP is not computed; red (0) to green (1) otherwise."""
        if ip_score is None:
            return 0.5, 0.5, 0.5, 1.0
        value = max(0.0, min(1.0, float(ip_score)))
        return 1.0 - value, value, 0.0, 1.0

    @staticmethod
    def _make_person_cube_marker(
        stamp,
        frame_id: str,
        track_id: int,
        position_xyz: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
        ip_score: float | None,
    ) -> Marker:
        """Build a human-sized cube marker at the torso, oriented with body yaw."""
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = "huipred_torso"
        marker.id = track_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(position_xyz[0])
        marker.pose.position.y = float(position_xyz[1])
        marker.pose.position.z = float(position_xyz[2])
        qx, qy, qz, qw = orientation_xyzw
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
        marker.scale.x = PERSON_CUBE_WIDTH_M
        marker.scale.y = PERSON_CUBE_DEPTH_M
        marker.scale.z = PERSON_CUBE_HEIGHT_M
        red, green, blue, alpha = HUIPredNode._ip_score_to_marker_rgba(ip_score)
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = alpha
        return marker

    @staticmethod
    def _make_track_detection2d(
        stamp,
        frame_id: str,
        track_id: int,
        bbox_xyxy: np.ndarray,
        box_conf: float,
        person_pose: dict,
    ) -> Detection2D:
        """Build a Detection2D with bbox, track id, torso pose, and yaw orientation."""
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy.tolist()]
        detection = Detection2D()
        detection.header.stamp = stamp
        detection.header.frame_id = frame_id
        detection.id = str(track_id)
        detection.bbox.center.position.x = (x1 + x2) / 2.0
        detection.bbox.center.position.y = (y1 + y2) / 2.0
        detection.bbox.center.theta = person_pose["bbox_theta"]
        detection.bbox.size_x = x2 - x1
        detection.bbox.size_y = y2 - y1

        hypothesis = ObjectHypothesis()
        hypothesis.class_id = "person"
        hypothesis.score = float(box_conf)

        result = ObjectHypothesisWithPose()
        result.hypothesis = hypothesis
        position = person_pose["position"]
        if position is not None:
            result.pose.pose.position.x = float(position[0])
            result.pose.pose.position.y = float(position[1])
            result.pose.pose.position.z = float(position[2])
        else:
            result.pose.pose.position.x = -1.0
            result.pose.pose.position.y = -1.0
            result.pose.pose.position.z = -1.0
        qx, qy, qz, qw = person_pose["orientation"]
        result.pose.pose.orientation.x = qx
        result.pose.pose.orientation.y = qy
        result.pose.pose.orientation.z = qz
        result.pose.pose.orientation.w = qw
        detection.results.append(result)
        return detection

    @staticmethod
    def _build_eye_animation_image(
        height: int,
        width: int,
        ip_score: float,
        box_xyxy: np.ndarray,
    ) -> np.ndarray:
        """Build a synthetic eye animation driven by highest-IP person.

        Openness: closed below MIN_EYES_SCALE_THRESHOLD, full open above MAX_EYES_SCALE_THRESHOLD.
        Gaze: eyes point at box position in frame. Color: light red -> pop green.
        No face, only two eyes; gap between eyes = 0.1 * image width.
        """
        out = np.zeros((height, width, 3), dtype=np.uint8)
        out[:] = (0, 0, 0)  # transparent when blended so only eyes show
        cx, cy = width // 2, height // 2
        half_gap = int(0.05 * width)  # gap between eyes = 0.1 * width
        eye_w = int(0.18 * width)
        eye_h_base = max(2, int(0.48 * height))
        pupil_radius = max(2, eye_w // 5)
        max_pupil_offset = eye_w // 2
        left_eye_cx = cx - half_gap - eye_w
        right_eye_cx = cx + half_gap + eye_w
        eye_cy = cy
        if ip_score is None:
            openness = 0.1
        else:
            s = float(ip_score)
            if s <= MIN_EYES_SCALE_THRESHOLD:
                openness = 0.1
            elif s >= MAX_EYES_SCALE_THRESHOLD:
                openness = 1.0
            else:
                openness = (s - MIN_EYES_SCALE_THRESHOLD) / (
                    MAX_EYES_SCALE_THRESHOLD - MIN_EYES_SCALE_THRESHOLD
                )
        if box_xyxy is not None:
            bx = (box_xyxy[0] + box_xyxy[2]) / 2.0
            by = (box_xyxy[1] + box_xyxy[3]) / 2.0
            gaze_x = (bx / width - 0.5) * 2.0
            gaze_y = (by / height - 0.5) * 2.0
        else:
            gaze_x = gaze_y = 0.0
        gaze_x = max(-1.0, min(1.0, gaze_x))
        gaze_y = max(-1.0, min(1.0, gaze_y))
        # Invert horizontal gaze for screen/mirrored display
        pupil_dx = int(-gaze_x * max_pupil_offset)
        pupil_dy = int(gaze_y * max_pupil_offset)
        if ip_score is None:
            t = 0.0
        else:
            t = max(0.0, min(1.0, float(ip_score)))
        b, g, r = int(255 - t * 155), int(200 + t * 55), int(255 - t * 255)
        eye_color = (b, g, r)
        for ex in (left_eye_cx, right_eye_cx):
            if openness <= 0.0:
                cv2.line(out, (ex - eye_w, eye_cy), (ex + eye_w, eye_cy), eye_color, 2)
            else:
                eh = max(1, int(eye_h_base * openness))
                cv2.ellipse(out, (ex, eye_cy), (eye_w, eh), 0, 0, 360, eye_color, -1)
                cv2.ellipse(out, (ex, eye_cy), (eye_w, eh), 0, 0, 360, (200, 200, 200), 1)
            pupil_x = max(ex - eye_w + pupil_radius, min(ex + eye_w - pupil_radius, ex + pupil_dx))
            pupil_y = max(eye_cy - eye_h_base + pupil_radius,
                         min(eye_cy + eye_h_base - pupil_radius, eye_cy + pupil_dy))
            cv2.circle(out, (pupil_x, pupil_y), pupil_radius, (30, 30, 30), -1)
        return out

    # ----- callbacks ----------------------------------------------------------

    def _synced_cb(self, rgb_msg: CompressedImage, depth_msg: CompressedImage):
        time_elapsed = time.perf_counter() - self._last_received_rgb_time
        self._last_received_rgb_time = time.perf_counter()
        self.get_logger().info(f"Just for info : received RGB message after {time_elapsed:.4f}s since last message")
        bgr = self._decode_compressed_rgb(rgb_msg)
        if self.compressed_depth:
            depth = self._decode_compressed_depth(depth_msg)
        else:
            depth = self._decode_depth_image(depth_msg, self.get_logger())
        if bgr is None:
            self.get_logger().warn("Failed to decode RGB message")
            return
        if depth is None:
            self._log_depth_decode_failure(depth_msg)
            return
        self._process_frame(bgr, depth, rgb_msg.header.stamp)

    def _rgb_only_cb(self, rgb_msg: CompressedImage):
        time_elapsed = time.perf_counter() - self._last_received_rgb_time
        self._last_received_rgb_time = time.perf_counter()
        self.get_logger().info(f"Just for info : received RGB message after {time_elapsed:.4f}s since last message")
        bgr = self._decode_compressed_rgb(rgb_msg)
        if bgr is None:
            self.get_logger().warn("Failed to decode RGB message")
            return
        self._process_frame(bgr, None, rgb_msg.header.stamp)

    # ----- dynamic estimation mode ------------------------------------------

    def _estimation_mode_cb(self, msg: String):
        new_mode = (msg.data or "").strip()
        if new_mode not in ("ip_inference", "depth_based", "box_based", "none_based"):
            self.get_logger().warn(
                f"Ignoring invalid estimation mode '{new_mode}'. Expected 'ip_inference', 'depth_based', 'box_based', or 'none_based'."
            )
            return

        if new_mode == self.current_estimation_mode:
            return

        old_mode = self.current_estimation_mode
        self.current_estimation_mode = new_mode
        self.get_logger().info(f"Switch estimation mode: {old_mode} -> {new_mode}")

        # Clear per-track IP outputs so we don't mix results from a previous mode.
        for tid, hist in self.track_history.items():
            hist["ip_output"] = []
            hist["ip_output_filtered"] = []

    # ----- main per-frame pipeline -------------------------------------------

    def _process_frame(self, bgr: np.ndarray, depth_image: np.ndarray, image_stamp):
        
        if not self._process_lock.acquire(blocking=False):
            self.get_logger().info("Skipping frame — previous frame still processing")
            return
        
        try:
            time_elapsed = time.perf_counter() - self._last_processed_rgb_time            
            if self._min_interval is not None:
                if time_elapsed < self._min_interval:
                    self.get_logger().info(f"Skipping frame — not enough time has passed since last processed message (time difference: {time_elapsed:.4f}s)")
                    return
                else:
                    self.get_logger().info(f"Processed frame — time difference is enough: {time_elapsed:.4f}s. Will process and reset last processed time.")
                    self._last_processed_rgb_time = time.perf_counter()
            self._process_frame_locked(bgr, depth_image, image_stamp)
        finally:
            self._process_lock.release()

    def _process_frame_locked(self, bgr: np.ndarray, depth_image: np.ndarray, image_stamp):
        t_start = time.perf_counter()
        t_ip = 0.0
        args = self._args

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        if image.width == 3840:
            image = image.resize((1920, 960))

        # NB, default camera (Gemini RGBD is 1280x800)
        
        # -- YOLO detection + tracking + pose --
        t_infer = time.perf_counter()
        results = self.pose_model.track(image, persist=True, verbose=False, tracker="bytetrack.yaml")[0]
        t_infer = time.perf_counter() - t_infer
                
        if args.debug:
            orig_image = results.orig_img
            debug_img_path = f"./debug_frames/frame_orig_image_yolo_{self.frame_idx:06d}.jpg"
            self.get_logger().info(f"Saving debug frame to {debug_img_path}")
            os.makedirs(os.path.dirname(debug_img_path), exist_ok=True)
            cv2.imwrite(debug_img_path, cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR))

        if results.boxes.id is None:
            current_track_ids = []
            boxes = confs = keypoints_all = scores_all = []
            depths = []
            torso_pixels = []
            torso_positions_3d = []
            person_poses = []
        else:
            current_track_ids = results.boxes.id.int().cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            keypoints_all = results.keypoints.xy
            scores_all = results.keypoints.conf
            ip_input_tensor = torch.cat(
                [keypoints_all, scores_all.unsqueeze(-1)], dim=2
            ).clone()
            keypoints_all = keypoints_all.cpu().numpy()
            scores_all = scores_all.cpu().numpy()

            depths = []
            torso_pixels = []
            torso_positions_3d = []
            person_poses = []
            if depth_image is not None:
                for i in range(len(current_track_ids)):
                    d_raw, torso_center_uv = estimate_torso_depth_and_center(
                        keypoints_all[i], scores_all[i], depth_image, args.kp_thresh, self.get_logger()
                    )
                    depth_m = d_raw / args.depth_scale if d_raw is not None else None
                    depths.append(depth_m)
                    torso_pixels.append(torso_center_uv)
                    if self._camera_matrix is None:
                        self.get_logger().warn("Camera matrix is not set, 3D projection will fail")
                    torso_positions_3d.append(
                        project_pixel_to_3d(
                            torso_center_uv,
                            depth_m,
                            self._camera_matrix,
                        )
                    )
                    person_poses.append(
                        compute_person_pose_from_shoulders(
                            keypoints_all[i],
                            scores_all[i],
                            depth_image,
                            self._camera_matrix,
                            args.depth_scale,
                            args.kp_thresh,
                            torso_center_uv,
                            depth_m,
                        )
                    )
            else:
                depths = [None] * len(current_track_ids)
                torso_pixels = [None] * len(current_track_ids)
                torso_positions_3d = [None] * len(current_track_ids)
                for i in range(len(current_track_ids)):
                    person_poses.append(
                        compute_person_pose_from_shoulders(
                            keypoints_all[i],
                            scores_all[i],
                            None,
                            self._camera_matrix,
                            args.depth_scale,
                            args.kp_thresh,
                            None,
                            None,
                        )
                    )

            for i, tid in enumerate(current_track_ids):
                tid = int(tid)
                if tid not in self.track_history:
                    self.track_history[tid] = {
                        "detections": [], "poses": [],
                        "ip_input_tensor": [], "indexes": [], "ip_output": [],
                        "ip_output_filtered": []
                    }
                self.track_history[tid]["detections"].append({
                    "frame": self.frame_idx,
                    "bbox": boxes[i].tolist(),
                    "conf": float(confs[i]),
                    "depth": depths[i] if depths[i] is not None else None,
                    "torso_pixel": torso_pixels[i].tolist() if torso_pixels[i] is not None else None,
                    "torso_position_3d": torso_positions_3d[i].tolist() if torso_positions_3d[i] is not None else None,
                })
                self.track_history[tid]["poses"].append({
                    "frame": self.frame_idx,
                    "keypoints": keypoints_all[i].tolist(),
                    "scores": scores_all[i].tolist(),
                })
                self.track_history[tid]["ip_input_tensor"].append(ip_input_tensor[i])
                self.track_history[tid]["indexes"].append(self.frame_idx)

            # -- Interaction prediction --
            if self.ip_model is not None and self.current_estimation_mode == "ip_inference":
                t_ip_s = time.perf_counter()
                cur = {int(t): self.track_history[int(t)] for t in current_track_ids}
                ip_dict = action_net_inference(
                    args, self.ip_model, self.ip_config, cur,
                    device=self._device,
                    image_size=(image.width, image.height),
                    backprojection=args.backprojection,
                )
                t_ip = time.perf_counter() - t_ip_s
                for tid, val in ip_dict.items():
                    self.track_history[tid]["ip_output"].append(val)
                    history_length = len(self.track_history[tid]["ip_output"])
                    FILTER_LENGTH = self.filter_length
                    last_ip_outputs = self.track_history[tid]["ip_output"][-min(FILTER_LENGTH,history_length):]
                    last_ip_outputs_filtered = []
                    for v in last_ip_outputs:
                        if isinstance(v, (int, float, np.floating)):
                            last_ip_outputs_filtered.append(float(v))
                        else:
                            last_ip_outputs_filtered.append(0.0)
                    print(f"Track {tid}  | last_ip_outputs_filtered: {[f'{v:.2f}' for v in last_ip_outputs_filtered]}")
                    self.track_history[tid]["ip_output_filtered"].append(float(np.mean(np.array(last_ip_outputs_filtered)))) # mean of last 3 ip outputs

            elif len(depths) > 0 and depths[0] is not None and self.current_estimation_mode == "depth_based":
                ip_estimation_depth_range = [self.ip_estimation_depth_min, self.ip_estimation_depth_max] # linear mapping between 0 and 1 for IP, using as input depth scaled between min and max depth
                for i, tid in enumerate(current_track_ids):
                    track_depth = self.track_history[tid]["detections"][-1]["depth"]
                    if track_depth is not None and track_depth > 0:
                        estimated_ip = (ip_estimation_depth_range[1] - track_depth) / (ip_estimation_depth_range[1] - ip_estimation_depth_range[0])
                        estimated_ip = max(0, min(1, estimated_ip))
                        print(f"Track {tid}  | estimated_ip: {estimated_ip:.2f} from depth {track_depth:.2f}m")
                        self.track_history[tid]["ip_output"].append(estimated_ip)
                        self.track_history[tid]["ip_output_filtered"].append(estimated_ip)

            elif self.current_estimation_mode == "box_based":
                # ip_estimation_box_range = [self.ip_estimation_box_min, self.ip_estimation_box_max]
                image_height = image.height
                for i, tid in enumerate(current_track_ids):
                    box = self.track_history[tid]["detections"][-1]["bbox"]
                    box_bottom_y = box[3]
                    box_bottom_relative = (image_height - box_bottom_y) / image_height
                    # 3m threhsold is 17.5% of the image height, 1.5m is 10% of the image height
                    estimated_ip = (0.26 - box_bottom_relative) / (0.26 - 0.1)
                    estimated_ip = max(0, min(1, estimated_ip))
                    # relative_box_height = (box[3] - box[1]) / image_height
                    # estimated_ip = (relative_box_height - ip_estimation_box_range[0]) / (ip_estimation_box_range[1] - ip_estimation_box_range[0])
                    # estimated_ip = max(0, min(1, estimated_ip))
                    # print(f"Track {tid}  | estimated_ip: {estimated_ip:.2f} from box height {relative_box_height:.2f}")
                    print(f"Track {tid}  | estimated_ip: {estimated_ip:.2f} from box bottom {box_bottom_relative:.2f} (interp 10% to 26% of image height)")
                    self.track_history[tid]["ip_output"].append(estimated_ip)
                    self.track_history[tid]["ip_output_filtered"].append(estimated_ip)
            
        # -- Build output image and publish --
        h, w = bgr.shape[:2]
        if self._overlay_mode == "overlay":
            img = self._build_overlay_image(
                bgr, current_track_ids, boxes, keypoints_all, scores_all
            )
        elif self._overlay_mode == "eye_animation":
            best_ip_score = None
            best_box = None
            if len(current_track_ids) > 0:
                best_score = -1.0
                best_idx = 0
                for i, tid_raw in enumerate(current_track_ids):
                    tid = int(tid_raw)
                    if tid in self.track_history and self.track_history[tid]["ip_output"]:
                        v = self.track_history[tid]["ip_output"][-1]
                        if isinstance(v, (int, float)) and v > best_score:
                            best_score = v
                            best_idx = i
                if best_score >= 0:
                    best_ip_score = best_score
                    best_box = boxes[best_idx]
            eye_img = self._build_eye_animation_image(h, w, best_ip_score, best_box)
            # Composite eye animation above base image at 60% opacity
            img = cv2.addWeighted(eye_img, 0.8, bgr, 0.2, 0)

        elif self._overlay_mode == "nodrawing":
            img = None

        else:
            self.get_logger().error(f"Invalid overlay mode: {self._overlay_mode}")
            return

        if img is not None:
            img_msg = CompressedImage()
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = self._camera_frame_id
            img_msg.format = "jpeg"
            img_msg.data = np.array(cv2.imencode(".jpg", img)[1]).tobytes()
            self._pub_overlay.publish(img_msg)

        if self.current_estimation_mode != "none_based":
            tracks_msg = Float32MultiArray()
            tracks_data = []
            for i, tid_raw in enumerate(current_track_ids):
                tid = int(tid_raw)

                # bbox + conf
                x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]
                conf = float(confs[i]) if len(confs) else -1.0

                # depth
                _depth_default_value = -1.0 if self.depth_topic != "" else -2.0
                depth = _depth_default_value
                if depths[i] is not None:
                    depth = float(depths[i])

                torso_u = -1.0
                torso_v = -1.0
                torso_x = -1.0
                torso_y = -1.0
                torso_z = depth
                if torso_pixels[i] is not None:
                    torso_u = float(torso_pixels[i][0])
                    torso_v = float(torso_pixels[i][1])
                if torso_positions_3d[i] is not None:
                    torso_x = float(torso_positions_3d[i][0])
                    torso_y = float(torso_positions_3d[i][1])
                    torso_z = float(torso_positions_3d[i][2])

                # IP outputs
                ip_out = -1.0
                ip_out_f = -1.0
                last_skeleton = [-1.0] * 17 * 3
                if tid in self.track_history:
                    if self.track_history[tid].get("ip_output"):
                        v = self.track_history[tid]["ip_output"][-1]
                        if isinstance(v, (int, float, np.floating)):
                            ip_out = float(v)
                        elif isinstance(v, str):
                            if "not_enough_frames" in v:
                                actual_frames = int(v.split("_")[-1])
                                ip_out = -1.0 * float(actual_frames)
                            elif "index_gap_too_large" in v:
                                ip_out = -2.0
                            elif "not_enough_valid_joints" in v:
                                ip_out = -3.0
                    if self.track_history[tid].get("ip_output_filtered"):
                        v = self.track_history[tid]["ip_output_filtered"][-1]
                        if isinstance(v, (int, float, np.floating)):
                            ip_out_f = float(v)
                    if self.track_history[tid].get("poses"):
                        skeleton_array_kps = np.array(self.track_history[tid]["poses"][-1]["keypoints"]).reshape(17, -1) # 17, 2
                        skeleton_array_kps = skeleton_array_kps.astype(np.float32)
                        skeleton_array_kps[:,0] /= w
                        skeleton_array_kps[:,1] /= h
                        skeleton_array_conf = np.array(self.track_history[tid]["poses"][-1]["scores"]).reshape(17, -1) # 17, 1
                        skeleton_array_conf = skeleton_array_conf.astype(np.float32)
                        skeleton_array = np.concatenate([skeleton_array_kps, skeleton_array_conf], axis=-1) # 17, 3
                        last_skeleton = skeleton_array.flatten().tolist()
                        # print(last_skeleton)
                
                tracks_data.extend([float(h), 
                                    float(w), 
                                    float(tid), 
                                    x1, 
                                    y1, 
                                    x2, 
                                    y2, 
                                    conf, 
                                    depth, 
                                    torso_u,
                                    torso_v,
                                    torso_x,
                                    torso_y,
                                    torso_z,
                                    ip_out, 
                                    ip_out_f, 
                                    t_infer, 
                                    t_ip, 
                                    float(self.ip_model_index)] + last_skeleton)

            tracks_msg.data = tracks_data
            self._pub_tracks.publish(tracks_msg)

            # Publish oriented person cubes for valid shoulder-based poses.
            stamp = image_stamp
            torso_markers_msg = MarkerArray()
            clear_marker = Marker()
            clear_marker.header.stamp = stamp
            clear_marker.header.frame_id = self._camera_frame_id
            clear_marker.action = Marker.DELETEALL
            clear_marker.ns = "huipred_torso"
            torso_markers_msg.markers.append(clear_marker)
            for i, tid_raw in enumerate(current_track_ids):
                person_pose = person_poses[i]
                if person_pose is None or person_pose["position"] is None:
                    continue
                tid = int(tid_raw)
                ip_score = None
                if tid in self.track_history:
                    ip_score = self._latest_ip_score(self.track_history[tid])
                torso_markers_msg.markers.append(
                    self._make_person_cube_marker(
                        stamp,
                        self._camera_frame_id,
                        tid,
                        person_pose["position"],
                        person_pose["orientation"],
                        ip_score,
                    )
                )
            self._pub_torso_markers.publish(torso_markers_msg)

            detections2d_msg = Detection2DArray()
            detections2d_msg.header.stamp = stamp
            detections2d_msg.header.frame_id = self._camera_frame_id
            for i, tid_raw in enumerate(current_track_ids):
                person_pose = person_poses[i]
                if person_pose is None:
                    continue
                detections2d_msg.detections.append(
                    self._make_track_detection2d(
                        stamp,
                        self._camera_frame_id,
                        int(tid_raw),
                        boxes[i],
                        float(confs[i]) if len(confs) else -1.0,
                        person_pose,
                    )
                )
            self._pub_tracks_detections2d.publish(detections2d_msg)

        t_total = time.perf_counter() - t_start
        self.get_logger().info(
            f"Frame {self.frame_idx} | {len(current_track_ids)} tracks | "
            f"yolo: {t_infer*1000:.1f}ms, ip: {t_ip*1000:.1f}ms, "
            f"total: {t_total*1000:.1f}ms"
        )
        self.frame_idx += 1


# ---------------------------------------------------------------------------

def main(parsed_args, ros_args):
    rclpy.init(args=ros_args)
    node = HUIPredNode(parsed_args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ROS2 node for HUI-Pred: subscribe to RGB/depth CompressedImage topics, run YOLO pose + interaction prediction.",
    )
    parser.add_argument("--rgb_topic", "-r", type=str, default="/rgbd/realsense_head_front/color/image_raw/compressed",
                        help="Topic for RGB CompressedImage")
    parser.add_argument("--depth_topic", "-dt", type=str, default="/rgbd/realsense_head_front/aligned_depth_to_color/image_raw/compressedDepth",
                        help="Topic for depth Image or CompressedImage/compressedDepth (optional; if set, RGB and depth are time-synced)")
    parser.add_argument("--camera_info_topic", type=str, default="/rgbd/realsense_head_front/color/camera_info",
                        help="Topic for CameraInfo used to project torso depth into 3D camera coordinates")
    parser.add_argument("--yolo_model_path", "-y", type=str, default="checkpoints/yolo26x-pose.pt",
                        help="Path to YOLO pose model")
    parser.add_argument("--interaction_prediction_checkpoint", "-ip", type=str, default="checkpoints/converted_mb_FineTuned_28_02_26_best_ap",
                        help="Path to interaction prediction checkpoint (optional)")
    parser.add_argument("--kp_thresh", type=float, default=0.3,
                        help="Keypoint confidence threshold")
    parser.add_argument("--depth_scale", type=float, default=1000.0,
                        help="Depth scale factor (depth units per meter, e.g. 1000 for mm)")
    parser.add_argument("--backprojection", "-bp", action="store_true", default=False,
                        help="Backproject joints from perspective to equirect before IP inference")
    parser.add_argument("--source_fps", type=float, default=15.0,
                        help="Assumed source/camera FPS; target FPS = source_fps / subsample_frames (from IP config)")
    parser.add_argument("--debug", "-d", action="store_true", default=False,
                        help="Save debug frames")
    parser.add_argument("--filter_length", type=int, default=3,
                        help="Filter length for IP output (mean of last N IP outputs)")
    parser.add_argument("--ip_estimation_depth_min", "-ipdmin", type=float, default=0.5,
                        help="Minimum depth for IP estimation (in meters) for depth-based IP estimation")
    parser.add_argument("--ip_estimation_depth_max", "-ipdmax", type=float, default=2.5,
                        help="Maximum depth for IP estimation (in meters) for depth-based IP estimation")
    parser.add_argument("--ip_estimation_box_min", "-ipbmin", type=float, default=0.15,
                        help="Minimum bbox height (relative to image height, 0.1-1.0) for box-based IP estimation")
    parser.add_argument("--ip_estimation_box_max", "-ipbmax", type=float, default=0.55,
                        help="Maximum bbox height (relative to image height, 0.1-1.0) for box-based IP estimation")
    parser.add_argument("--default_estimation_mode", "-dem", type=str, choices=["ip_inference", "depth_based", "box_based", "none_based"],
                        default="ip_inference",
                        help="Default estimation mode: 'ip_inference' = use IP inference model; 'depth_based' = use depth-based IP estimation; 'box_based' = use bbox height as proxy; 'none_based' = do not publish tracks")
    parser.add_argument("--overlay_mode", "-om", type=str, choices=["overlay", "eye_animation", "nodrawing"],
                        default="overlay",
                        help="Display mode: 'overlay' = camera image with bbox/skeleton/IP overlay; 'eye_animation' = synthetic eye animation driven by highest-IP person; 'nodrawing' = no drawing and no publishing of image")
    parser.add_argument(
        "--estimation_mode_topic",
        "-emt",
        type=str,
        default="/huipred/estimation_mode",
        help="ROS topic publishing std_msgs/String to switch estimation mode ('ip_inference', 'depth_based', 'box_based', or 'none_based')",
    )
    parsed_args, ros_args = parser.parse_known_args()
    main(parsed_args, ros_args)
