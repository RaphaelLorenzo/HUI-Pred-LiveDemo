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
from sensor_msgs.msg import CompressedImage, Image as RosImage
from std_msgs.msg import Float32MultiArray
import message_filters


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

COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
    [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
]

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

def estimate_torso_depth(
    keypoints: np.ndarray,
    scores: np.ndarray,
    depth_image: np.ndarray,
    kp_thresh: float,
):
    """
    Estimate depth at the center of the torso using diagonal sampling.

    Uses 6 points: 3 sampled at 0.25, 0.5, 0.75 on diagonal left_shoulder->right_hip
    and 3 on diagonal right_shoulder->left_hip.  Takes median of valid samples.
    """
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
    LEFT_HIP, RIGHT_HIP = 11, 12
    sample_ratios = [0.25, 0.5, 0.75]
    depth_samples = []
    h, w = depth_image.shape[:2]

    if scores[LEFT_SHOULDER] > kp_thresh and scores[RIGHT_HIP] > kp_thresh:
        pt1, pt2 = keypoints[LEFT_SHOULDER], keypoints[RIGHT_HIP]
        for r in sample_ratios:
            x, y = int(pt1[0] + r * (pt2[0] - pt1[0])), int(pt1[1] + r * (pt2[1] - pt1[1]))
            if 0 <= x < w and 0 <= y < h:
                depth_samples.append(depth_image[y, x])

    if scores[RIGHT_SHOULDER] > kp_thresh and scores[LEFT_HIP] > kp_thresh:
        pt1, pt2 = keypoints[RIGHT_SHOULDER], keypoints[LEFT_HIP]
        for r in sample_ratios:
            x, y = int(pt1[0] + r * (pt2[0] - pt1[0])), int(pt1[1] + r * (pt2[1] - pt1[1]))
            if 0 <= x < w and 0 <= y < h:
                depth_samples.append(depth_image[y, x])

    if len(depth_samples) == 0:
        return -1.0
    return float(np.median(depth_samples))


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
    min_frames_inference = input_length_in_frames # cannot allow different length because of stacking
    max_index_gap_allowed = 2

    eq_w, eq_h = (3840, 1920) if backprojection else image_size
    whtensor = torch.tensor([eq_w, eq_h], device=device)

    return_dict = {tid: "NC" for tid in current_tracks_history}
    valid_ids, valid_input_tensors = [], []

    for track_id, track_history in current_tracks_history.items():
        indexes = track_history["indexes"]
        input_tensors = track_history["ip_input_tensor"]
        if len(indexes) < min_frames_inference:
            return_dict[track_id] = "not_enough_frames"
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
        self.compressed_depth = "compressed" in depth_topic
        yolo_path = args.yolo_model_path
        ip_ckpt = args.interaction_prediction_checkpoint or ""
        self._use_depth = depth_topic != ""

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
        self.ip_config = None
        if ip_ckpt:
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

        # -- State --
        self.track_history: dict = {}
        self.frame_idx = 0
        self.filter_length = args.filter_length
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._process_lock = threading.Lock()

        # -- Publisher: /huipred/overlay (annotated image with detections) --
        self._pub_overlay = self.create_publisher(
            CompressedImage, "/huipred/overlay/compressed", 10
        )

        # -- Publisher: /huipred/tracks (flat float array; 8 floats per track) --
        # Per track: [x1, y1, x2, y2, conf, depth(-1 if none), ip_output(-1 if none), ip_output_filtered(-1 if none)]
        self._pub_tracks = self.create_publisher(
            Float32MultiArray, "/huipred/tracks", 10
        )
        self._overlay_mode = args.overlay_mode
        self.get_logger().info(f"Overlay mode: {self._overlay_mode}")

        # -- Subscribers --
        if self._use_depth:
            self.get_logger().info(
                f"Subscribing (synced): RGB={rgb_topic}  Depth={depth_topic}"
            )
            self._sub_rgb = message_filters.Subscriber(self, CompressedImage, rgb_topic)
            if self.compressed_depth:
                self._sub_depth = message_filters.Subscriber(self, CompressedImage, depth_topic)
            else:
                self._sub_depth = message_filters.Subscriber(self, RosImage, depth_topic)
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._sub_rgb, self._sub_depth], queue_size=10, slop=0.1,
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
    def _decode_compressed_depth(msg: CompressedImage):
        """Decode a CompressedImage carrying depth data.

        Handles both raw-compressed (PNG/TIFF) and the ROS compressedDepth
        transport plugin format (12-byte header before the PNG payload).
        """
        raw = bytes(msg.data)
        fmt = msg.format.lower()

        if "compresseddepth" in fmt:
            # compressedDepth transport: 12-byte ConfigHeader then PNG/RVL
            if len(raw) <= 12:
                return None
            raw = raw[12:]

        buf = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def _decode_depth_image(msg: RosImage):
        """Decode a raw sensor_msgs/Image depth message to numpy array.

        Supports common depth encodings:
        - 16UC1 / mono16 -> uint16 depth
        - 32FC1 -> float32 depth
        """
        if msg.height == 0 or msg.width == 0:
            return None

        enc = (msg.encoding or "").lower()

        if enc in ("16uc1", "mono16"):
            dtype = np.dtype(np.uint16)
            channels = 1
        elif enc == "32fc1":
            dtype = np.dtype(np.float32)
            channels = 1
        else:
            return None

        # Respect endianness from ROS message.
        if msg.is_bigendian:
            dtype = dtype.newbyteorder(">")
        else:
            dtype = dtype.newbyteorder("<")

        try:
            arr = np.frombuffer(msg.data, dtype=dtype)
            expected_pixels = msg.height * msg.width * channels
            if arr.size < expected_pixels:
                return None
            arr = arr[:expected_pixels].reshape((msg.height, msg.width))
            return arr
        except Exception:
            return None

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
            depth = self._decode_depth_image(depth_msg)
        if bgr is None:
            self.get_logger().warn("Failed to decode RGB message")
            return
        if depth is None:
            self.get_logger().warn("Failed to decode depth message")
            return
        self._process_frame(bgr, depth)

    def _rgb_only_cb(self, rgb_msg: CompressedImage):
        time_elapsed = time.perf_counter() - self._last_received_rgb_time
        self._last_received_rgb_time = time.perf_counter()
        self.get_logger().info(f"Just for info : received RGB message after {time_elapsed:.4f}s since last message")
        bgr = self._decode_compressed_rgb(rgb_msg)
        if bgr is None:
            self.get_logger().warn("Failed to decode RGB message")
            return
        self._process_frame(bgr, None)

    # ----- main per-frame pipeline -------------------------------------------

    def _process_frame(self, bgr: np.ndarray, depth_image: np.ndarray):
        
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
            self._process_frame_locked(bgr, depth_image)
        finally:
            self._process_lock.release()

    def _process_frame_locked(self, bgr: np.ndarray, depth_image: np.ndarray):
        t_start = time.perf_counter()
        t_ip = 0.0
        args = self._args

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        if image.width == 3840:
            image = image.resize((1920, 960))

        # -- YOLO detection + tracking + pose --
        t_infer = time.perf_counter()
        results = self.pose_model.track(image, persist=True, verbose=False)[0]
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
            if depth_image is not None:
                for i in range(len(current_track_ids)):
                    d_raw = estimate_torso_depth(
                        keypoints_all[i], scores_all[i], depth_image, args.kp_thresh
                    )
                    depths.append(d_raw / args.depth_scale if d_raw > 0 else None)
            else:
                depths = [None] * len(current_track_ids)

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
                    **({"depth": depths[i]} if depths[i] is not None else {}),
                })
                self.track_history[tid]["poses"].append({
                    "frame": self.frame_idx,
                    "keypoints": keypoints_all[i].tolist(),
                    "scores": scores_all[i].tolist(),
                })
                self.track_history[tid]["ip_input_tensor"].append(ip_input_tensor[i])
                self.track_history[tid]["indexes"].append(self.frame_idx)

            # -- Interaction prediction --
            if self.ip_model is not None:
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
            img_msg.header.frame_id = "camera_optical_frame"
            img_msg.format = "jpeg"
            img_msg.data = np.array(cv2.imencode(".jpg", img)[1]).tobytes()
            self._pub_overlay.publish(img_msg)

        tracks_msg = Float32MultiArray()
        tracks_data = []
        for i, tid_raw in enumerate(current_track_ids):
            tid = int(tid_raw)

            # bbox + conf
            x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]
            conf = float(confs[i]) if len(confs) else -1.0

            # depth
            depth = -1.0
            if i < len(depths) and depths[i] is not None:
                depth = float(depths[i])

            # IP outputs
            ip_out = -1.0
            ip_out_f = -1.0
            if tid in self.track_history:
                if self.track_history[tid].get("ip_output"):
                    v = self.track_history[tid]["ip_output"][-1]
                    if isinstance(v, (int, float, np.floating)):
                        ip_out = float(v)
                if self.track_history[tid].get("ip_output_filtered"):
                    v = self.track_history[tid]["ip_output_filtered"][-1]
                    if isinstance(v, (int, float, np.floating)):
                        ip_out_f = float(v)

            tracks_data.extend([float(h), float(w), float(tid), x1, y1, x2, y2, conf, depth, ip_out, ip_out_f])

        tracks_msg.data = tracks_data
        self._pub_tracks.publish(tracks_msg)

        t_total = time.perf_counter() - t_start
        self.get_logger().info(
            f"Frame {self.frame_idx} | {len(current_track_ids)} tracks | "
            f"yolo: {t_infer*1000:.1f}ms, ip: {t_ip*1000:.1f}ms, "
            f"total: {t_total*1000:.1f}ms"
        )
        self.frame_idx += 1


# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(
        description="ROS2 node for HUI-Pred: subscribe to RGB/depth CompressedImage topics, run YOLO pose + interaction prediction.",
    )
    parser.add_argument("--rgb_topic", "-r", type=str, default="/camera/color/image_raw/compressed",
                        help="Topic for RGB CompressedImage")
    parser.add_argument("--depth_topic", "-dt", type=str, default=None,
                        help="Topic for depth CompressedImage (optional; if set, RGB and depth are time-synced)")
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
    parser.add_argument("--overlay_mode", "-om", type=str, choices=["overlay", "eye_animation", "nodrawing"],
                        default="overlay",
                        help="Display mode: 'overlay' = camera image with bbox/skeleton/IP overlay; 'eye_animation' = synthetic eye animation driven by highest-IP person; 'nodrawing' = no drawing and no publishing of image")
    parsed_args, ros_args = parser.parse_known_args(args)

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
    main()
