#!/usr/bin/env python3.10
"""
ROS2 node for real-time human-robot interaction prediction using YOLO pose estimation.

Subscribes to CompressedImage topics for RGB and depth, runs YOLO pose detection
+ tracking, and publishes bounding box, track_id, and predicted interaction score
on /huipred/detections.

In Docker (ROS Humble uses Python 3.10), run with:
    python3.10 ros2_node_yolo26.py --rgb_topic /camera/color/image_raw/compressed --depth_topic /camera/depth/image_raw/compressed --interaction_prediction_checkpoint checkpoints/mb_FineTuned_28_02_26_best_ap.pth
"""

import argparse
import types
import time
import cv2
from PIL import Image
from functools import partial

import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
import message_filters

from ultralytics import YOLO
from utils.geometry_utils import rot_matrix_torch

from predictors.mlp import MLPInteractionPredictor
from predictors.lstm import LSTMInteractionPredictor
from predictors.MotionBERT.lib.model.DSTformer import DSTformer
from predictors.MotionBERT.lib.model.model_action import ActionNet
from predictors.STG_NF.model_pose import STG_NF
from predictors.STGCN.net.st_gcn import Model as STGCN
from predictors.SkateFormer.model.SkateFormer import SkateFormer

import sys
import numpy as np

# Create a bridge between NumPy 1.x and 2.x naming conventions
# if not hasattr(np, "_core"):
#     sys.modules["numpy._core"] = np.core

# ---------------------------------------------------------------------------
# Behavior prototype thresholds
# ---------------------------------------------------------------------------
INITIAL_INTERACTION_PREDICTION_THRESHOLD = 0.1
INITIAL_MIN_FRAMES_ENGAGED = 2
BREAKUP_FRAMES_DISENGAGED = 10
MIN_LIGHTS_SCALE_THRESHOLD = 0.1
MAX_LIGHTS_SCALE_THRESHOLD = 0.8

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

    if "select_input_range" in config and config["select_input_range"] != [[0, -1]]:
        sequence_length = config["select_input_range"][1] - config["select_input_range"][0]
    else:
        sequence_length = config["input_length_in_frames"] // config["subsample_frames"]

    assert config["subsample_frames"] == 1, "Subsampling not supported for live prediction"

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
    """Run ActionNet interaction prediction on current tracks."""
    assert config["mb_input_norm"] == "vid"
    min_valid_keypoints = config["min_keypoints_filter"]
    input_length_in_frames = 16
    max_index_gap_allowed = 2

    eq_w, eq_h = (3840, 1920) if backprojection else image_size
    whtensor = torch.tensor([eq_w, eq_h], device=device)

    return_dict = {tid: "NC" for tid in current_tracks_history}
    valid_ids, valid_input_tensors = [], []

    for track_id, track_history in current_tracks_history.items():
        indexes = track_history["indexes"]
        input_tensors = track_history["ip_input_tensor"]
        if len(indexes) < input_length_in_frames:
            return_dict[track_id] = "not_enough_frames"
            continue
        last_idx = np.array(indexes[-input_length_in_frames:])
        if np.any(np.diff(last_idx) > max_index_gap_allowed):
            return_dict[track_id] = "index_gap_too_large"
            continue
        last_t = torch.stack(input_tensors[-input_length_in_frames:], dim=0)
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
        yolo_path = args.yolo_model_path
        ip_ckpt = args.interaction_prediction_checkpoint or ""
        self._use_depth = depth_topic != ""

        # Args-like namespace consumed by helper functions
        self._args = types.SimpleNamespace(
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
            self.get_logger().info(f"Loading IP checkpoint from {ip_ckpt}")
            ckpt = torch.load(ip_ckpt, map_location="cpu", weights_only=False)
            if "model_state_dict" not in ckpt:
                raise ValueError("Checkpoint missing 'model_state_dict'")
            if "config" not in ckpt:
                if "hyperparameters" in ckpt:
                    ckpt["config"] = ckpt["hyperparameters"]
                else:
                    raise ValueError("Checkpoint missing 'config'/'hyperparameters'")
            self.ip_config = ckpt["config"]
            self.get_logger().info(
                f"IP model type: {self.ip_config['force_model_type']} | "
                f"AUC {ckpt.get('val_auc', 0):.4f} | AP {ckpt.get('val_ap', 0):.4f}"
            )
            self.ip_model = load_model_from_config(self.ip_config, device="cuda")
            self.ip_model.load_state_dict(ckpt["model_state_dict"], strict=True)
            if not isinstance(self.ip_model, ActionNet):
                raise NotImplementedError(
                    f"{type(self.ip_model).__name__} not yet supported for live IP"
                )
            self.get_logger().info("IP model loaded successfully")

        # -- State --
        self.track_history: dict = {}
        self.frame_idx = 0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -- Publisher: /huipred/detections (bbox, track_id, interaction_score) --
        self._pub_detections = self.create_publisher(
            Detection2DArray, "/huipred/detections", 10
        )

        # -- Subscribers --
        if self._use_depth:
            self.get_logger().info(
                f"Subscribing (synced): RGB={rgb_topic}  Depth={depth_topic}"
            )
            self._sub_rgb = message_filters.Subscriber(self, CompressedImage, rgb_topic)
            self._sub_depth = message_filters.Subscriber(self, CompressedImage, depth_topic)
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._sub_rgb, self._sub_depth], queue_size=10, slop=0.1,
            )
            self._sync.registerCallback(self._synced_cb)
        else:
            self.get_logger().info(f"Subscribing (RGB only): {rgb_topic}")
            self._sub_rgb = self.create_subscription(
                CompressedImage, rgb_topic, self._rgb_only_cb, 10,
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

    # ----- callbacks ----------------------------------------------------------

    def _synced_cb(self, rgb_msg: CompressedImage, depth_msg: CompressedImage):
        bgr = self._decode_compressed_rgb(rgb_msg)
        depth = self._decode_compressed_depth(depth_msg)
        if bgr is None:
            self.get_logger().warn("Failed to decode RGB message")
            return
        self._process_frame(bgr, depth)

    def _rgb_only_cb(self, rgb_msg: CompressedImage):
        bgr = self._decode_compressed_rgb(rgb_msg)
        if bgr is None:
            self.get_logger().warn("Failed to decode RGB message")
            return
        self._process_frame(bgr, None)

    # ----- main per-frame pipeline -------------------------------------------

    def _process_frame(self, bgr: np.ndarray, depth_image: np.ndarray):
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

        # -- Publish detections on /huipred/detections --
        msg = Detection2DArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_optical_frame"
        for i, tid_raw in enumerate(current_track_ids):
            tid = int(tid_raw)
            x1, y1, x2, y2 = boxes[i]
            det = Detection2D()
            det.header = msg.header
            det.bbox.center.x = float((x1 + x2) / 2.0)
            det.bbox.center.y = float((y1 + y2) / 2.0)
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)
            hyp = ObjectHypothesisWithPose()
            hyp.class_id = str(tid)
            if tid in self.track_history and self.track_history[tid]["ip_output"]:
                v = self.track_history[tid]["ip_output"][-1]
                hyp.score = float(v) if isinstance(v, (int, float)) else -1.0
            else:
                hyp.score = -1.0
            det.results.append(hyp)
            msg.detections.append(det)
        self._pub_detections.publish(msg)

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
    parser.add_argument("--interaction_prediction_checkpoint", "-ip", type=str, default="checkpoints/mb_FineTuned_28_02_26_best_ap.pth",
                        help="Path to interaction prediction checkpoint (optional)")
    parser.add_argument("--kp_thresh", type=float, default=0.3,
                        help="Keypoint confidence threshold")
    parser.add_argument("--depth_scale", type=float, default=1000.0,
                        help="Depth scale factor (depth units per meter, e.g. 1000 for mm)")
    parser.add_argument("--backprojection", "-bp", action="store_true", default=False,
                        help="Backproject joints from perspective to equirect before IP inference")
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
