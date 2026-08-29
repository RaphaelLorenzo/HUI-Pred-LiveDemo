#!/usr/bin/env python3.10
"""
ROS2 node for 3D human pose via YOLO26 tracking + MotionBERT lifting.

Subscribes to RGB, depth, and camera_info, publishes sphere markers for each
H36M joint in the camera optical frame on /huipred/pose3d_markers, and a debug
2D overlay on /huipred/pose3d_debug/image_raw, and the same 3D markers transformed
to base_link on /huipred/pose3d_markers_base_link (via /tf), and yaw-only torso
PoseArray on /huipred/pose3d_poses_base_link.

Run (Docker / ROS Humble, Python 3.10):
    python3.10 ros2_node_3d_pose.py
"""

import argparse
import threading
import time
from functools import partial

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
import torch
import torch.nn as nn
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image as RosImage
from geometry_msgs.msg import Point, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
import message_filters

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None

from ultralytics import YOLO

from predictors.MotionBERT.lib.model.DSTformer import DSTformer
from predictors.MotionBERT.lib.utils.utils_data import crop_scale, flip_data
from utils.other_utils import read_yaml_to_dic
from utils.visualization import get_track_color

# H36M joint indices used for scaling / placement
H36M_LEFT_ANKLE, H36M_RIGHT_ANKLE = 6, 3
H36M_LEFT_SHOULDER, H36M_RIGHT_SHOULDER = 11, 14
H36M_TORSO = 8  # thorax (mid-shoulders)
COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER = 5, 6
MIN_VALID_KEYPOINTS = 5
TARGET_LIMB_LENGTH_M = 1.5
MARKER_DIAMETER_M = 0.10
BONE_LINE_WIDTH_M = 0.03
SEQ_LEN = 15
CONF_ALPHA_MIN = 0.5

# H36M skeleton (MotionBERT vismo convention): L=green, M=blue, R=red
H36M_LIMB_SEQ = [
    [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
    [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16],
]
COLOR_LEFT = (0.0, 1.0, 0.0)
COLOR_MID = (0.0, 0.0, 1.0)
COLOR_RIGHT = (1.0, 0.0, 0.0)
H36M_JOINT_COLORS = [
    COLOR_MID, COLOR_RIGHT, COLOR_RIGHT, COLOR_RIGHT,
    COLOR_LEFT, COLOR_LEFT, COLOR_LEFT,
    COLOR_MID, COLOR_MID, COLOR_MID, COLOR_MID,
    COLOR_LEFT, COLOR_LEFT, COLOR_LEFT,
    COLOR_RIGHT, COLOR_RIGHT, COLOR_RIGHT,
]
H36M_LIMB_COLORS = [
    COLOR_RIGHT, COLOR_RIGHT, COLOR_RIGHT,
    COLOR_LEFT, COLOR_LEFT, COLOR_LEFT,
    COLOR_MID, COLOR_MID, COLOR_MID,
    COLOR_LEFT, COLOR_RIGHT, COLOR_MID,
    COLOR_LEFT, COLOR_LEFT, COLOR_RIGHT, COLOR_RIGHT,
]

COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
    [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
]


def confidence_to_alpha(conf: float) -> float:
    """Map 2D keypoint confidence to display alpha; zero below CONF_ALPHA_MIN."""
    if conf < CONF_ALPHA_MIN:
        return 0.0
    return min(1.0, (conf - CONF_ALPHA_MIN) / (1.0 - CONF_ALPHA_MIN))


def is_valid_coco_detection(scores: np.ndarray, kp_thresh: float) -> bool:
    """Require both shoulders and at least MIN_VALID_KEYPOINTS confident joints."""
    if scores[COCO_LEFT_SHOULDER] <= kp_thresh or scores[COCO_RIGHT_SHOULDER] <= kp_thresh:
        return False
    return int((scores > kp_thresh).sum()) >= MIN_VALID_KEYPOINTS


def dim_bgr_color(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    """Scale BGR color brightness by alpha while keeping the same hue as the bbox."""
    b, g, r = color
    return int(b * alpha), int(g * alpha), int(r * alpha)


def draw_person_2d(
    debug: np.ndarray,
    kp: np.ndarray,
    sc: np.ndarray,
    color: tuple[int, int, int],
):
    """Draw one person's bbox-matching skeleton directly (no alpha blend onto camera)."""
    for j, k in COCO_SKELETON:
        line_alpha = confidence_to_alpha(min(float(sc[j]), float(sc[k])))
        if line_alpha <= 0.0:
            continue
        pt1 = tuple(kp[j].astype(int))
        pt2 = tuple(kp[k].astype(int))
        cv2.line(debug, pt1, pt2, dim_bgr_color(color, line_alpha), 2)
    for kp_pt, score in zip(kp, sc):
        kp_alpha = confidence_to_alpha(float(score))
        if kp_alpha <= 0.0:
            continue
        cv2.circle(debug, (int(kp_pt[0]), int(kp_pt[1])), 4, dim_bgr_color(color, kp_alpha), -1)


def h36m_confidences_from_coco(scores: np.ndarray) -> np.ndarray:
    """Map COCO 17 confidences to H36M 17 (same pairing as coco2h36m)."""
    c = np.zeros(17, dtype=np.float32)
    c[0] = 0.5 * (scores[11] + scores[12])
    c[1] = scores[12]
    c[2] = scores[14]
    c[3] = scores[16]
    c[4] = scores[11]
    c[5] = scores[13]
    c[6] = scores[15]
    c[8] = 0.5 * (scores[5] + scores[6])
    c[7] = 0.5 * (c[0] + c[8])
    c[9] = scores[0]
    c[10] = 0.5 * (scores[1] + scores[2])
    c[11] = scores[5]
    c[12] = scores[7]
    c[13] = scores[9]
    c[14] = scores[6]
    c[15] = scores[8]
    c[16] = scores[10]
    return c


def coco2h36m(x: np.ndarray) -> np.ndarray:
    """Convert COCO 17-keypoint sequence to H36M. x: (T, 17, C)."""
    y = np.zeros_like(x)
    y[:, 0, :] = (x[:, 11, :] + x[:, 12, :]) * 0.5
    y[:, 1, :] = x[:, 12, :]
    y[:, 2, :] = x[:, 14, :]
    y[:, 3, :] = x[:, 16, :]
    y[:, 4, :] = x[:, 11, :]
    y[:, 5, :] = x[:, 13, :]
    y[:, 6, :] = x[:, 15, :]
    y[:, 8, :] = (x[:, 5, :] + x[:, 6, :]) * 0.5
    y[:, 7, :] = (y[:, 0, :] + y[:, 8, :]) * 0.5
    y[:, 9, :] = x[:, 0, :]
    y[:, 10, :] = (x[:, 1, :] + x[:, 2, :]) * 0.5
    y[:, 11, :] = x[:, 5, :]
    y[:, 12, :] = x[:, 7, :]
    y[:, 13, :] = x[:, 9, :]
    y[:, 14, :] = x[:, 6, :]
    y[:, 15, :] = x[:, 8, :]
    y[:, 16, :] = x[:, 10, :]
    return y


def estimate_torso_depth_and_center(keypoints, scores, depth_image, kp_thresh):
    """Median depth and pixel location from torso diagonal samples."""
    left_shoulder, right_shoulder = 5, 6
    left_hip, right_hip = 11, 12
    sample_ratios = np.linspace(1.0 / 6.0, 5.0 / 6.0, 5)
    depth_samples, sample_points = [], []
    h, w = depth_image.shape[:2]

    if scores[left_shoulder] > kp_thresh and scores[right_hip] > kp_thresh:
        pt1, pt2 = keypoints[left_shoulder], keypoints[right_hip]
        for r in sample_ratios:
            x, y = int(pt1[0] + r * (pt2[0] - pt1[0])), int(pt1[1] + r * (pt2[1] - pt1[1]))
            if 0 <= x < w and 0 <= y < h:
                depth_value = float(depth_image[y, x])
                if np.isfinite(depth_value) and depth_value > 0:
                    depth_samples.append(depth_value)
                    sample_points.append((float(x), float(y)))

    if scores[right_shoulder] > kp_thresh and scores[left_hip] > kp_thresh:
        pt1, pt2 = keypoints[right_shoulder], keypoints[left_hip]
        for r in sample_ratios:
            x, y = int(pt1[0] + r * (pt2[0] - pt1[0])), int(pt1[1] + r * (pt2[1] - pt1[1]))
            if 0 <= x < w and 0 <= y < h:
                depth_value = float(depth_image[y, x])
                if np.isfinite(depth_value) and depth_value > 0:
                    depth_samples.append(depth_value)
                    sample_points.append((float(x), float(y)))

    if len(depth_samples) < 3:
        return None, None

    torso_center_uv = np.median(np.array(sample_points, dtype=np.float32), axis=0)
    torso_depth = float(np.median(np.array(depth_samples, dtype=np.float32)))
    return torso_depth, torso_center_uv


def project_pixel_to_3d(pixel_xy, depth_m, camera_matrix):
    """Project one pixel + depth into camera-frame 3D coordinates."""
    if pixel_xy is None or depth_m is None or depth_m <= 0 or camera_matrix is None:
        return None
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    if fx == 0.0 or fy == 0.0:
        return None
    u, v = float(pixel_xy[0]), float(pixel_xy[1])
    return np.array([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m], dtype=np.float32)


def build_padded_sequence(pose_history: list, seq_len: int) -> np.ndarray:
    """Last seq_len COCO poses; pad start by repeating the oldest frame in the window."""
    seq = pose_history[-seq_len:]
    if len(seq) < seq_len:
        seq = [seq[0]] * (seq_len - len(seq)) + seq
    return np.stack(seq, axis=0)


def scale_skeleton_to_meters(joints_3d: np.ndarray) -> np.ndarray:
    """Scale root-relative H36M skeleton so ankle-shoulder limbs are ~1.5 m."""
    factors = []
    left_len = np.linalg.norm(joints_3d[H36M_LEFT_ANKLE] - joints_3d[H36M_LEFT_SHOULDER])
    right_len = np.linalg.norm(joints_3d[H36M_RIGHT_ANKLE] - joints_3d[H36M_RIGHT_SHOULDER])
    if left_len > 1e-6:
        factors.append(TARGET_LIMB_LENGTH_M / left_len)
    if right_len > 1e-6:
        factors.append(TARGET_LIMB_LENGTH_M / right_len)
    if not factors:
        return joints_3d
    return joints_3d * float(np.mean(factors))


def place_skeleton_in_camera(joints_3d: np.ndarray, torso_3d_target: np.ndarray) -> np.ndarray:
    """Translate scaled skeleton so H36M thorax sits on the depth-based torso point."""
    offset = torso_3d_target - joints_3d[H36M_TORSO]
    return joints_3d + offset


def transform_points(transform, points: np.ndarray) -> np.ndarray:
    """Apply a geometry_msgs/Transform to Nx3 points."""
    t = transform.transform.translation
    q = transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    offset = np.array([t.x, t.y, t.z], dtype=np.float64)
    return (rot @ points.astype(np.float64).T).T + offset


def yaw_from_shoulders_base(joints_base: np.ndarray) -> float | None:
    """Yaw in base_link (z up) from H36M left/right shoulder positions."""
    width = joints_base[H36M_RIGHT_SHOULDER] - joints_base[H36M_LEFT_SHOULDER]
    width_xy = width.copy()
    width_xy[2] = 0.0
    norm = float(np.linalg.norm(width_xy))
    if norm < 1e-6:
        return None
    width_xy /= norm
    forward = np.cross(np.array([0.0, 0.0, 1.0]), width_xy)
    return float(np.arctan2(forward[1], forward[0]))


def joint_marker_id(tid: int, joint_idx: int) -> int:
    return tid * 100 + joint_idx


def bone_marker_id(tid: int, bone_idx: int) -> int:
    return tid * 100 + 20 + bone_idx


class Pose3DNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("pose_3d_node")

        self.kp_thresh = args.kp_thresh
        self.depth_scale = args.depth_scale
        self.seq_len = args.seq_len
        self._camera_matrix = None
        self._camera_frame_id = "camera_optical_frame"
        self._process_lock = threading.Lock()
        self._cv_bridge = CvBridge() if CvBridge is not None else None
        self.track_poses: dict[int, list] = {}  # track_id -> list of (17,3) COCO keypoints
        self._active_marker_ids: set[int] = set()
        self._active_marker_ids_base: set[int] = set()
        self._base_link_frame = args.base_link_frame
        self._tf_timeout = args.tf_timeout

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        depth_topic = args.depth_topic
        self.compressed_depth = (
            depth_topic.endswith("/compressed")
            or depth_topic.endswith("/compressedDepth")
            or "/compressedDepth/" in depth_topic
        )

        self.get_logger().info(f"Loading YOLO from {args.yolo_model_path}")
        self.pose_model = YOLO(args.yolo_model_path)

        self.get_logger().info(f"Loading MotionBERT from {args.motionbert_checkpoint}")
        mb_config = read_yaml_to_dic(args.motionbert_config)
        self.mb_model = DSTformer(
            dim_in=3,
            dim_out=3,
            dim_feat=mb_config["dim_feat"],
            dim_rep=mb_config["dim_rep"],
            depth=mb_config["depth"],
            num_heads=mb_config["num_heads"],
            mlp_ratio=mb_config["mlp_ratio"],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            maxlen=mb_config["maxlen"],
            num_joints=mb_config["num_joints"],
        )
        checkpoint = torch.load(args.motionbert_checkpoint, map_location="cpu")
        state_dict = checkpoint["model_pos"]
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        self.mb_model.load_state_dict(state_dict, strict=True)
        self.mb_model.eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mb_model.to(self._device)
        self.mb_rootrel = bool(mb_config.get("rootrel", True))
        self.mb_flip = bool(mb_config.get("flip", True))

        self._pub_markers = self.create_publisher(MarkerArray, args.marker_topic, 10)
        self._pub_markers_base = self.create_publisher(
            MarkerArray, args.base_link_marker_topic, 10
        )
        self._pub_poses_base = self.create_publisher(PoseArray, args.pose_array_topic, 10)
        self._debug_enabled = args.debug_image_topic.lower() != "none"
        self._pub_debug = None
        if self._debug_enabled:
            self._pub_debug = self.create_publisher(RosImage, args.debug_image_topic, 10)

        self.create_subscription(
            CameraInfo, args.camera_info_topic, self._camera_info_cb, 10
        )
        self._sub_rgb = message_filters.Subscriber(self, CompressedImage, args.rgb_topic)
        if self.compressed_depth:
            self._sub_depth = message_filters.Subscriber(self, CompressedImage, depth_topic)
        else:
            self._sub_depth = message_filters.Subscriber(self, RosImage, depth_topic)
        sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub_rgb, self._sub_depth], queue_size=50, slop=0.5
        )
        sync.registerCallback(self._synced_cb)
        self.get_logger().info(
            f"Ready | RGB={args.rgb_topic} depth={depth_topic} "
            f"markers={args.marker_topic} base={args.base_link_marker_topic} "
            f"poses={args.pose_array_topic} "
            f"debug={'off' if not self._debug_enabled else args.debug_image_topic}"
        )

    def _lookup_camera_to_base(self, stamp):
        try:
            return self._tf_buffer.lookup_transform(
                self._base_link_frame,
                self._camera_frame_id,
                stamp,
                timeout=Duration(seconds=self._tf_timeout),
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _publish_skeleton_markers(
        self,
        pub,
        stamp,
        frame_id: str,
        ns: str,
        skeletons: list,
        active_ids: set,
    ):
        """skeletons: list of (tid, joints_3d (17,3), h36m_conf (17,))."""
        marker_array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        clear.ns = ns
        clear.header.stamp = stamp
        clear.header.frame_id = frame_id
        marker_array.markers.append(clear)

        new_ids: set[int] = set()
        for tid, joints_3d, h36m_conf in skeletons:
            for j_idx, xyz in enumerate(joints_3d):
                alpha = confidence_to_alpha(float(h36m_conf[j_idx]))
                if alpha <= 0.0:
                    continue
                mid = joint_marker_id(tid, j_idx)
                new_ids.add(mid)
                r, g, b = H36M_JOINT_COLORS[j_idx]
                m = Marker()
                m.header.stamp = stamp
                m.header.frame_id = frame_id
                m.ns = ns
                m.id = mid
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(xyz[0])
                m.pose.position.y = float(xyz[1])
                m.pose.position.z = float(xyz[2])
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = MARKER_DIAMETER_M
                m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, alpha
                marker_array.markers.append(m)

            for bone_idx, (j1, j2) in enumerate(H36M_LIMB_SEQ):
                alpha = confidence_to_alpha(min(float(h36m_conf[j1]), float(h36m_conf[j2])))
                if alpha <= 0.0:
                    continue
                mid = bone_marker_id(tid, bone_idx)
                new_ids.add(mid)
                p1, p2 = joints_3d[j1], joints_3d[j2]
                r, g, b = H36M_LIMB_COLORS[bone_idx]
                m = Marker()
                m.header.stamp = stamp
                m.header.frame_id = frame_id
                m.ns = ns
                m.id = mid
                m.type = Marker.LINE_LIST
                m.action = Marker.ADD
                m.points = [
                    Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])),
                    Point(x=float(p2[0]), y=float(p2[1]), z=float(p2[2])),
                ]
                m.scale.x = BONE_LINE_WIDTH_M
                m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, alpha
                marker_array.markers.append(m)

        for mid in active_ids - new_ids:
            del_m = Marker()
            del_m.header.stamp = stamp
            del_m.header.frame_id = frame_id
            del_m.ns = ns
            del_m.id = mid
            del_m.action = Marker.DELETE
            marker_array.markers.append(del_m)

        pub.publish(marker_array)
        active_ids.clear()
        active_ids.update(new_ids)

    def _publish_pose_array(self, stamp, skeletons_base: list):
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self._base_link_frame
        for _, joints_base, _ in skeletons_base:
            yaw = yaw_from_shoulders_base(joints_base)
            if yaw is None:
                continue
            torso = joints_base[H36M_TORSO]
            pose = Pose()
            pose.position.x = float(torso[0])
            pose.position.y = float(torso[1])
            pose.position.z = float(torso[2])
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = float(np.sin(yaw / 2.0))
            pose.orientation.w = float(np.cos(yaw / 2.0))
            msg.poses.append(pose)
        self._pub_poses_base.publish(msg)

    def _camera_info_cb(self, msg: CameraInfo):
        self._camera_matrix = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self._camera_frame_id = msg.header.frame_id

    @staticmethod
    def _decode_compressed_rgb(msg: CompressedImage):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    @staticmethod
    def _extract_png_payload(raw: bytes):
        png_magic = b"\x89PNG\r\n\x1a\n"
        png_offset = raw.find(png_magic)
        if png_offset >= 0:
            return raw[png_offset:]
        if len(raw) > 12 and raw[12:].startswith(png_magic):
            return raw[12:]
        return None

    def _decode_compressed_depth(self, msg: CompressedImage):
        raw = bytes(msg.data)
        if self._cv_bridge is not None:
            try:
                depth = self._cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if depth is not None and depth.size > 0:
                    return depth[:, :, 0] if depth.ndim == 3 else depth
            except Exception:
                pass
        png_payload = self._extract_png_payload(raw)
        if png_payload is None:
            return None
        depth = cv2.imdecode(np.frombuffer(png_payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        return depth[:, :, 0] if depth.ndim == 3 else depth

    @staticmethod
    def _decode_depth_image(msg: RosImage):
        enc = (msg.encoding or "").lower()
        if enc in ("16uc1", "mono16"):
            dtype = np.dtype(np.uint16)
        elif enc == "32fc1":
            dtype = np.dtype(np.float32)
        else:
            return None
        if msg.is_bigendian:
            dtype = dtype.newbyteorder(">")
        else:
            dtype = dtype.newbyteorder("<")
        row_stride = msg.step if msg.step > 0 else msg.width * dtype.itemsize
        arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row_stride // dtype.itemsize)
        return np.ascontiguousarray(arr[:, :msg.width])

    def _synced_cb(self, rgb_msg: CompressedImage, depth_msg):
        if not self._process_lock.acquire(blocking=False):
            return
        try:
            bgr = self._decode_compressed_rgb(rgb_msg)
            depth = (
                self._decode_compressed_depth(depth_msg)
                if self.compressed_depth
                else self._decode_depth_image(depth_msg)
            )
            if bgr is None or depth is None:
                return
            self._process_frame(bgr, depth, rgb_msg.header.stamp)
        finally:
            self._process_lock.release()

    def _motionbert_infer(self, batch_input: np.ndarray) -> np.ndarray:
        """batch_input: (N, T, 17, 3) normalized H36M 2D poses -> (N, 17, 3) last frame."""
        tensor = torch.from_numpy(batch_input).to(self._device)
        with torch.no_grad():
            if self.mb_flip:
                pred1 = self.mb_model(tensor)
                pred2 = flip_data(self.mb_model(flip_data(tensor)))
                pred = (pred1 + pred2) / 2.0
            else:
                pred = self.mb_model(tensor)
            if self.mb_rootrel:
                pred[:, :, 0, :] = 0.0
        return pred[:, -1].cpu().numpy()

    def _publish_debug_image(self, stamp, frame_id: str, bgr: np.ndarray):
        if not self._debug_enabled or self._pub_debug is None:
            return
        img = np.ascontiguousarray(bgr, dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            return
        msg = RosImage()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = img.shape[1] * 3
        msg.data = img.tobytes()
        self._pub_debug.publish(msg)

    def _process_frame(self, bgr: np.ndarray, depth_raw: np.ndarray, stamp):
        t0 = time.perf_counter()
        results = self.pose_model.track(bgr, persist=True, verbose=False, tracker="bytetrack.yaml")[0]

        debug = bgr.copy() if self._debug_enabled else None

        if results.boxes.id is None:
            self._publish_skeleton_markers(
                self._pub_markers, stamp, self._camera_frame_id, "pose3d",
                [], self._active_marker_ids,
            )
            self._publish_skeleton_markers(
                self._pub_markers_base, stamp, self._base_link_frame, "pose3d_base",
                [], self._active_marker_ids_base,
            )
            self._publish_pose_array(stamp, [])
            if debug is not None:
                self._publish_debug_image(stamp, self._camera_frame_id, debug)
            return

        track_ids = results.boxes.id.int().cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        box_confs = results.boxes.conf.cpu().numpy()
        keypoints_all = results.keypoints.xy.cpu().numpy()
        scores_all = results.keypoints.conf.cpu().numpy()

        # One entry per track id (YOLO can occasionally repeat an id in one frame).
        best_idx_for_tid: dict[int, int] = {}
        for i, tid_raw in enumerate(track_ids):
            tid = int(tid_raw)
            if tid not in best_idx_for_tid or box_confs[i] > box_confs[best_idx_for_tid[tid]]:
                best_idx_for_tid[tid] = i
        frame_indices = [
            i for i in best_idx_for_tid.values()
            if is_valid_coco_detection(scores_all[i], self.kp_thresh)
        ]
        track_colors = {int(track_ids[i]): get_track_color(int(track_ids[i])) for i in frame_indices}

        if debug is not None:
            for i in frame_indices:
                tid = int(track_ids[i])
                color = track_colors[tid]
                x1, y1, x2, y2 = boxes[i].astype(int)
                cv2.rectangle(debug, (x1, y1), (x2, y2), color, 3)
                label = f"ID:{tid}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                (_, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.putText(
                    debug, label, (x1, max(y1 - 8, th + 4)),
                    font, font_scale, color, thickness,
                )
                draw_person_2d(debug, keypoints_all[i], scores_all[i], color)

        infer_ids, infer_inputs, torso_targets, infer_scores = [], [], [], []

        for i in frame_indices:
            tid = int(track_ids[i])
            kp_xy = keypoints_all[i]
            kp_sc = scores_all[i]
            coco_pose = np.concatenate([kp_xy, kp_sc[:, None]], axis=1).astype(np.float32)

            if tid not in self.track_poses:
                self.track_poses[tid] = []
            self.track_poses[tid].append(coco_pose)

            torso_depth_raw, torso_uv = estimate_torso_depth_and_center(
                kp_xy, kp_sc, depth_raw, self.kp_thresh
            )
            if torso_depth_raw is None or self._camera_matrix is None:
                continue
            torso_depth_m = torso_depth_raw / self.depth_scale
            torso_3d = project_pixel_to_3d(torso_uv, torso_depth_m, self._camera_matrix)
            if torso_3d is None:
                continue

            seq = build_padded_sequence(self.track_poses[tid], self.seq_len)
            h36m_seq = coco2h36m(seq)
            h36m_seq = crop_scale(h36m_seq, scale_range=[1, 1])
            if np.allclose(h36m_seq, 0):
                continue

            infer_ids.append(tid)
            infer_inputs.append(h36m_seq)
            torso_targets.append(torso_3d)
            infer_scores.append(kp_sc)

        skeletons_cam = []
        if infer_inputs:
            batch = np.stack(infer_inputs, axis=0)
            joints_batch = self._motionbert_infer(batch)

            for tid, joints_3d, torso_3d, kp_sc in zip(
                infer_ids, joints_batch, torso_targets, infer_scores
            ):
                joints_3d = scale_skeleton_to_meters(joints_3d)
                joints_3d = place_skeleton_in_camera(joints_3d, torso_3d)
                h36m_conf = h36m_confidences_from_coco(kp_sc)
                skeletons_cam.append((tid, joints_3d, h36m_conf))

        self._publish_skeleton_markers(
            self._pub_markers, stamp, self._camera_frame_id, "pose3d",
            skeletons_cam, self._active_marker_ids,
        )

        skeletons_base = []
        camera_to_base = self._lookup_camera_to_base(stamp)
        if camera_to_base is not None and skeletons_cam:
            for tid, joints_3d, h36m_conf in skeletons_cam:
                joints_base = transform_points(camera_to_base, joints_3d)
                skeletons_base.append((tid, joints_base, h36m_conf))
        elif camera_to_base is None and skeletons_cam:
            self.get_logger().warning(
                f"No TF {self._camera_frame_id} -> {self._base_link_frame}; "
                "skipping base_link markers and poses",
                throttle_duration_sec=5.0,
            )

        self._publish_skeleton_markers(
            self._pub_markers_base, stamp, self._base_link_frame, "pose3d_base",
            skeletons_base, self._active_marker_ids_base,
        )
        self._publish_pose_array(stamp, skeletons_base)

        if debug is not None:
            self._publish_debug_image(stamp, self._camera_frame_id, debug)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.get_logger().info(
            f"{len(frame_indices)} valid tracks | {len(skeletons_cam)} posed | {dt_ms:.1f} ms"
        )


def main(parsed_args, ros_args):
    rclpy.init(args=ros_args)
    node = Pose3DNode(parsed_args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ROS2 node: YOLO26 2D pose + MotionBERT 3D lifting -> joint markers.",
    )
    parser.add_argument(
        "--rgb_topic", type=str,
        default="/rgbd/realsense_head_front/color/image_raw/compressed",
    )
    parser.add_argument(
        "--depth_topic", type=str,
        default="/rgbd/realsense_head_front/aligned_depth_to_color/image_raw/compressedDepth",
    )
    parser.add_argument(
        "--camera_info_topic", type=str,
        default="/rgbd/realsense_head_front/color/camera_info",
    )
    parser.add_argument(
        "--yolo_model_path", type=str, default="checkpoints/yolo26x-pose.pt",
    )
    parser.add_argument(
        "--motionbert_checkpoint", type=str, default="checkpoints/mb_pose_estimation_ft.bin",
    )
    parser.add_argument(
        "--motionbert_config", type=str,
        default="predictors/MotionBERT/configs/pose3d/MB_ft_h36m.yaml",
    )
    parser.add_argument(
        "--marker_topic", type=str, default="/huipred/pose3d_markers",
    )
    parser.add_argument(
        "--base_link_marker_topic", type=str,
        default="/huipred/pose3d_markers_base_link",
    )
    parser.add_argument(
        "--base_link_frame", type=str, default="base_link",
    )
    parser.add_argument(
        "--pose_array_topic", type=str, default="/huipred/pose3d_poses_base_link",
    )
    parser.add_argument(
        "--tf_timeout", type=float, default=0.1,
        help="Seconds to wait for camera->base_link TF lookup",
    )
    parser.add_argument(
        "--debug_image_topic", type=str, default="/huipred/pose3d_debug/image_raw",
        help='Debug overlay topic, or "none" to disable',
    )
    parser.add_argument("--kp_thresh", type=float, default=0.3)
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parsed_args, ros_args = parser.parse_known_args()
    main(parsed_args, ros_args)
