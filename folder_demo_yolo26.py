import argparse
import os
import time
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# # Load model
pose_model = YOLO("checkpoints/yolo26x-pose.pt")

# results_test = pose_model("frame_000146.jpg", verbose=True)[0]
# print(f"Results test: {results_test.boxes.id}")
# exit()

import torch
import torch.nn as nn
from utils.visualization import InfoPanel, BehaviorPanel, get_track_color, concatenate_with_panel
from utils.geometry_utils import rot_matrix_torch
from functools import partial

from predictors.mlp import MLPInteractionPredictor
from predictors.lstm import LSTMInteractionPredictor
from predictors.MotionBERT.lib.model.DSTformer import DSTformer
from predictors.MotionBERT.lib.model.model_action import ActionNet
from predictors.STG_NF.model_pose import STG_NF
from predictors.STGCN.net.st_gcn import Model as STGCN
from predictors.SkateFormer.model.SkateFormer import SkateFormer

from utils.other_utils import read_yaml_to_dic

# -----------------------------------------------------------------------------
# Behavior prototype thresholds (for "someone wants to interact" detection)
# -----------------------------------------------------------------------------
# Threshold above which a person is considered potentially engaged
INITIAL_INTERACTION_PREDICTION_THRESHOLD = 0.1
# Minimum consecutive frames above threshold to be considered "engaged"
INITIAL_MIN_FRAMES_ENGAGED = 2
# Consecutive frames below threshold to be considered "disengaged"
BREAKUP_FRAMES_DISENGAGED = 10
# Engagement score range used to scale lights: below MIN = idle (blue), above MAX = offering (green)
MIN_LIGHTS_SCALE_THRESHOLD = 0.1
MAX_LIGHTS_SCALE_THRESHOLD = 0.8

# COCO pose skeleton connections (17 keypoints format)
COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], 
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], 
    [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
]


def estimate_torso_depth(keypoints: np.ndarray, scores: np.ndarray, depth_image: np.ndarray, kp_thresh: float):
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
        return -1.0  # Sentinel value for no valid samples
    
    return float(np.median(depth_samples))


METADATA_CKPT_COLUMNS = ["recording", 
                    "episode", 
                    "image_height", 
                    "image_width", 
                    "unique_track_identifier", 
                    "track_id", 
                    "image_file", 
                    "image_index", 
                    "validity", 
                    "current_segment", 
                    "total_segments", 
                    "position_in_segment", 
                    "length_of_current_segment", 
                    "timestamp", 
                    "timestamp_sec", 
                    "timestamp_track", 
                    "engagement", 
                    "time_to_first_interaction", 
                    "mask_rle"
                    ]

def load_model_from_config(config: dict, device: torch.device) -> torch.nn.Module:
    
    include_columns = config["include_columns"]
    data_columns = [col for col in include_columns if col not in METADATA_CKPT_COLUMNS]
    input_dim = len(data_columns)
    
    print(f"Input dimension: {input_dim} | Data columns: {data_columns}")

    if "select_input_range" in config and config["select_input_range"] != [[0,-1]]:
        sequence_length = config["select_input_range"][1] - config["select_input_range"][0]
    else:
        sequence_length = config["input_length_in_frames"] // config["subsample_frames"]
    
    assert(config["subsample_frames"] == 1), "Subsampling frames is not supported for live interaction prediction"
    
    # Instantiate model
    if config["force_model_type"] == "mlp":
        model = MLPInteractionPredictor(
            input_dim=input_dim,
            sequence_length=sequence_length,
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"]
        ).to(device)
        
    elif config["force_model_type"] == "lstm":
        model = LSTMInteractionPredictor(
            input_dim=input_dim,
            sequence_length=sequence_length,
            hidden_dim=config["lstm_hidden_dim"],
            num_layers=config["lstm_num_layers"],
            dropout=config["lstm_dropout"],
            bidirectional=False
        ).to(device)
        
    elif config["force_model_type"].lower().startswith("motionbert") or config["force_model_type"].lower().startswith("mb"):
        # Default parameters from pretrain/MB_pretrain.yaml
        backbone = DSTformer(
            dim_in=3, 
            dim_out=3, # for the 3D reconstruction, but we will not use it if we return the representation
            dim_feat=256 if "_lite" in config["force_model_type"].lower() else 512, 
            dim_rep=512, 
            depth=5, 
            num_heads=8, 
            mlp_ratio=4 if "_lite" in config["force_model_type"].lower() else 2, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            maxlen=243, 
            num_joints=17,
            desired_return="representation"
        )
        
        # Will directly load the full weights from the checkpoint don't care if pretrained, finetuned or not

        model = ActionNet(backbone=backbone, 
                          dim_rep=512, 
                          num_classes=1, 
                          dropout_ratio=config["mb_head_dropout"], 
                          version='class', 
                          hidden_dim=config["mb_head_hidden_dim"], 
                          num_joints=17).to(device)

    elif config["force_model_type"] == "stg_nf":
        model = STG_NF(device=device,
                        pose_shape=(2, sequence_length, 18),
                        hidden_channels=config["stg_nf_hidden_channels"],
                        K=config["stg_nf_K"],
                        L=config["stg_nf_L"],
                        R=config["stg_nf_R"],
                        actnorm_scale=config["stg_nf_actnorm_scale"],
                        flow_permutation="permute",
                        flow_coupling="affine",
                        LU_decomposed=True,
                        learn_top=False,
                        edge_importance=config["stg_nf_edge_importance"],
                        temporal_kernel_size=None,
                        strategy="uniform",
                        max_hops=config["stg_nf_max_hops"],).to(device)

    elif config["force_model_type"] == "stgcn":
        model = STGCN(
            in_channels=config["stgcn_in_channels"],
            num_class=1,
            graph_args={"layout": config["stgcn_layout"], "strategy": 'spatial'},
            edge_importance_weighting=config["stgcn_edge_importance_weighting"],
        ).to(device)
        
    elif config["force_model_type"] == "skateformer":
        assert(sequence_length % 8 == 0), "Sequence length must be divisible by 8 for SkateFormer"
        Tdim = sequence_length // 8
        ncolumns_input = len(config["include_columns"]) 
        # 74 = D3 (ViTPose) will be mapped to NW-UCLA (20 joints) (in this case at loading, D=56)
        # 263 = D8 (ViTPose + Sapiens) will be mapped to NTU without spine (24 joints) (in this case at loading, D=245)
        # 212 = D9 (Sapiens) will be mapped to NTU without spine (24 joints) (in this case at loading, D=194)
        if ncolumns_input == 74:
            num_joints_mapped = 20
            types_spatial_sizes = [(Tdim, 4), (Tdim, 5), (Tdim, 4), (Tdim, 5)]
        elif ncolumns_input == 263 or ncolumns_input == 212:
            num_joints_mapped = 24
            types_spatial_sizes = [(Tdim, 8), (Tdim, 12), (Tdim, 8), (Tdim, 12)]
        else:
            raise ValueError(f"Invalid number of input columns: {ncolumns_input} (expect 74, 263 or 212 i.e. D3, D8 or D9)")

        model = SkateFormer(
            in_channels=config["skateformer_in_channels"],
            depths=(2, 2, 2, 2),
            channels=(96, 192, 192, 192),
            num_classes=1,
            embed_dim=96,
            num_people=1,
            num_points=num_joints_mapped,
            kernel_size=7,
            num_heads=32,
            attn_drop=0.5,
            head_drop=0.0,
            rel=True,
            drop_path=0.2,
            type_1_size=types_spatial_sizes[0],
            type_2_size=types_spatial_sizes[1],
            type_3_size=types_spatial_sizes[2],
            type_4_size=types_spatial_sizes[3],
            mlp_ratio=1.0,
            index_t=True,
        ).to(device)
        
    else:
        raise ValueError(f"Invalid model type: {config['force_model_type']}")

    model.eval()
    return model


def coco2h36m(x):
    '''
        Input: x (M x T x V x C) or (B, T, V, C)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = torch.zeros(x.shape, device=x.device)
    y[:,:,0,:] = (x[:,:,11,:] + x[:,:,12,:]) * 0.5
    y[:,:,1,:] = x[:,:,12,:]
    y[:,:,2,:] = x[:,:,14,:]
    y[:,:,3,:] = x[:,:,16,:]
    y[:,:,4,:] = x[:,:,11,:]
    y[:,:,5,:] = x[:,:,13,:]
    y[:,:,6,:] = x[:,:,15,:]
    y[:,:,8,:] = (x[:,:,5,:] + x[:,:,6,:]) * 0.5
    y[:,:,7,:] = (y[:,:,0,:] + y[:,:,8,:]) * 0.5
    y[:,:,9,:] = x[:,:,0,:]
    y[:,:,10,:] = (x[:,:,1,:] + x[:,:,2,:]) * 0.5
    y[:,:,11,:] = x[:,:,5,:]
    y[:,:,12,:] = x[:,:,7,:]
    y[:,:,13,:] = x[:,:,9,:]
    y[:,:,14,:] = x[:,:,6,:]
    y[:,:,15,:] = x[:,:,8,:]
    y[:,:,16,:] = x[:,:,10,:]
    return y


def backproject_points_to_equirect(
    points: torch.Tensor,
    persp_size: tuple,
    eq_size: tuple = (1920, 3840),
    fov: float = 70.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    Backproject 2D points from a perspective image to equirectangular coordinates.
    
    Args:
        points: Tensor of shape (..., 2) containing (x, y) coordinates in perspective image
        persp_size: Tuple (width, height) of the perspective image
        eq_size: Tuple (height, width) of the equirectangular image (default 1920x3840)
        fov: Horizontal field of view in degrees
        yaw, pitch, roll: Camera orientation in degrees
        device: Torch device
    
    Returns:
        Tensor of shape (..., 2) containing (x, y) coordinates in equirectangular image
    """
    if device is None:
        device = points.device
    
    out_w, out_h = persp_size
    eq_h, eq_w = eq_size
    
    # FOV in radians
    fov_rad = torch.deg2rad(torch.tensor(fov, dtype=torch.float32, device=device))
    aspect = out_h / out_w
    fov_y = 2 * torch.arctan(torch.tan(fov_rad / 2) * aspect)
    
    # Get original shape and flatten points
    original_shape = points.shape[:-1]
    points_flat = points.reshape(-1, 2)
    
    # Convert pixel coordinates to normalized coordinates in tangent plane
    # x: [0, out_w] -> [-tan(fov/2), tan(fov/2)]
    # y: [0, out_h] -> [tan(fov_y/2), -tan(fov_y/2)] (flipped for image coords)
    tan_fov_x = torch.tan(fov_rad / 2)
    tan_fov_y = torch.tan(fov_y / 2)
    
    x_norm = (points_flat[:, 0] / out_w - 0.5) * 2 * tan_fov_x
    y_norm = -(points_flat[:, 1] / out_h - 0.5) * 2 * tan_fov_y  # flip y
    
    # Create 3D direction vectors (pinhole model, z=1)
    z = torch.ones_like(x_norm)
    directions = torch.stack([x_norm, y_norm, z], dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    # Apply rotation (camera orientation)
    R = rot_matrix_torch(yaw, pitch, roll, device)
    dirs_rot = directions @ R.T
    
    # Convert to spherical coordinates
    lon = torch.atan2(dirs_rot[:, 0], dirs_rot[:, 2])
    lat = torch.asin(torch.clamp(dirs_rot[:, 1], -1.0, 1.0))
    
    # Convert to equirectangular pixel coordinates
    eq_x = (lon / (2 * torch.pi) + 0.5) * eq_w
    eq_y = (0.5 - lat / torch.pi) * eq_h
    
    # Stack and reshape back
    eq_points = torch.stack([eq_x, eq_y], dim=-1)
    eq_points = eq_points.reshape(*original_shape, 2)
    
    return eq_points


# Global variable for backprojection debug window
_backproj_debug_frame = None


def action_net_inference(args: argparse.Namespace, model: ActionNet, config:dict, current_tracks_history: dict, device: torch.device, image_size: tuple, backprojection: bool = False) -> np.ndarray:
    """
    Perform inference on the ActionNet model.
    
    Args:
        backprojection: If True, backproject joint coordinates from perspective to 
                       equirectangular (3840x1920) before normalization. Uses FOV=70°, 
                       yaw/pitch/roll=0.
    """
    global _backproj_debug_frame
    
    assert(config["mb_input_norm"] == "vid"), "Only video normalization is supported for live interaction prediction"
    
    min_valid_keypoints = config["min_keypoints_filter"]
    input_length_in_frames = 16 #config["input_length_in_frames"]
    max_index_gap_allowed = 2
    
    # Target size for normalization
    eq_w, eq_h = (3840, 1920) if backprojection else image_size
    whtensor = torch.tensor([eq_w, eq_h], device=device)
    
    # print(current_tracks_history.keys())
    return_dict = {track_id: "NC" for track_id in current_tracks_history.keys()}
    valid_ids = []
    valid_input_tensors = []
    
    for track_id, track_history in current_tracks_history.items():
        indexes = track_history["indexes"]
        input_tensors = track_history["ip_input_tensor"]
        if len(indexes) < input_length_in_frames:
            return_dict[track_id] = "not_enough_frames"
            continue
        
        last_indexes = np.array(indexes[-input_length_in_frames:]) # T
        last_indexes_gap = np.diff(last_indexes) # T-1
        if np.any(last_indexes_gap > max_index_gap_allowed):
            return_dict[track_id] = "index_gap_too_large"
            continue
        
        last_input_tensors = input_tensors[-input_length_in_frames:]
        last_input_tensors = torch.stack(last_input_tensors, dim=0) # T, 17, 3
        last_input_scores = last_input_tensors[:, :, 2] # T, 17
        last_input_valid_joints = (last_input_scores > args.kp_thresh).sum(dim=1) # T
        if torch.any(last_input_valid_joints < min_valid_keypoints):
            return_dict[track_id] = "not_enough_valid_joints"
            continue
        else:
            valid_ids.append(track_id)
            valid_input_tensors.append(last_input_tensors)

    if len(valid_ids) == 0:
        return return_dict
    
    valid_input_tensors = torch.stack(valid_input_tensors, dim=0) # B, T, 17, 3
    
    # Backprojection: perspective -> equirectangular
    if backprojection:
        persp_w, persp_h = image_size
        # Extract xy coordinates and scores
        xy_coords = valid_input_tensors[..., :2]  # B, T, 17, 2
        scores = valid_input_tensors[..., 2:3]    # B, T, 17, 1
        
        # Backproject to equirectangular coordinates
        eq_coords = backproject_points_to_equirect(
            points=xy_coords,
            persp_size=(persp_w, persp_h),
            eq_size=(eq_h, eq_w),  # (height, width)
            fov=70.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            device=device
        )
        
        # Recombine with scores
        valid_input_tensors = torch.cat([eq_coords, scores], dim=-1)  # B, T, 17, 3
        
        # Debug visualization: show projected joints on equirectangular canvas
        if args.display or getattr(args, "save_output", False):
            # Create debug canvas (scaled down for display)
            debug_scale = 0.25
            debug_w, debug_h = int(eq_w * debug_scale), int(eq_h * debug_scale)
            debug_frame = np.zeros((debug_h, debug_w, 3), dtype=np.uint8)
            debug_frame[:] = (30, 30, 30)  # Dark gray background
            
            # Draw grid lines for reference
            for x in range(0, debug_w, debug_w // 8):
                cv2.line(debug_frame, (x, 0), (x, debug_h), (50, 50, 50), 1)
            for y in range(0, debug_h, debug_h // 4):
                cv2.line(debug_frame, (0, y), (debug_w, y), (50, 50, 50), 1)
            
            # Draw joints and skeleton for each valid track (last frame only)
            eq_coords_np = eq_coords.cpu().numpy()
            scores_np = valid_input_tensors[..., 2].cpu().numpy()
            
            for i, track_id in enumerate(valid_ids):
                color = get_track_color(track_id)
                # Use last frame
                kp = eq_coords_np[i, -1]  # 17, 2
                sc = scores_np[i, -1]     # 17
                
                # Scale to debug frame
                kp_scaled = kp * debug_scale
                
                # Draw skeleton
                for j, k in COCO_SKELETON:
                    if sc[j] > args.kp_thresh and sc[k] > args.kp_thresh:
                        pt1 = (int(kp_scaled[j, 0]), int(kp_scaled[j, 1]))
                        pt2 = (int(kp_scaled[k, 0]), int(kp_scaled[k, 1]))
                        # Check if points are valid (within bounds)
                        if 0 <= pt1[0] < debug_w and 0 <= pt1[1] < debug_h and \
                           0 <= pt2[0] < debug_w and 0 <= pt2[1] < debug_h:
                            cv2.line(debug_frame, pt1, pt2, color, 2)
                
                # Draw keypoints
                for kp_idx, (kp_pt, score) in enumerate(zip(kp_scaled, sc)):
                    if score > args.kp_thresh:
                        x, y = int(kp_pt[0]), int(kp_pt[1])
                        if 0 <= x < debug_w and 0 <= y < debug_h:
                            cv2.circle(debug_frame, (x, y), 4, color, -1)
                
                # Draw track ID label
                if sc[0] > args.kp_thresh:  # Use nose position
                    label_x, label_y = int(kp_scaled[0, 0]), int(kp_scaled[0, 1]) - 10
                    if 0 <= label_x < debug_w and 0 <= label_y < debug_h:
                        cv2.putText(debug_frame, f"ID:{track_id}", (label_x, label_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Add title
            cv2.putText(debug_frame, "Backprojection Debug (Equirect 3840x1920)", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            _backproj_debug_frame = debug_frame
    
    valid_input_tensors = coco2h36m(valid_input_tensors)
    scale = min(eq_w, eq_h) / 2.0
    valid_input_tensors[...,:2] = valid_input_tensors[...,:2] - whtensor / 2.0
    valid_input_tensors[...,:2] = valid_input_tensors[...,:2] / scale
    valid_input_tensors = valid_input_tensors.unsqueeze(1) # B, M, T, 17, 3
    valid_input_tensors.to(device)

    with torch.no_grad():
        model_output = model(valid_input_tensors) # B, M, 1
        probabilities = torch.sigmoid(model_output.squeeze())
        if probabilities.ndim == 0:
            probabilities = probabilities.unsqueeze(0) # torch.tensor of shape (1,)
        
        for i, track_id in enumerate(valid_ids):
            return_dict[track_id] = probabilities[i].item()
            
    return return_dict


def get_backproj_debug_frame():
    """Get the latest backprojection debug frame for display."""
    global _backproj_debug_frame
    if _backproj_debug_frame is None:
        debug_h, debug_w = int(1920*0.25), int(3840*0.25)
        debug_frame = np.zeros((debug_h, debug_w, 3), dtype=np.uint8)
        debug_frame[:] = (30, 30, 30)  # Dark gray background
        
        # Draw grid lines for reference
        for x in range(0, debug_w, debug_w // 8):
            cv2.line(debug_frame, (x, 0), (x, debug_h), (50, 50, 50), 1)
        for y in range(0, debug_h, debug_h // 4):
            cv2.line(debug_frame, (0, y), (debug_w, y), (50, 50, 50), 1)
        
        return debug_frame
    
    return _backproj_debug_frame


def create_frame_iterator(args: argparse.Namespace):
    """
    Create a frame iterator based on input type (video or folder).
    
    Yields:
        Tuple of (frame_idx, PIL.Image, depth_image or None, total_frames)
    """
    if args.video is not None:
        # Video input
        video_path = os.path.expanduser(args.video)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB and then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            if image.width == 3840:
                image = image.resize((1920, 960))
            
            yield frame_idx, image, None, total_frames
            frame_idx += 1
        
        cap.release()
    else:
        # Folder input
        image_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(args.directory)) for f in fn if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]
        image_paths.sort()
        total_frames = len(image_paths)
        
        depth_paths = None
        if args.depth is not None:
            depth_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(args.depth)) for f in fn if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]
            depth_paths.sort()
            if abs(len(image_paths) - len(depth_paths)) > 5:
                raise ValueError(f"Number of images (in {args.directory}) and depth images (in {args.depth}) must be the same got {len(image_paths)} and {len(depth_paths)}")
            else:
                # cut to the same length
                image_paths = image_paths[:len(depth_paths)]
                depth_paths = depth_paths[:len(image_paths)]
        
        for frame_idx, image_path in enumerate(image_paths):
            image = Image.open(image_path).convert("RGB")
            if image.width == 3840:
                image = image.resize((1920, 960))
            
            depth_image = None
            if depth_paths is not None:
                depth_image = cv2.imread(depth_paths[frame_idx], cv2.IMREAD_UNCHANGED)
                if depth_image is None:
                    print(f"Warning: Could not load depth image {depth_paths[frame_idx]}")
            
            yield frame_idx, image, depth_image, total_frames


def process_input(args: argparse.Namespace) -> dict:
    """Process video or folder with detection, tracking and pose estimation."""
    track_history = {}
    
    # Load interaction prediction model
    ip_model = None
    ip_config = None
    if args.interaction_prediction_checkpoint is not None:
        
        if os.path.isdir(args.interaction_prediction_checkpoint) and "converted_" in args.interaction_prediction_checkpoint:
            assert(os.path.exists(os.path.join(args.interaction_prediction_checkpoint, "meta.yaml"))), "Meta file not found"
            meta = read_yaml_to_dic(os.path.join(args.interaction_prediction_checkpoint, "meta.yaml"))
            ip_config = meta["config"]
            ip_model = load_model_from_config(ip_config, device="cuda")
            ip_model.load_state_dict(torch.load(os.path.join(args.interaction_prediction_checkpoint, "model_state_dict.pt"), map_location='cpu'), strict=True)
            if type(ip_model) != ActionNet:
                raise NotImplementedError(f"Model type {type(ip_model).__name__} not supported for live interaction prediction for now")
            print(f"Loaded model of type {type(ip_model).__name__} and weights from checkpoint")    
        elif os.path.isfile(args.interaction_prediction_checkpoint):
            ip_checkpoint = torch.load(args.interaction_prediction_checkpoint, map_location='cpu', weights_only=False) 
            # Make sure the checkpoint contains the model weights
            if 'model_state_dict' not in ip_checkpoint:
                raise ValueError("Checkpoint does not contain 'model_state_dict' key. Cannot load model weights.")
            
            # Extract config and handle backward compatibility
            if 'config' not in ip_checkpoint:
                if 'hyperparameters' in ip_checkpoint:
                    ip_checkpoint["config"] = ip_checkpoint["hyperparameters"]
                raise ValueError("Checkpoint does not contain 'config' or 'hyperparameters' key. Cannot recreate dataloader.")

            ip_config = ip_checkpoint['config']
            print(f"Model type: {ip_config['force_model_type']} | cross evaluation type: {ip_config['cross_eval_type']}")
            print(f"AUC {ip_checkpoint['val_auc']:.4f} | AP {ip_checkpoint['val_ap']:.4f}")
            
            print(ip_config.keys())
            
            ip_model = load_model_from_config(ip_config, device="cuda")
            ip_model.load_state_dict(ip_checkpoint['model_state_dict'], strict=True)
            if type(ip_model) != ActionNet:
                raise NotImplementedError(f"Model type {type(ip_model).__name__} not supported for live interaction prediction for now")
            
            print(f"Loaded model of type {type(ip_model).__name__} and weights from checkpoint")    
    
    # Initialize info panel for display and/or save
    info_panel = InfoPanel(
        width=600,
        history_length=100,
        min_track_appearances=16,
        y_max_meters=6.0,
    ) if (args.display or getattr(args, "save_output", False)) else None
    behavior_panel = BehaviorPanel(width=220) if (args.display or getattr(args, "save_output", False)) else None

    # Create frame iterator
    frame_iterator = create_frame_iterator(args)
    
    # Video writers for save_output
    save_output = getattr(args, "save_output", False)
    video_writer_main = None
    video_writer_backproj = None
    output_basename = None
    if save_output:
        if args.video is not None:
            output_basename = Path(args.video).stem
        else:
            output_basename = Path(args.directory).name if args.directory else "output"
            if "/episodes/" in args.directory:
                output_basename = args.directory.split("/")[-4:]
                output_basename = "_".join(output_basename)
            print(f"Saving output to ./output/videos/{output_basename}.mp4")
        os.makedirs("./output/videos", exist_ok=True)
    
    for frame_idx, image, depth_image, total_frames in frame_iterator:
        t_frame_start = time.perf_counter()
        t_ip = 0  # Initialize IP inference time
        
        # Detection + tracking + pose with YOLO pose model
        t_infer_start = time.perf_counter()
        results = pose_model.track(image, persist=True, verbose=False)[0]
        t_infer = time.perf_counter() - t_infer_start
        
        # Skip frame if no detections
        if results.boxes.id is None:
            current_track_ids = []
            boxes = []
            confs = []
            keypoints_all = []
            scores_all = []
            depths = []
        else:
            current_track_ids = results.boxes.id.int().cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            keypoints_all = results.keypoints.xy #.cpu().numpy() # B,17,2
            scores_all = results.keypoints.conf #.cpu().numpy() # B,17
            ip_input_tensor = torch.cat([keypoints_all, scores_all.unsqueeze(-1)], dim=2).clone() # B,17,3                        
            keypoints_all = keypoints_all.cpu().numpy()
            scores_all = scores_all.cpu().numpy()
            
            # Compute depth for each person if depth image is available
            depths = []
            if depth_image is not None:
                for i in range(len(current_track_ids)):
                    depth_raw = estimate_torso_depth(
                        keypoints_all[i], scores_all[i], depth_image, args.kp_thresh
                    )
                    # Convert to meters using depth_scale (e.g., 1000 for mm depth images)
                    depth_meters = depth_raw / args.depth_scale if depth_raw > 0 else None
                    depths.append(depth_meters)
            else:
                depths = [None] * len(current_track_ids)
            
            # Update track history
            for i, track_id in enumerate(current_track_ids):
                track_id = int(track_id)
                if track_id not in track_history:
                    track_history[track_id] = {"detections": [], "poses": [], "ip_input_tensor": [], "indexes": [], "ip_output": []}
                
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
                track_history[track_id]["ip_input_tensor"].append(ip_input_tensor[i])
                track_history[track_id]["indexes"].append(frame_idx)
            
            if args.interaction_prediction_checkpoint is not None:
                # perform ip for current tracks
                t_ip_start = time.perf_counter()
                current_tracks_history = {int(track_id): track_history[track_id] for track_id in current_track_ids}
                ip_dict = action_net_inference(args, 
                                                     ip_model, 
                                                     ip_config, 
                                                     current_tracks_history, 
                                                     device=torch.device("cuda"), 
                                                     image_size=(image.width, image.height),
                                                     backprojection=args.backprojection)
                t_ip = time.perf_counter() - t_ip_start

                for track_id, ip_output in ip_dict.items():
                    track_history[track_id]["ip_output"].append(ip_output)

        # Display and/or build frames for save_output
        if args.display or save_output:
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            for i, track_id in enumerate(current_track_ids):
                color = get_track_color(int(track_id))
                x1, y1, x2, y2 = boxes[i].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Build label with ID, depth (if available), and ip_output (if available)
                label = f"ID:{track_id}"
                if depths[i] is not None and depths[i] > 0:
                    label += f" {depths[i]:.2f}m"
                
                # Add ip_output to label if available
                tid = int(track_id)
                if tid in track_history and track_history[tid]["ip_output"]:
                    ip_val = track_history[tid]["ip_output"][-1]  # Get latest ip_output
                    if isinstance(ip_val, float):
                        label += f" IP:{ip_val:.1%}"
                    else:
                        label += f" IP:{ip_val}"
                
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
                
                # Update info panel depth history
                info_panel.update_track_depth(int(track_id), frame_idx, depths[i])
                
                # Update info panel IP history
                tid = int(track_id)
                if tid in track_history and track_history[tid]["ip_output"]:
                    ip_val = track_history[tid]["ip_output"][-1]
                    info_panel.update_track_ip(tid, frame_idx, ip_val)
            
            # Prune old tracks from panel history
            info_panel.prune_old_tracks(frame_idx)
            
            # Engagement-based lights scale: max IP score over current tracks, mapped to [MIN, MAX] -> [0, 1]
            max_engagement_score = 0.0
            for tid in current_track_ids:
                tid = int(tid)
                if tid in track_history and track_history[tid]["ip_output"]:
                    v = track_history[tid]["ip_output"][-1]
                    if isinstance(v, (int, float)):
                        max_engagement_score = max(max_engagement_score, float(v))
            span = max(1e-6, MAX_LIGHTS_SCALE_THRESHOLD - MIN_LIGHTS_SCALE_THRESHOLD)
            lights_scale = (max_engagement_score - MIN_LIGHTS_SCALE_THRESHOLD) / span
            lights_scale = max(0.0, min(1.0, lights_scale))
            
            # Draw info panel and behavior panel, then concatenate
            t_frame = time.perf_counter() - t_frame_start
            panel = info_panel.draw(
                frame_height=frame.shape[0],
                frame_idx=frame_idx,
                num_tracks=len(current_track_ids),
                latency_ms=t_frame * 1000,
                ip_latency_ms=t_ip * 1000,
            )
            behavior_img = behavior_panel.draw(frame_height=frame.shape[0], lights_scale=lights_scale)
            display_frame = concatenate_with_panel(frame, panel, behavior_img)
            
            if args.display:
                cv2.imshow("Pose", display_frame)
            
            # Show backprojection debug window if enabled
            if args.display and args.backprojection:
                backproj_frame = get_backproj_debug_frame()
                if backproj_frame is not None:
                    cv2.imshow("Backprojection Debug", backproj_frame)
                    cv2.moveWindow("Backprojection Debug", 30,1280)  # Move it
            if args.display and cv2.waitKey(1) == ord("q"):
                break
            
            # Save output to videos when enabled
            if save_output:
                model_basename = args.interaction_prediction_checkpoint.split("/")[-1].split(".")[0]
                output_basename_full = f"{output_basename}_{model_basename}"
                backproj_frame = get_backproj_debug_frame() if args.backprojection else None
                # Initialize writers on first frame
                if video_writer_main is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps = 30.0
                    h_main, w_main = display_frame.shape[:2]
                    video_writer_main = cv2.VideoWriter(
                        f"./output/videos/{output_basename_full}.mp4",
                        fourcc, fps, (w_main, h_main)
                    )
                    if args.backprojection and backproj_frame is not None:
                        h_bp, w_bp = backproj_frame.shape[:2]
                        video_writer_backproj = cv2.VideoWriter(
                            f"./output/videos/{output_basename_full}_backprojection.mp4",
                            fourcc, fps, (w_bp, h_bp)
                        )
                video_writer_main.write(display_frame)
                if video_writer_backproj is not None and backproj_frame is not None:
                    video_writer_backproj.write(backproj_frame)

        t_frame = time.perf_counter() - t_frame_start
        print(f"Frame {frame_idx}/{total_frames} | {len(current_track_ids)} tracks | "
              f"yolo: {t_infer*1000:.1f}ms, ip: {t_ip*1000:.1f}ms, total: {t_frame*1000:.1f}ms")

    if save_output:
        saved_backproj = video_writer_backproj is not None
        if video_writer_main is not None:
            video_writer_main.release()
        if video_writer_backproj is not None:
            video_writer_backproj.release()
        print(f"Saved main output to ./output/videos/{output_basename_full}.mp4")
        if saved_backproj:
            print(f"Saved backprojection to ./output/videos/{output_basename_full}_backprojection.mp4")
    if args.display:
        cv2.destroyAllWindows()
    return track_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--directory", "-d", type=str, help="Path to folder containing images")
    input_group.add_argument("--episode_directory", "-ed", type=str, help="Path to episode directory containing images")
    input_group.add_argument("--video", "-v", type=str, help="Path to video file")
    parser.add_argument("--episode_input_type", "-eit", type=str, choices=["images_360", "images"], help="Type of episode input")
    parser.add_argument("--depth", type=str, default=None, help="Path to folder containing depth images (optional, only for directory input)")
    parser.add_argument("--interaction_prediction_checkpoint", "-ip", type=str, default=None, help="Path to interaction prediction checkpoint (optional)")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Depth scale factor (depth units per meter, e.g., 1000 for mm)")
    parser.add_argument("--display", action="store_true", default=False, help="Display results with cv2")
    parser.add_argument("--kp_thresh", type=float, default=0.3, help="Keypoint confidence threshold")
    parser.add_argument("--backprojection", "-bp", action="store_true", default=False, 
                        help="Backproject joint coordinates from perspective to equirectangular (3840x1920) "
                             "before IP inference. Uses FOV=70°, yaw/pitch/roll=0. Shows debug window when --display is enabled.")
    parser.add_argument("--save_output", "-so", action="store_true", default=False,
                        help="Save the displayed output to ./output/videos/ (main view and backprojection visu when --backprojection).")
    args = parser.parse_args()
    multiple_episodes = False
    if args.episode_directory is not None:
        assert os.path.exists(args.episode_directory), "Episode directory does not exist"
        if args.episode_directory.endswith("episodes") or args.episode_directory.endswith("episodes/"):
            all_episodes = [d for d in os.listdir(args.episode_directory) if os.path.isdir(os.path.join(args.episode_directory, d))]
            all_episodes.sort()
            all_episodes = all_episodes[1:] # skip the first episode
            print(f"Found {len(all_episodes)} episodes")
            multiple_episodes = True
        else:
            multiple_episodes = False
            if args.episode_input_type == "images_360":
                args.directory = os.path.join(args.episode_directory, "images_360")
            elif args.episode_input_type == "images":
                args.directory = os.path.join(args.episode_directory, "images")
                args.depth = os.path.join(args.episode_directory, "depth")
            else:
                raise ValueError(f"Invalid episode input type: {args.episode_input_type}")
    
    if multiple_episodes:    
        for episode in all_episodes:
            image_dir_name = "images_360" if args.episode_input_type == "images_360" else "images"
            args.directory = os.path.join(args.episode_directory, episode, image_dir_name)
            print(f"[DIRECTORY] {args.directory}")
            args.depth = os.path.join(args.episode_directory, episode, "depth")
            history = process_input(args)
            print(f"Processed {len(history)} unique tracks in episode {episode}")
            for track_id, data in history.items():
                print(f"  Track {track_id}: {len(data['detections'])} frames in episode {episode}")
    else:
        history = process_input(args)
        print(f"Processed {len(history)} unique tracks")
        for track_id, data in history.items():
            print(f"  Track {track_id}: {len(data['detections'])} frames")
