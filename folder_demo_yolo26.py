import argparse
import os
import time
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import torch
import torch.nn as nn
from utils.visualization import InfoPanel, get_track_color, concatenate_with_panel
from functools import partial

from predictors.mlp import MLPInteractionPredictor
from predictors.lstm import LSTMInteractionPredictor
from predictors.MotionBERT.lib.model.DSTformer import DSTformer
from predictors.MotionBERT.lib.model.model_action import ActionNet
from predictors.STG_NF.model_pose import STG_NF
from predictors.STGCN.net.st_gcn import Model as STGCN
from predictors.SkateFormer.model.SkateFormer import SkateFormer

# COCO pose skeleton connections (17 keypoints format)
COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], 
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], 
    [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
]


def estimate_torso_depth(keypoints: np.ndarray, scores: np.ndarray, depth_image: np.ndarray, kp_thresh: float) -> float | None:
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


# Load model
pose_model = YOLO("checkpoints/yolo26x-pose.pt")


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
    

def action_net_inference(args: argparse.Namespace, model: ActionNet, config:dict, current_tracks_history: dict, device: torch.device, image_size: tuple) -> np.ndarray:
    """
    Perform inference on the ActionNet model.
    """
    
    assert(config["mb_input_norm"] == "vid"), "Only video normalization is supported for live interaction prediction"
    
    min_valid_keypoints = config["min_keypoints_filter"]
    input_length_in_frames = config["input_length_in_frames"]
    max_index_gap_allowed = 2
    w,h = image_size
    whtensor = torch.tensor([w, h], device=device)
    
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
    valid_input_tensors = coco2h36m(valid_input_tensors)
    scale = min(w,h) / 2.0
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
        image_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(args.directory)) for f in fn if f.endswith(".jpg") or f.endswith(".png")]
        image_paths.sort()
        total_frames = len(image_paths)
        
        depth_paths = None
        if args.depth is not None:
            depth_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(args.depth)) for f in fn if f.endswith(".jpg") or f.endswith(".png")]
            depth_paths.sort()
            assert len(image_paths) == len(depth_paths), "Number of images and depth images must be the same"
        
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
    
    # Initialize info panel for display
    info_panel = InfoPanel(
        width=600,
        history_length=100,
        min_track_appearances=16,
        y_max_meters=6.0,
    ) if args.display else None
    
    # Create frame iterator
    frame_iterator = create_frame_iterator(args)
    
    for frame_idx, image, depth_image, total_frames in frame_iterator:
        t_frame_start = time.perf_counter()
        
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
                current_tracks_history = {int(track_id): track_history[track_id] for track_id in current_track_ids}
                ip_dict = action_net_inference(args, 
                                                     ip_model, 
                                                     ip_config, 
                                                     current_tracks_history, 
                                                     device=torch.device("cuda"), 
                                                     image_size=(image.width, image.height))

                for track_id, ip_output in ip_dict.items():
                    track_history[track_id]["ip_output"].append(ip_output)

        # Display
        if args.display:
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
            
            # Draw info panel and concatenate
            t_frame = time.perf_counter() - t_frame_start
            panel = info_panel.draw(
                frame_height=frame.shape[0],
                frame_idx=frame_idx,
                num_tracks=len(current_track_ids),
                latency_ms=t_frame * 1000,
            )
            display_frame = concatenate_with_panel(frame, panel)
            
            cv2.imshow("Pose", display_frame)
            if cv2.waitKey(1) == ord("q"):
                break

        t_frame = time.perf_counter() - t_frame_start
        print(f"Frame {frame_idx}/{total_frames} | {len(current_track_ids)} tracks | "
              f"infer: {t_infer*1000:.1f}ms, total: {t_frame*1000:.1f}ms")

    if args.display:
        cv2.destroyAllWindows()
    return track_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--directory", "-d", type=str, help="Path to folder containing images")
    input_group.add_argument("--video", "-v", type=str, help="Path to video file")
    parser.add_argument("--depth", type=str, default=None, help="Path to folder containing depth images (optional, only for directory input)")
    parser.add_argument("--interaction_prediction_checkpoint", "-ip", type=str, default=None, help="Path to interaction prediction checkpoint (optional)")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Depth scale factor (depth units per meter, e.g., 1000 for mm)")
    parser.add_argument("--display", action="store_true", default=False, help="Display results with cv2")
    parser.add_argument("--kp_thresh", type=float, default=0.3, help="Keypoint confidence threshold")
    args = parser.parse_args()
    
    history = process_input(args)
    print(f"Processed {len(history)} unique tracks")
    for track_id, data in history.items():
        print(f"  Track {track_id}: {len(data['detections'])} frames")
