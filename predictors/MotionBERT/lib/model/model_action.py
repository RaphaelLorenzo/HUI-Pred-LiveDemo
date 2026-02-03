import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attentive_pooler import AttentiveClassifier
import einops
class ActionHeadClassificationTimeAvg(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_classes=60, num_joints=17, hidden_dim=2048):
        super(ActionHeadClassificationTimeAvg, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)                # (N, M, J, C)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)                 # (N, J*C)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)    
        feat = self.fc2(feat)
        return feat
    
class ActionHeadClassificationJointAvg(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_classes=60, hidden_dim=2048, fixed_time_length=243):
        super(ActionHeadClassificationJointAvg, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep*fixed_time_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 2, 4, 3)      # (N, M, T, J, C) -> (N, M, T, C, J)
        feat = feat.mean(dim=-1)                # (N, M, T, C)
        feat = feat.reshape(N, M, -1)           # (N, M, T*C)
        feat = feat.mean(dim=1)                 # (N, T*C)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat = self.fc2(feat)
        return feat

class ActionHeadClassificationFullAvg(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_classes=60, hidden_dim=2048):
        super(ActionHeadClassificationFullAvg, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 4, 2, 3)      # (N, M, T, J, C) -> (N, M, C, T, J)
        feat = feat.mean(dim=-1).mean(dim=-1)   # (N, M, C)
        feat = feat.mean(dim=1)                 # (N, C)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)    
        feat = self.fc2(feat)
        return feat
        
class ActionHeadEmbed(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_joints=17, hidden_dim=2048):
        super(ActionHeadEmbed, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = F.normalize(feat, dim=-1)
        return feat

class ActionNet(nn.Module):
    def __init__(self, backbone, dim_rep=512, num_classes=60, dropout_ratio=0., version='class', hidden_dim=2048, num_joints=17, fixed_time_length=None):
        super(ActionNet, self).__init__()
        self.backbone = backbone
        self.feat_J = num_joints
        if version=='class' or version=='class_time_avg':
            self.head = ActionHeadClassificationTimeAvg(dropout_ratio=dropout_ratio, dim_rep=dim_rep, num_classes=num_classes, hidden_dim=hidden_dim, num_joints=num_joints)
        elif version=='class_joint_avg':
            if fixed_time_length is None:
                raise ValueError("fixed_time_length must be provided for class_joint_avg")
            self.head = ActionHeadClassificationJointAvg(dropout_ratio=dropout_ratio, dim_rep=dim_rep, num_classes=num_classes, hidden_dim=hidden_dim, fixed_time_length=fixed_time_length)
        elif version=='class_full_avg':
            self.head = ActionHeadClassificationFullAvg(dropout_ratio=dropout_ratio, dim_rep=dim_rep, num_classes=num_classes, hidden_dim=hidden_dim)
        elif version=='class_attention':
            self.head = AttentiveClassifier(embed_dim=dim_rep, num_classes=num_classes, num_heads=8)
        elif version=='embed':
            self.head = ActionHeadEmbed(dropout_ratio=dropout_ratio, dim_rep=dim_rep, hidden_dim=hidden_dim, num_joints=num_joints)
        else:
            raise Exception('Version Error.')
        
    def forward(self, x, use_window=None):
        '''
            Input: (B, M, T, J, C) 
        '''
        
        B, M, Tin, J, C = x.shape
        x = x.reshape(B*M, Tin, J, C)                           # [B*M, Tin, J, C] usually M = 1        
        feat = self.backbone(x)                                 # [N, Trep, J, C] with Trep that can be different from Tin when backbone is MBForesight (Trep = Tpast + Tfuture)
        _, Trep, _, _ = feat.shape                              
        feat = feat.reshape([B, M, Trep, self.feat_J, -1])      # [B, M, Trep, J, C], put back the M dimension
        
        if use_window is not None:
            # use only a limited time window for the downstream task (only the future part for example)
            feat = feat[:,:,use_window[0]:use_window[1],:,:]
        
        out = self.head(feat)
        return out