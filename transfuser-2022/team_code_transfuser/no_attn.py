"""
消融实验: 移除跨模态注意力 (No Cross-Attention)

基于 Bi-Attenfusion.py，替换为纯 MLP
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
import timm


class MLPOnlyBlock(nn.Module):
    """
    纯 MLP 块 — 无跨模态交互
    图像和 LiDAR 各自通过独立的 MLP 处理自身特征
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1_img = nn.LayerNorm(dim)
        self.norm1_lidar = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_img = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        self.mlp_lidar = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, img_tokens, lidar_tokens):
        img_tokens = img_tokens + self.mlp_img(self.norm1_img(img_tokens))
        lidar_tokens = lidar_tokens + self.mlp_lidar(self.norm1_lidar(lidar_tokens))
        return img_tokens, lidar_tokens


class MultiScaleMLPOnly(nn.Module):
    """多尺度纯 MLP 模块 — 堆叠多个 MLPOnlyBlock"""
    def __init__(self, dim, num_heads, num_blocks=2, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            MLPOnlyBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop
            )
            for _ in range(num_blocks)
        ])

    def forward(self, img_tokens, lidar_tokens):
        for block in self.blocks:
            img_tokens, lidar_tokens = block(img_tokens, lidar_tokens)
        return img_tokens, lidar_tokens


class NoAttnCrossViTBackbone(nn.Module):
    """
    消融实验 Backbone: 无跨模态注意力 (4层多尺度 MLP Only)

    保留4层级的多尺度架构和速度嵌入，但每层仅做独立 MLP，无跨模态交互
    """
    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__()
        self.config = config

        self.avgpool_img = nn.AdaptiveAvgPool2d((config.img_vert_anchors, config.img_horz_anchors))
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((config.lidar_vert_anchors, config.lidar_horz_anchors))

        self.image_encoder = ImageCNN(
            architecture=image_architecture, normalize=True,
            out_features=config.perception_output_features
        )

        if config.use_point_pillars:
            in_channels = config.num_features[-1]
        else:
            in_channels = 2 * config.lidar_seq_len
        if config.use_target_point_image:
            in_channels += 1

        self.lidar_encoder = LidarEncoder(
            architecture=lidar_architecture, in_channels=in_channels,
            out_features=config.perception_output_features
        )

        self.crossvit1 = MultiScaleMLPOnly(
            dim=self.image_encoder.features.feature_info[1]['num_chs'],
            num_heads=config.n_head, num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp, qkv_bias=True,
            drop=config.resid_pdrop, attn_drop=config.attn_pdrop
        )
        self.crossvit2 = MultiScaleMLPOnly(
            dim=self.image_encoder.features.feature_info[2]['num_chs'],
            num_heads=config.n_head, num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp, qkv_bias=True,
            drop=config.resid_pdrop, attn_drop=config.attn_pdrop
        )
        self.crossvit3 = MultiScaleMLPOnly(
            dim=self.image_encoder.features.feature_info[3]['num_chs'],
            num_heads=config.n_head, num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp, qkv_bias=True,
            drop=config.resid_pdrop, attn_drop=config.attn_pdrop
        )
        self.crossvit4 = MultiScaleMLPOnly(
            dim=self.image_encoder.features.feature_info[4]['num_chs'],
            num_heads=config.n_head, num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp, qkv_bias=True,
            drop=config.resid_pdrop, attn_drop=config.attn_pdrop
        )

        self.use_velocity = use_velocity
        if use_velocity:
            self.vel_emb1 = nn.Linear(1, self.image_encoder.features.feature_info[1]['num_chs'])
            self.vel_emb2 = nn.Linear(1, self.image_encoder.features.feature_info[2]['num_chs'])
            self.vel_emb3 = nn.Linear(1, self.image_encoder.features.feature_info[3]['num_chs'])
            self.vel_emb4 = nn.Linear(1, self.image_encoder.features.feature_info[4]['num_chs'])

        if self.image_encoder.features.feature_info[4]['num_chs'] != config.perception_output_features:
            self.change_channel_conv_image = nn.Conv2d(
                self.image_encoder.features.feature_info[4]['num_chs'],
                config.perception_output_features, (1, 1)
            )
            self.change_channel_conv_lidar = nn.Conv2d(
                self.image_encoder.features.feature_info[4]['num_chs'],
                config.perception_output_features, (1, 1)
            )
        else:
            self.change_channel_conv_image = nn.Sequential()
            self.change_channel_conv_lidar = nn.Sequential()

        channel = config.bev_features_chanels
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=config.bev_upsample_factor, mode='bilinear', align_corners=False)
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))
        self.c5_conv = nn.Conv2d(config.perception_output_features, channel, (1, 1))

    def top_down(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample(p4)))
        p2 = self.relu(self.up_conv3(self.upsample(p3)))
        return p2, p3, p4, p5

    def forward(self, image, lidar, velocity):
        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image
        lidar_tensor = lidar

        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.act1(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)

        lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.act1(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)

        # Layer1
        image_features = self.image_encoder.features.layer1(image_features)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)

        image_embd = self.avgpool_img(image_features)
        lidar_embd = self.avgpool_lidar(lidar_features)
        B, C1, H_img, W_img = image_embd.shape
        _, _, H_lidar, W_lidar = lidar_embd.shape

        img_tokens = image_embd.flatten(2).transpose(1, 2)
        lidar_tokens = lidar_embd.flatten(2).transpose(1, 2)
        if self.use_velocity:
            vel_emb = self.vel_emb1(velocity).unsqueeze(1)
            img_tokens = img_tokens + vel_emb
            lidar_tokens = lidar_tokens + vel_emb

        img_tokens, lidar_tokens = self.crossvit1(img_tokens, lidar_tokens)
        image_features_fused = img_tokens.transpose(1, 2).reshape(B, C1, H_img, W_img)
        lidar_features_fused = lidar_tokens.transpose(1, 2).reshape(B, C1, H_lidar, W_lidar)
        image_features_fused = F.interpolate(image_features_fused, size=(image_features.shape[2], image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_fused = F.interpolate(lidar_features_fused, size=(lidar_features.shape[2], lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_fused
        lidar_features = lidar_features + lidar_features_fused

        # Layer2
        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)

        image_embd = self.avgpool_img(image_features)
        lidar_embd = self.avgpool_lidar(lidar_features)
        C2 = image_embd.shape[1]
        img_tokens = image_embd.flatten(2).transpose(1, 2)
        lidar_tokens = lidar_embd.flatten(2).transpose(1, 2)
        if self.use_velocity:
            vel_emb = self.vel_emb2(velocity).unsqueeze(1)
            img_tokens = img_tokens + vel_emb
            lidar_tokens = lidar_tokens + vel_emb

        img_tokens, lidar_tokens = self.crossvit2(img_tokens, lidar_tokens)
        image_features_fused = img_tokens.transpose(1, 2).reshape(B, C2, H_img, W_img)
        lidar_features_fused = lidar_tokens.transpose(1, 2).reshape(B, C2, H_lidar, W_lidar)
        image_features_fused = F.interpolate(image_features_fused, size=(image_features.shape[2], image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_fused = F.interpolate(lidar_features_fused, size=(lidar_features.shape[2], lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_fused
        lidar_features = lidar_features + lidar_features_fused

        # Layer3
        image_features = self.image_encoder.features.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)

        image_embd = self.avgpool_img(image_features)
        lidar_embd = self.avgpool_lidar(lidar_features)
        C3 = image_embd.shape[1]
        img_tokens = image_embd.flatten(2).transpose(1, 2)
        lidar_tokens = lidar_embd.flatten(2).transpose(1, 2)
        if self.use_velocity:
            vel_emb = self.vel_emb3(velocity).unsqueeze(1)
            img_tokens = img_tokens + vel_emb
            lidar_tokens = lidar_tokens + vel_emb

        img_tokens, lidar_tokens = self.crossvit3(img_tokens, lidar_tokens)
        image_features_fused = img_tokens.transpose(1, 2).reshape(B, C3, H_img, W_img)
        lidar_features_fused = lidar_tokens.transpose(1, 2).reshape(B, C3, H_lidar, W_lidar)
        image_features_fused = F.interpolate(image_features_fused, size=(image_features.shape[2], image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_fused = F.interpolate(lidar_features_fused, size=(lidar_features.shape[2], lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_fused
        lidar_features = lidar_features + lidar_features_fused

        # Layer4
        image_features = self.image_encoder.features.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)

        image_embd = self.avgpool_img(image_features)
        lidar_embd = self.avgpool_lidar(lidar_features)
        C4 = image_embd.shape[1]
        img_tokens = image_embd.flatten(2).transpose(1, 2)
        lidar_tokens = lidar_embd.flatten(2).transpose(1, 2)
        if self.use_velocity:
            vel_emb = self.vel_emb4(velocity).unsqueeze(1)
            img_tokens = img_tokens + vel_emb
            lidar_tokens = lidar_tokens + vel_emb

        img_tokens, lidar_tokens = self.crossvit4(img_tokens, lidar_tokens)
        image_features_fused = img_tokens.transpose(1, 2).reshape(B, C4, H_img, W_img)
        lidar_features_fused = lidar_tokens.transpose(1, 2).reshape(B, C4, H_lidar, W_lidar)
        image_features_fused = F.interpolate(image_features_fused, size=(image_features.shape[2], image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_fused = F.interpolate(lidar_features_fused, size=(lidar_features.shape[2], lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_fused
        lidar_features = lidar_features + lidar_features_fused

        image_features = self.change_channel_conv_image(image_features)
        lidar_features = self.change_channel_conv_lidar(lidar_features)

        x4 = lidar_features
        image_features_grid = image_features

        image_features = self.image_encoder.features.global_pool(image_features)
        image_features = torch.flatten(image_features, 1)
        lidar_features = self.lidar_encoder._model.global_pool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)
        fused_features = image_features + lidar_features

        features = self.top_down(x4)
        return features, image_features_grid, fused_features


def normalize_imagenet(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape(1, 3, 1, 1)
    return (x - mean) / std


class ImageCNN(nn.Module):
    def __init__(self, architecture, normalize=True, out_features=512):
        super().__init__()
        self.normalize = normalize
        self.features = timm.create_model(architecture, pretrained=True)
        self.features.fc = None
        arch_lower = architecture.lower()
        if arch_lower.startswith('regnet'):
            self.features.conv1 = self.features.stem.conv
            self.features.bn1 = self.features.stem.bn
            self.features.act1 = nn.Sequential()
            self.features.maxpool = nn.Sequential()
            self.features.layer1 = self.features.s1
            self.features.layer2 = self.features.s2
            self.features.layer3 = self.features.s3
            self.features.layer4 = self.features.s4
            self.features.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.features.head = nn.Sequential()
        elif arch_lower.startswith('convnext'):
            self.features.conv1 = self.features.stem._modules['0']
            self.features.bn1 = self.features.stem._modules['1']
            self.features.act1 = nn.Sequential()
            self.features.maxpool = nn.Sequential()
            self.features.layer1 = self.features.stages._modules['0']
            self.features.layer2 = self.features.stages._modules['1']
            self.features.layer3 = self.features.stages._modules['2']
            self.features.layer4 = self.features.stages._modules['3']
            self.features.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.features.head = nn.Sequential()


class LidarEncoder(nn.Module):
    def __init__(self, architecture, in_channels, out_features=512):
        super().__init__()
        self._model = timm.create_model(architecture, pretrained=True)
        self._model.fc = None
        if architecture.lower().startswith('regnet'):
            self._model.stem.conv = nn.Conv2d(
                in_channels, self._model.stem.conv.out_channels,
                kernel_size=self._model.stem.conv.kernel_size,
                stride=self._model.stem.conv.stride,
                padding=self._model.stem.conv.padding, bias=False
            )
            self._model.conv1 = self._model.stem.conv
            self._model.bn1 = self._model.stem.bn
            self._model.act1 = nn.Sequential()
            self._model.maxpool = nn.Sequential()
            self._model.layer1 = self._model.s1
            self._model.layer2 = self._model.s2
            self._model.layer3 = self._model.s3
            self._model.layer4 = self._model.s4
            self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        elif architecture.lower().startswith('convnext'):
            self._model.stem._modules['0'] = nn.Conv2d(
                in_channels, self._model.stem._modules['0'].out_channels,
                kernel_size=self._model.stem._modules['0'].kernel_size,
                stride=self._model.stem._modules['0'].stride,
                padding=self._model.stem._modules['0'].padding, bias=False
            )
            self._model.conv1 = self._model.stem._modules['0']
            self._model.bn1 = self._model.stem._modules['1']
            self._model.act1 = nn.Sequential()
            self._model.maxpool = nn.Sequential()
            self._model.layer1 = self._model.stages._modules['0']
            self._model.layer2 = self._model.stages._modules['1']
            self._model.layer3 = self._model.stages._modules['2']
            self._model.layer4 = self._model.stages._modules['3']
            self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        else:
            self._model.conv1 = nn.Conv2d(
                in_channels, self._model.conv1.out_channels,
                kernel_size=self._model.conv1.kernel_size,
                stride=self._model.conv1.stride,
                padding=self._model.conv1.padding, bias=False
            )
            self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
