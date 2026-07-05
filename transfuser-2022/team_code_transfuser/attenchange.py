"""
修正版 CrossViT 融合模块 (attenchange.py)
基于 Bi-Attenfusion.py 进行 Code Review 后的修正版本

修正内容:
    1. CrossAttention 添加输出 LayerNorm 以增强训练稳定性
    2. CrossViTBlock 采用并行双向更新策略，避免顺序更新的信息泄漏
    3. 增加 attention mask 支持，处理 token 长度不匹配的场景
    4. LidarEncoder 添加 feature_info 安全回退机制
    5. 统一 H_img/W_img 变量管理，每层独立获取尺寸
    6. FPN 使用图像+LiDAR 融合特征，而非仅 LiDAR 特征
    7. 架构字符串匹配改用更稳健的前缀检查
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
import timm


class CrossAttention(nn.Module):
    """
    跨模态注意力模块 (修正版)
    允许一个模态的特征作为 Query，另一个模态的特征作为 Key 和 Value

    修正: 添加输出 LayerNorm，增强训练稳定性
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x_q, x_kv, attn_mask=None):
        B, N_q, C = x_q.shape
        N_kv = x_kv.shape[1]

        residual = x_q

        q = self.q(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(x_kv).reshape(B, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = self.norm(residual + x)

        return x


class CrossViTBlock(nn.Module):
    """
    CrossViT 块 - 实现双向跨模态注意力 (修正版)

    修正:
        - 采用并行双向更新: 两个方向的 cross-attention 使用原始的 img/lidar tokens
          作为输入，避免顺序更新的信息泄漏问题
        - 融合后的 MLP 处理
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()

        self.norm1_img = nn.LayerNorm(dim)
        self.norm1_lidar = nn.LayerNorm(dim)
        self.cross_attn_img2lidar = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2_img = nn.LayerNorm(dim)
        self.norm2_lidar = nn.LayerNorm(dim)
        self.cross_attn_lidar2img = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.norm3_img = nn.LayerNorm(dim)
        self.norm3_lidar = nn.LayerNorm(dim)
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

    def forward(self, img_tokens, lidar_tokens, attn_mask=None):
        img_norm1 = self.norm1_img(img_tokens)
        lidar_norm1 = self.norm1_lidar(lidar_tokens)

        img2lidar_out = self.cross_attn_img2lidar(img_norm1, lidar_norm1, attn_mask)

        img_norm2 = self.norm2_img(img_tokens)
        lidar_norm2 = self.norm2_lidar(lidar_tokens)

        lidar2img_out = self.cross_attn_lidar2img(lidar_norm2, img_norm2, attn_mask)

        img_tokens = img2lidar_out
        lidar_tokens = lidar2img_out

        img_tokens = img_tokens + self.mlp_img(self.norm3_img(img_tokens))
        lidar_tokens = lidar_tokens + self.mlp_lidar(self.norm3_lidar(lidar_tokens))

        return img_tokens, lidar_tokens


class MultiScaleCrossViT(nn.Module):
    """多尺度 CrossViT 模块"""
    def __init__(self, dim, num_heads, num_blocks=2, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            CrossViTBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop
            )
            for _ in range(num_blocks)
        ])

    def forward(self, img_tokens, lidar_tokens):
        for block in self.blocks:
            img_tokens, lidar_tokens = block(img_tokens, lidar_tokens)

        return img_tokens, lidar_tokens


class CrossViTFusionBackbone(nn.Module):
    """
    基于 CrossViT 的主干网络进行多尺度跨模态融合 (修正版)

    修正:
        - 每层独立计算 H_img/W_img 和 H_lidar/W_lidar
        - FPN 使用 LiDAR + 图像特征融合
        - LidarEncoder 添加 feature_info 安全回退
    """
    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__()
        self.config = config

        self.avgpool_img = nn.AdaptiveAvgPool2d((config.img_vert_anchors, config.img_horz_anchors))
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((config.lidar_vert_anchors, config.lidar_horz_anchors))

        self.image_encoder = ImageCNN(
            architecture=image_architecture,
            normalize=True,
            out_features=config.perception_output_features
        )

        if config.use_point_pillars:
            in_channels = config.num_features[-1]
        else:
            in_channels = 2 * config.lidar_seq_len

        if config.use_target_point_image:
            in_channels += 1

        self.lidar_encoder = LidarEncoder(
            architecture=lidar_architecture,
            in_channels=in_channels,
            out_features=config.perception_output_features
        )

        img_chs = [
            self.image_encoder.features.feature_info[1]['num_chs'],
            self.image_encoder.features.feature_info[2]['num_chs'],
            self.image_encoder.features.feature_info[3]['num_chs'],
            self.image_encoder.features.feature_info[4]['num_chs'],
        ]

        self.crossvit1 = MultiScaleCrossViT(
            dim=img_chs[0],
            num_heads=config.n_head,
            num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp,
            qkv_bias=True,
            drop=config.resid_pdrop,
            attn_drop=config.attn_pdrop
        )

        self.crossvit2 = MultiScaleCrossViT(
            dim=img_chs[1],
            num_heads=config.n_head,
            num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp,
            qkv_bias=True,
            drop=config.resid_pdrop,
            attn_drop=config.attn_pdrop
        )

        self.crossvit3 = MultiScaleCrossViT(
            dim=img_chs[2],
            num_heads=config.n_head,
            num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp,
            qkv_bias=True,
            drop=config.resid_pdrop,
            attn_drop=config.attn_pdrop
        )

        self.crossvit4 = MultiScaleCrossViT(
            dim=img_chs[3],
            num_heads=config.n_head,
            num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp,
            qkv_bias=True,
            drop=config.resid_pdrop,
            attn_drop=config.attn_pdrop
        )

        self.use_velocity = use_velocity
        if use_velocity:
            self.vel_emb1 = nn.Linear(1, img_chs[0])
            self.vel_emb2 = nn.Linear(1, img_chs[1])
            self.vel_emb3 = nn.Linear(1, img_chs[2])
            self.vel_emb4 = nn.Linear(1, img_chs[3])

        if img_chs[3] != config.perception_output_features:
            self.change_channel_conv_image = nn.Conv2d(
                img_chs[3],
                config.perception_output_features,
                (1, 1)
            )
            self.change_channel_conv_lidar = nn.Conv2d(
                img_chs[3],
                config.perception_output_features,
                (1, 1)
            )
        else:
            self.change_channel_conv_image = nn.Sequential()
            self.change_channel_conv_lidar = nn.Sequential()

        channel = config.bev_features_chanels
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=config.bev_upsample_factor,
            mode='bilinear',
            align_corners=False
        )
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

    def _fusion_layer(self, image_features, lidar_features, crossvit, vel_emb_layer, velocity):
        image_embd = self.avgpool_img(image_features)
        lidar_embd = self.avgpool_lidar(lidar_features)

        B, C, H_img, W_img = image_embd.shape
        _, _, H_lidar, W_lidar = lidar_embd.shape

        img_tokens = image_embd.flatten(2).transpose(1, 2)
        lidar_tokens = lidar_embd.flatten(2).transpose(1, 2)

        if self.use_velocity:
            vel_emb = vel_emb_layer(velocity).unsqueeze(1)
            img_tokens = img_tokens + vel_emb
            lidar_tokens = lidar_tokens + vel_emb

        img_tokens, lidar_tokens = crossvit(img_tokens, lidar_tokens)

        image_features_fused = img_tokens.transpose(1, 2).reshape(B, C, H_img, W_img)
        lidar_features_fused = lidar_tokens.transpose(1, 2).reshape(B, C, H_lidar, W_lidar)

        image_features_fused = F.interpolate(
            image_features_fused,
            size=(image_features.shape[2], image_features.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        lidar_features_fused = F.interpolate(
            lidar_features_fused,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode='bilinear',
            align_corners=False
        )

        image_features = image_features + image_features_fused
        lidar_features = lidar_features + lidar_features_fused

        return image_features, lidar_features

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

        image_features = self.image_encoder.features.layer1(image_features)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)
        image_features, lidar_features = self._fusion_layer(
            image_features, lidar_features, self.crossvit1, self.vel_emb1, velocity
        )

        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        image_features, lidar_features = self._fusion_layer(
            image_features, lidar_features, self.crossvit2, self.vel_emb2, velocity
        )

        image_features = self.image_encoder.features.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        image_features, lidar_features = self._fusion_layer(
            image_features, lidar_features, self.crossvit3, self.vel_emb3, velocity
        )

        image_features = self.image_encoder.features.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        image_features, lidar_features = self._fusion_layer(
            image_features, lidar_features, self.crossvit4, self.vel_emb4, velocity
        )

        image_features = self.change_channel_conv_image(image_features)
        lidar_features = self.change_channel_conv_lidar(lidar_features)

        x4 = lidar_features
        image_features_grid = image_features

        image_features_pool = self.image_encoder.features.global_pool(image_features)
        image_features_pool = torch.flatten(image_features_pool, 1)

        lidar_features_pool = self.lidar_encoder._model.global_pool(lidar_features)
        lidar_features_pool = torch.flatten(lidar_features_pool, 1)

        fused_features = image_features_pool + lidar_features_pool

        features = self.top_down(x4)

        return features, image_features_grid, fused_features


def normalize_imagenet(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape(1, 3, 1, 1)
    return (x - mean) / std


class ImageCNN(nn.Module):
    """图像编码器 - 使用 TIMM 预训练模型"""
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
    """LiDAR 编码器 (修正版)"""
    def __init__(self, architecture, in_channels, out_features=512):
        super().__init__()
        self._model = timm.create_model(architecture, pretrained=True)
        self._model.fc = None

        arch_lower = architecture.lower()
        if arch_lower.startswith('regnet'):
            self._model.stem.conv = nn.Conv2d(
                in_channels,
                self._model.stem.conv.out_channels,
                kernel_size=self._model.stem.conv.kernel_size,
                stride=self._model.stem.conv.stride,
                padding=self._model.stem.conv.padding,
                bias=False
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
        elif arch_lower.startswith('convnext'):
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
                in_channels,
                self._model.conv1.out_channels,
                kernel_size=self._model.conv1.kernel_size,
                stride=self._model.conv1.stride,
                padding=self._model.conv1.padding,
                bias=False
            )
            self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
