"""
单模态Backbone模块 - 用于消融实验对比
包含仅图像和仅LiDAR两种单模态架构
基于CrossViT架构，将跨模态注意力替换为自注意力
"""
import torch
from torch import nn
import torch.nn.functional as F
import timm

from crossvit_fusion import (
    ImageCNN, LidarEncoder, normalize_imagenet,
    MultiScaleCrossViT
)


class SelfAttentionBlock(nn.Module):
    """
    自注意力块 - 替代CrossViT中的跨模态注意力
    单模态特征通过自注意力进行特征增强
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, tokens):
        tokens_norm = self.norm1(tokens)
        attn_out, _ = self.self_attn(tokens_norm, tokens_norm, tokens_norm)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class MultiScaleSelfAttention(nn.Module):
    """
    多尺度自注意力模块
    堆叠多个自注意力块
    """
    def __init__(self, dim, num_heads, num_blocks=2, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop
            )
            for _ in range(num_blocks)
        ])

    def forward(self, tokens):
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class ImageOnlyBackbone(nn.Module):
    """
    仅图像模态Backbone
    使用图像编码器 + 自注意力增强，不使用LiDAR输入
    
    用于消融实验：验证LiDAR模态的贡献
    """
    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__()
        self.config = config

        self.avgpool_img = nn.AdaptiveAvgPool2d((config.img_vert_anchors, config.img_horz_anchors))

        self.image_encoder = ImageCNN(
            architecture=image_architecture,
            normalize=True,
            out_features=config.perception_output_features
        )

        self.self_attn1 = MultiScaleSelfAttention(
            dim=self.image_encoder.features.feature_info[1]['num_chs'],
            num_heads=config.n_head,
            num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp,
            qkv_bias=True,
            drop=config.resid_pdrop,
            attn_drop=config.attn_pdrop
        )
        self.self_attn2 = MultiScaleSelfAttention(
            dim=self.image_encoder.features.feature_info[2]['num_chs'],
            num_heads=config.n_head,
            num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp,
            qkv_bias=True,
            drop=config.resid_pdrop,
            attn_drop=config.attn_pdrop
        )
        self.self_attn3 = MultiScaleSelfAttention(
            dim=self.image_encoder.features.feature_info[3]['num_chs'],
            num_heads=config.n_head,
            num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp,
            qkv_bias=True,
            drop=config.resid_pdrop,
            attn_drop=config.attn_pdrop
        )
        self.self_attn4 = MultiScaleSelfAttention(
            dim=self.image_encoder.features.feature_info[4]['num_chs'],
            num_heads=config.n_head,
            num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp,
            qkv_bias=True,
            drop=config.resid_pdrop,
            attn_drop=config.attn_pdrop
        )

        self.use_velocity = use_velocity
        if use_velocity:
            self.vel_emb1 = nn.Linear(1, self.image_encoder.features.feature_info[1]['num_chs'])
            self.vel_emb2 = nn.Linear(1, self.image_encoder.features.feature_info[2]['num_chs'])
            self.vel_emb3 = nn.Linear(1, self.image_encoder.features.feature_info[3]['num_chs'])
            self.vel_emb4 = nn.Linear(1, self.image_encoder.features.feature_info[4]['num_chs'])

        if self.image_encoder.features.feature_info[4]['num_chs'] != config.perception_output_features:
            self.change_channel_conv = nn.Conv2d(
                self.image_encoder.features.feature_info[4]['num_chs'],
                config.perception_output_features,
                (1, 1)
            )
        else:
            self.change_channel_conv = nn.Sequential()

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

    def _process_layer(self, image_features, self_attn, vel_emb_layer, velocity):
        image_embd = self.avgpool_img(image_features)
        B, C, H, W = image_embd.shape
        img_tokens = image_embd.flatten(2).transpose(1, 2)

        if self.use_velocity:
            vel_emb = vel_emb_layer(velocity).unsqueeze(1)
            img_tokens = img_tokens + vel_emb

        img_tokens = self_attn(img_tokens)

        image_features_fused = img_tokens.transpose(1, 2).reshape(B, C, H, W)
        image_features_fused = F.interpolate(
            image_features_fused,
            size=(image_features.shape[2], image_features.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        image_features = image_features + image_features_fused
        return image_features

    def forward(self, image, lidar, velocity):
        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image

        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.act1(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)

        image_features = self.image_encoder.features.layer1(image_features)
        image_features = self._process_layer(image_features, self.self_attn1, self.vel_emb1, velocity)

        image_features = self.image_encoder.features.layer2(image_features)
        image_features = self._process_layer(image_features, self.self_attn2, self.vel_emb2, velocity)

        image_features = self.image_encoder.features.layer3(image_features)
        image_features = self._process_layer(image_features, self.self_attn3, self.vel_emb3, velocity)

        image_features = self.image_encoder.features.layer4(image_features)
        image_features = self._process_layer(image_features, self.self_attn4, self.vel_emb4, velocity)

        image_features = self.change_channel_conv(image_features)
        image_features_grid = image_features

        x4 = image_features

        image_features = self.image_encoder.features.global_pool(image_features)
        image_features = torch.flatten(image_features, 1)

        fused_features = image_features
        features = self.top_down(x4)

        return features, image_features_grid, fused_features


class LidarOnlyBackbone(nn.Module):
    """
    仅LiDAR模态Backbone
    使用LiDAR编码器 + 自注意力增强，不使用图像输入
    
    用于消融实验：验证图像模态的贡献
    """
    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__()
        self.config = config

        self.avgpool_lidar = nn.AdaptiveAvgPool2d((config.lidar_vert_anchors, config.lidar_horz_anchors))

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

        lidar_info = self.lidar_encoder._model.feature_info if hasattr(self.lidar_encoder._model, 'feature_info') else None

        if lidar_info is not None:
            ch1 = lidar_info[1]['num_chs']
            ch2 = lidar_info[2]['num_chs']
            ch3 = lidar_info[3]['num_chs']
            ch4 = lidar_info[4]['num_chs']
        else:
            ch1 = 64
            ch2 = 64
            ch3 = 128
            ch4 = 256

        self.self_attn1 = MultiScaleSelfAttention(
            dim=ch1,
            num_heads=config.n_head,
            num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp,
            qkv_bias=True,
            drop=config.resid_pdrop,
            attn_drop=config.attn_pdrop
        )
        self.self_attn2 = MultiScaleSelfAttention(
            dim=ch2,
            num_heads=config.n_head,
            num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp,
            qkv_bias=True,
            drop=config.resid_pdrop,
            attn_drop=config.attn_pdrop
        )
        self.self_attn3 = MultiScaleSelfAttention(
            dim=ch3,
            num_heads=config.n_head,
            num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp,
            qkv_bias=True,
            drop=config.resid_pdrop,
            attn_drop=config.attn_pdrop
        )
        self.self_attn4 = MultiScaleSelfAttention(
            dim=ch4,
            num_heads=config.n_head,
            num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp,
            qkv_bias=True,
            drop=config.resid_pdrop,
            attn_drop=config.attn_pdrop
        )

        self.use_velocity = use_velocity
        if use_velocity:
            self.vel_emb1 = nn.Linear(1, ch1)
            self.vel_emb2 = nn.Linear(1, ch2)
            self.vel_emb3 = nn.Linear(1, ch3)
            self.vel_emb4 = nn.Linear(1, ch4)

        if ch4 != config.perception_output_features:
            self.change_channel_conv = nn.Conv2d(
                ch4,
                config.perception_output_features,
                (1, 1)
            )
        else:
            self.change_channel_conv = nn.Sequential()

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

    def _process_layer(self, lidar_features, self_attn, vel_emb_layer, velocity):
        lidar_embd = self.avgpool_lidar(lidar_features)
        B, C, H, W = lidar_embd.shape
        lidar_tokens = lidar_embd.flatten(2).transpose(1, 2)

        if self.use_velocity:
            vel_emb = vel_emb_layer(velocity).unsqueeze(1)
            lidar_tokens = lidar_tokens + vel_emb

        lidar_tokens = self_attn(lidar_tokens)

        lidar_features_fused = lidar_tokens.transpose(1, 2).reshape(B, C, H, W)
        lidar_features_fused = F.interpolate(
            lidar_features_fused,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        lidar_features = lidar_features + lidar_features_fused
        return lidar_features

    def forward(self, image, lidar, velocity):
        lidar_tensor = lidar

        lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.act1(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)

        lidar_features = self.lidar_encoder._model.layer1(lidar_features)
        lidar_features = self._process_layer(lidar_features, self.self_attn1, self.vel_emb1, velocity)

        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        lidar_features = self._process_layer(lidar_features, self.self_attn2, self.vel_emb2, velocity)

        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        lidar_features = self._process_layer(lidar_features, self.self_attn3, self.vel_emb3, velocity)

        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        lidar_features = self._process_layer(lidar_features, self.self_attn4, self.vel_emb4, velocity)

        lidar_features = self.change_channel_conv(lidar_features)
        image_features_grid = lidar_features

        x4 = lidar_features

        lidar_features = self.lidar_encoder._model.global_pool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)

        fused_features = lidar_features
        features = self.top_down(x4)

        return features, image_features_grid, fused_features
