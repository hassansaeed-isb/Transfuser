"""
CrossViT fusion backbone (crossvit_fusion).

This module is the evaluation/training entry point for the bi-directional
cross-attention fusion architecture (formerly named bi-attenfusion).
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
import timm


class CrossAttention(nn.Module):
    """
    跨模态注意力模块
    允许一个模态的特征作为Query,另一个模态的特征作为Key和Value
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Query来自一个模态,Key和Value来自另一个模态
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        """
        参数:
            x_q: Query特征 (B, N_q, C)
            x_kv: Key-Value特征 (B, N_kv, C)
        返回:
            增强后的Query特征 (B, N_q, C)
        """
        B, N_q, C = x_q.shape
        N_kv = x_kv.shape[1]
        
        # 生成Query
        q = self.q(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # 生成Key和Value
        kv = self.kv(x_kv).reshape(B, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力到Value
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class CrossViTBlock(nn.Module):
    """
    CrossViT块 - 实现双向跨模态注意力
    图像特征和LiDAR特征相互增强
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        
        # 图像到LiDAR的跨注意力
        self.norm1_img = nn.LayerNorm(dim)
        self.norm1_lidar = nn.LayerNorm(dim)
        self.cross_attn_img2lidar = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )
        
        # LiDAR到图像的跨注意力
        self.norm2_img = nn.LayerNorm(dim)
        self.norm2_lidar = nn.LayerNorm(dim)
        self.cross_attn_lidar2img = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        # MLP前馈网络
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

    def forward(self, img_tokens, lidar_tokens):
        """
        双向跨模态注意力融合
        
        参数:
            img_tokens: 图像token (B, N_img, C)
            lidar_tokens: LiDAR token (B, N_lidar, C)
        返回:
            增强后的图像和LiDAR token
        """
        # 第一阶段: 图像查询LiDAR信息
        img_tokens = img_tokens + self.cross_attn_img2lidar(
            self.norm1_img(img_tokens), 
            self.norm1_lidar(lidar_tokens)
        )
        
        # 第二阶段: LiDAR查询图像信息
        lidar_tokens = lidar_tokens + self.cross_attn_lidar2img(
            self.norm2_lidar(lidar_tokens),
            self.norm2_img(img_tokens)
        )
        
        # 第三阶段: 独立的MLP处理
        img_tokens = img_tokens + self.mlp_img(self.norm3_img(img_tokens))
        lidar_tokens = lidar_tokens + self.mlp_lidar(self.norm3_lidar(lidar_tokens))
        
        return img_tokens, lidar_tokens


class MultiScaleCrossViT(nn.Module):
    """
    多尺度CrossViT模块
    在不同尺度上进行跨模态特征融合
    """
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
        """
        通过多个CrossViT块处理特征
        
        参数:
            img_tokens: 图像token (B, N_img, C)
            lidar_tokens: LiDAR token (B, N_lidar, C)
        返回:
            融合后的图像和LiDAR token
        """
        for block in self.blocks:
            img_tokens, lidar_tokens = block(img_tokens, lidar_tokens)
        
        return img_tokens, lidar_tokens


class CrossViTFusionBackbone(nn.Module):
    """
    基于CrossViT的主干网络进行多尺度跨模态融合
    
    参数:
        config: 全局配置对象
        image_architecture: 图像分支架构 (ResNet, RegNet, ConvNext)
        lidar_architecture: LiDAR分支架构
        use_velocity: 是否使用速度输入
    """
    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__()
        self.config = config
        
        # 自适应池化层,统一特征图尺寸
        self.avgpool_img = nn.AdaptiveAvgPool2d((config.img_vert_anchors, config.img_horz_anchors))
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((config.lidar_vert_anchors, config.lidar_horz_anchors))
        
        # 图像编码器 - 使用预训练模型
        self.image_encoder = ImageCNN(
            architecture=image_architecture, 
            normalize=True,
            out_features=config.perception_output_features
        )
        
        # LiDAR编码器
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
        
        # 为每个层级创建融合模块
        self.crossvit1 = MultiScaleCrossViT(
            dim=self.image_encoder.features.feature_info[1]['num_chs'],
            num_heads=config.n_head,
            num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp,
            qkv_bias=True,
            drop=config.resid_pdrop,
            attn_drop=config.attn_pdrop
        )
        
        self.crossvit2 = MultiScaleCrossViT(
            dim=self.image_encoder.features.feature_info[2]['num_chs'],
            num_heads=config.n_head,
            num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp,
            qkv_bias=True,
            drop=config.resid_pdrop,
            attn_drop=config.attn_pdrop
        )
        
        self.crossvit3 = MultiScaleCrossViT(
            dim=self.image_encoder.features.feature_info[3]['num_chs'],
            num_heads=config.n_head,
            num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp,
            qkv_bias=True,
            drop=config.resid_pdrop,
            attn_drop=config.attn_pdrop
        )
        
        self.crossvit4 = MultiScaleCrossViT(
            dim=self.image_encoder.features.feature_info[4]['num_chs'],
            num_heads=config.n_head,
            num_blocks=config.crossvit_blocks,
            mlp_ratio=config.block_exp,
            qkv_bias=True,
            drop=config.resid_pdrop,
            attn_drop=config.attn_pdrop
        )
        
        # 速度嵌入
        self.use_velocity = use_velocity
        if use_velocity:
            self.vel_emb1 = nn.Linear(1, self.image_encoder.features.feature_info[1]['num_chs'])
            self.vel_emb2 = nn.Linear(1, self.image_encoder.features.feature_info[2]['num_chs'])
            self.vel_emb3 = nn.Linear(1, self.image_encoder.features.feature_info[3]['num_chs'])
            self.vel_emb4 = nn.Linear(1, self.image_encoder.features.feature_info[4]['num_chs'])
        
        # 通道数调整卷积
        if self.image_encoder.features.feature_info[4]['num_chs'] != config.perception_output_features:
            self.change_channel_conv_image = nn.Conv2d(
                self.image_encoder.features.feature_info[4]['num_chs'], 
                config.perception_output_features, 
                (1, 1)
            )
            self.change_channel_conv_lidar = nn.Conv2d(
                self.image_encoder.features.feature_info[4]['num_chs'], 
                config.perception_output_features, 
                (1, 1)
            )
        else:
            self.change_channel_conv_image = nn.Sequential()
            self.change_channel_conv_lidar = nn.Sequential()
        
        # FPN特征金字塔网络
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
        """
        FPN自顶向下路径
        
        参数:
            x: 最高层特征图
        返回:
            p2, p3, p4, p5: 多尺度特征金字塔
        """
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample(p4)))
        p2 = self.relu(self.up_conv3(self.upsample(p3)))
        return p2, p3, p4, p5
    
    def forward(self, image, lidar, velocity):
        """
        进行图像和LiDAR特征融合的前向传播
        
        参数:
            image: 输入图像 (B, 3, H, W)
            lidar: 输入LiDAR鸟瞰图 (B, C, H, W)
            velocity: 速度输入 (B, 1)
        返回:
            features: FPN特征金字塔
            image_features_grid: 图像特征网格
            fused_features: 融合后的全局特征向量
        """
        # 图像归一化
        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image
        
        lidar_tensor = lidar
        
        # ===== 初始卷积层 =====
        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.act1(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)
        
        lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.act1(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)
        
        # ===== Layer 1 + CrossViT融合 =====
        image_features = self.image_encoder.features.layer1(image_features)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)
        
        # 池化到统一尺寸
        image_embd = self.avgpool_img(image_features)
        lidar_embd = self.avgpool_lidar(lidar_features)
        
        # 转换为token序列 (B, C, H, W) -> (B, H*W, C)
        B, C1, H_img, W_img = image_embd.shape
        _, _, H_lidar, W_lidar = lidar_embd.shape
        
        img_tokens = image_embd.flatten(2).transpose(1, 2)  # (B, N_img, C)
        lidar_tokens = lidar_embd.flatten(2).transpose(1, 2)  # (B, N_lidar, C)
        
        # 添加速度条件
        if self.use_velocity:
            vel_emb = self.vel_emb1(velocity).unsqueeze(1)  # (B, 1, C)
            img_tokens = img_tokens + vel_emb
            lidar_tokens = lidar_tokens + vel_emb
        
        # 融合
        img_tokens, lidar_tokens = self.crossvit1(img_tokens, lidar_tokens)
        
        # 转换回特征图
        image_features_fused = img_tokens.transpose(1, 2).reshape(B, C1, H_img, W_img)
        lidar_features_fused = lidar_tokens.transpose(1, 2).reshape(B, C1, H_lidar, W_lidar)
        
        # 插值回原始尺寸并残差连接
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
        
        # ===== Layer 2 + CrossViT融合 =====
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
        
        # ===== Layer 3 + CrossViT融合 =====
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
        
        # ===== Layer 4 + CrossViT融合 =====
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
        
        # ===== 通道数调整 =====
        image_features = self.change_channel_conv_image(image_features)
        lidar_features = self.change_channel_conv_lidar(lidar_features)
        
        # 保存特征用于后续处理
        x4 = lidar_features
        image_features_grid = image_features
        
        # ===== 全局池化 =====
        image_features = self.image_encoder.features.global_pool(image_features)
        image_features = torch.flatten(image_features, 1)
        
        lidar_features = self.lidar_encoder._model.global_pool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)
        
        # 融合全局特征
        fused_features = image_features + lidar_features
        
        # 生成FPN特征金字塔
        features = self.top_down(x4)
        
        return features, image_features_grid, fused_features


# 辅助函数和类(从transfuser.py导入)
def normalize_imagenet(x):
    """ImageNet标准化"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape(1, 3, 1, 1)
    return (x - mean) / std


class ImageCNN(nn.Module):
    """图像编码器 - 使用TIMM预训练模型"""
    def __init__(self, architecture, normalize=True, out_features=512):
        super().__init__()
        self.normalize = normalize
        self.features = timm.create_model(architecture, pretrained=True)
        self.features.fc = None
        
        if architecture.startswith('regnet'):
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
        elif architecture.startswith('convnext'):
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
    """LiDAR编码器"""
    def __init__(self, architecture, in_channels, out_features=512):
        super().__init__()
        self._model = timm.create_model(architecture, pretrained=True)
        self._model.fc = None
        
        # 修改第一层卷积以适应LiDAR输入通道数
        if architecture.startswith('regnet'):
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
        else:
            # ResNet架构
            self._model.conv1 = nn.Conv2d(
                in_channels,
                self._model.conv1.out_channels,
                kernel_size=self._model.conv1.kernel_size,
                stride=self._model.conv1.stride,
                padding=self._model.conv1.padding,
                bias=False
            )
            self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
