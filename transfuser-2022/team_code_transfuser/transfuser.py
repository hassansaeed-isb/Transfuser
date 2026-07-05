# 导入必要的库
import math
import torch
from torch import nn
import torch.nn.functional as F
import timm

class TransfuserBackbone(nn.Module):
    """
    多尺度融合Transformer,用于图像和LiDAR特征融合
    
    参数说明:
        image_architecture: 图像分支使用的架构。支持ResNet、RegNet和ConvNext
        lidar_architecture: LiDAR分支使用的架构。支持ResNet、RegNet和ConvNext
        use_velocity: 是否在Transformer中使用速度输入
    """

    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__()
        self.config = config

        # 自适应平均池化层,用于统一图像和LiDAR特征图的尺寸
        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))
        
        # 图像编码器,使用预训练模型并进行ImageNet归一化
        self.image_encoder = ImageCNN(architecture=image_architecture, normalize=True,
                                      out_features=self.config.perception_output_features)

        # 根据配置确定LiDAR输入通道数
        if(config.use_point_pillars == True):
            in_channels = config.num_features[-1]  # 使用PointPillars特征
        else:
            in_channels = 2 * config.lidar_seq_len  # 使用原始LiDAR序列

        # 如果使用目标点图像,增加一个通道
        if(self.config.use_target_point_image == True):
            in_channels += 1

        # LiDAR编码器
        self.lidar_encoder = LidarEncoder(architecture=lidar_architecture, in_channels=in_channels,
                                          out_features=self.config.perception_output_features)

        # Transformer模块1: 用于第1层特征融合
        self.transformer1 = GPT(n_embd=self.image_encoder.features.feature_info[1]['num_chs'],
                            n_head=config.n_head,
                            block_exp=config.block_exp,
                            n_layer=config.n_layer,
                            img_vert_anchors=config.img_vert_anchors,
                            img_horz_anchors=config.img_horz_anchors,
                            lidar_vert_anchors=config.lidar_vert_anchors,
                            lidar_horz_anchors=config.lidar_horz_anchors,
                            seq_len=config.seq_len,
                            embd_pdrop=config.embd_pdrop,
                            attn_pdrop=config.attn_pdrop,
                            resid_pdrop=config.resid_pdrop,
                            config=config, use_velocity=use_velocity)

        # Transformer模块2: 用于第2层特征融合
        self.transformer2 = GPT(n_embd=self.image_encoder.features.feature_info[2]['num_chs'],
                            n_head=config.n_head,
                            block_exp=config.block_exp,
                            n_layer=config.n_layer,
                            img_vert_anchors=config.img_vert_anchors,
                            img_horz_anchors=config.img_horz_anchors,
                            lidar_vert_anchors=config.lidar_vert_anchors,
                            lidar_horz_anchors=config.lidar_horz_anchors,
                            seq_len=config.seq_len,
                            embd_pdrop=config.embd_pdrop,
                            attn_pdrop=config.attn_pdrop,
                            resid_pdrop=config.resid_pdrop,
                            config=config, use_velocity=use_velocity)

        # Transformer模块3: 用于第3层特征融合
        self.transformer3 = GPT(n_embd=self.image_encoder.features.feature_info[3]['num_chs'],
                            n_head=config.n_head,
                            block_exp=config.block_exp,
                            n_layer=config.n_layer,
                            img_vert_anchors=config.img_vert_anchors,
                            img_horz_anchors=config.img_horz_anchors,
                            lidar_vert_anchors=config.lidar_vert_anchors,
                            lidar_horz_anchors=config.lidar_horz_anchors,
                            seq_len=config.seq_len,
                            embd_pdrop=config.embd_pdrop,
                            attn_pdrop=config.attn_pdrop,
                            resid_pdrop=config.resid_pdrop,
                            config=config, use_velocity=use_velocity)

        # Transformer模块4: 用于第4层特征融合
        self.transformer4 = GPT(n_embd=self.image_encoder.features.feature_info[4]['num_chs'],
                            n_head=config.n_head,
                            block_exp=config.block_exp,
                            n_layer=config.n_layer,
                            img_vert_anchors=config.img_vert_anchors,
                            img_horz_anchors=config.img_horz_anchors,
                            lidar_vert_anchors=config.lidar_vert_anchors,
                            lidar_horz_anchors=config.lidar_horz_anchors,
                            seq_len=config.seq_len,
                            embd_pdrop=config.embd_pdrop,
                            attn_pdrop=config.attn_pdrop,
                            resid_pdrop=config.resid_pdrop,
                            config=config, use_velocity=use_velocity)

        # 如果最后一层的通道数与目标输出特征数不匹配,使用1x1卷积调整通道数
        if(self.image_encoder.features.feature_info[4]['num_chs'] != self.config.perception_output_features):
            self.change_channel_conv_image = nn.Conv2d(self.image_encoder.features.feature_info[4]['num_chs'], self.config.perception_output_features, (1, 1))
            self.change_channel_conv_lidar = nn.Conv2d(self.image_encoder.features.feature_info[4]['num_chs'], self.config.perception_output_features, (1, 1))
        else:
            # 如果通道数已经匹配,使用空的Sequential模块
            self.change_channel_conv_image = nn.Sequential()
            self.change_channel_conv_lidar = nn.Sequential()

        # FPN(特征金字塔网络)融合模块
        channel = self.config.bev_features_chanels
        self.relu = nn.ReLU(inplace=True)
        
        # 自顶向下路径的上采样和卷积层
        self.upsample = nn.Upsample(scale_factor=self.config.bev_upsample_factor, mode='bilinear', align_corners=False)
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))
        
        # 横向连接的卷积层
        self.c5_conv = nn.Conv2d(self.config.perception_output_features, channel, (1, 1))
        
    def top_down(self, x):
        """
        FPN自顶向下路径,逐步上采样并融合特征
        
        参数:
            x: 输入的高层特征图
        返回:
            p2, p3, p4, p5: 不同尺度的特征金字塔
        """
        p5 = self.relu(self.c5_conv(x))  # 最高层特征
        p4 = self.relu(self.up_conv5(self.upsample(p5)))  # 上采样并融合
        p3 = self.relu(self.up_conv4(self.upsample(p4)))  # 继续上采样
        p2 = self.relu(self.up_conv3(self.upsample(p3)))  # 最低层特征
        
        return p2, p3, p4, p5

    def forward(self, image, lidar, velocity):
        """
        使用Transformer进行图像和LiDAR特征融合的前向传播
        
        参数:
            image: 输入图像张量
            lidar: 输入LiDAR鸟瞰图张量
            velocity: 来自速度计的速度输入
            
        返回:
            features: FPN特征金字塔
            image_features_grid: 图像特征网格(用于辅助任务)
            fused_features: 融合后的全局特征向量
        """

        # 如果需要归一化,对图像进行ImageNet标准化
        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image

        lidar_tensor = lidar

        # ===== 初始卷积层 =====
        # 图像分支: conv1 -> bn1 -> act1 -> maxpool
        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.act1(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)
        
        # LiDAR分支: conv1 -> bn1 -> act1 -> maxpool
        lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.act1(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)

        # ===== Layer 1 特征提取 =====
        image_features = self.image_encoder.features.layer1(image_features)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)

        # Layer 1 融合: 图像特征尺寸约为(B, 72, 40, 176), LiDAR特征尺寸约为(B, 72, 64, 64)
        # 使用自适应平均池化统一尺寸
        image_embd_layer1 = self.avgpool_img(image_features)
        lidar_embd_layer1 = self.avgpool_lidar(lidar_features)

        # 通过Transformer进行跨模态特征融合
        image_features_layer1, lidar_features_layer1 = self.transformer1(image_embd_layer1, lidar_embd_layer1, velocity)
        
        # 将融合后的特征插值回原始尺寸
        image_features_layer1 = F.interpolate(image_features_layer1, size=(image_features.shape[2],image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_layer1 = F.interpolate(lidar_features_layer1, size=(lidar_features.shape[2],lidar_features.shape[3]), mode='bilinear', align_corners=False)
        
        # 残差连接: 将融合特征加到原始特征上
        image_features = image_features + image_features_layer1
        lidar_features = lidar_features + lidar_features_layer1

        # ===== Layer 2 特征提取与融合 =====
        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        
        # Layer 2 融合: 图像特征尺寸约为(B, 216, 20, 88), LiDAR特征尺寸约为(B, 216, 32, 32)
        image_embd_layer2 = self.avgpool_img(image_features)
        lidar_embd_layer2 = self.avgpool_lidar(lidar_features)
        image_features_layer2, lidar_features_layer2 = self.transformer2(image_embd_layer2, lidar_embd_layer2, velocity)
        image_features_layer2 = F.interpolate(image_features_layer2, size=(image_features.shape[2],image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_layer2 = F.interpolate(lidar_features_layer2, size=(lidar_features.shape[2],lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer2
        lidar_features = lidar_features + lidar_features_layer2

        # ===== Layer 3 特征提取与融合 =====
        image_features = self.image_encoder.features.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        
        # Layer 3 融合: 图像特征尺寸约为(B, 576, 10, 44), LiDAR特征尺寸约为(B, 576, 16, 16)
        image_embd_layer3 = self.avgpool_img(image_features)
        lidar_embd_layer3 = self.avgpool_lidar(lidar_features)
        image_features_layer3, lidar_features_layer3 = self.transformer3(image_embd_layer3, lidar_embd_layer3, velocity)
        image_features_layer3 = F.interpolate(image_features_layer3, size=(image_features.shape[2],image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_layer3 = F.interpolate(lidar_features_layer3, size=(lidar_features.shape[2],lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer3
        lidar_features = lidar_features + lidar_features_layer3

        # ===== Layer 4 特征提取与融合 =====
        image_features = self.image_encoder.features.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        
        # Layer 4 融合: 图像特征尺寸约为(B, 1512, 5, 22), LiDAR特征尺寸约为(B, 1512, 8, 8)
        image_embd_layer4 = self.avgpool_img(image_features)
        lidar_embd_layer4 = self.avgpool_lidar(lidar_features)

        image_features_layer4, lidar_features_layer4 = self.transformer4(image_embd_layer4, lidar_embd_layer4, velocity)
        image_features_layer4 = F.interpolate(image_features_layer4, size=(image_features.shape[2],image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_layer4 = F.interpolate(lidar_features_layer4, size=(lidar_features.shape[2],lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer4
        lidar_features = lidar_features + lidar_features_layer4

        # ===== 通道数调整 =====
        # 将通道数下采样到配置的输出特征数(通常为512)
        image_features = self.change_channel_conv_image(image_features)
        lidar_features = self.change_channel_conv_lidar(lidar_features)

        # 保存LiDAR特征用于FPN
        x4 = lidar_features
        # 保存图像特征网格用于辅助任务
        image_features_grid = image_features

        # ===== 全局池化 =====
        # 对图像特征进行全局平均池化并展平
        image_features = self.image_encoder.features.global_pool(image_features)
        image_features = torch.flatten(image_features, 1)
        
        # 对LiDAR特征进行全局平均池化并展平
        lidar_features = self.lidar_encoder._model.global_pool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)

        # 融合图像和LiDAR的全局特征
        fused_features = image_features + lidar_features

        # 通过FPN自顶向下路径生成多尺度特征
        features = self.top_down(x4)
        return features, image_features_grid, fused_features


class SegDecoder(nn.Module):
    def __init__(self, config, latent_dim=512):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.num_class = config.num_class

        self.deconv1 = nn.Sequential(
                    nn.Conv2d(self.latent_dim, self.config.deconv_channel_num_1, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.config.deconv_channel_num_1, self.config.deconv_channel_num_2, 3, 1, 1),
                    nn.ReLU(True),
                    )
        self.deconv2 = nn.Sequential(
                    nn.Conv2d(self.config.deconv_channel_num_2, self.config.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    )
        self.deconv3 = nn.Sequential(
                    nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.config.deconv_channel_num_3, self.num_class, 3, 1, 1),
                    )

    def forward(self, x):
        x = self.deconv1(x)
        x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_1, mode='bilinear', align_corners=False)
        x = self.deconv2(x)
        x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_2, mode='bilinear', align_corners=False)
        x = self.deconv3(x)
        x = F.interpolate(x, size=(self.config.img_resolution[0], self.config.img_resolution[1]), mode='bilinear', align_corners=False)

        return x


class DepthDecoder(nn.Module):
    """
    深度估计解码器
    将潜在特征解码为单通道深度图
    """
    def __init__(self, config, latent_dim=512):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim  # 输入潜在特征的维度

        # 第一个反卷积块: 两层3x3卷积 + ReLU
        self.deconv1 = nn.Sequential(
                    nn.Conv2d(self.latent_dim, self.config.deconv_channel_num_1, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.config.deconv_channel_num_1, self.config.deconv_channel_num_2, 3, 1, 1),
                    nn.ReLU(True),
                    )
        # 第二个反卷积块: 两层3x3卷积 + ReLU
        self.deconv2 = nn.Sequential(
                    nn.Conv2d(self.config.deconv_channel_num_2, self.config.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    )
        # 第三个反卷积块: 输出单通道深度图
        self.deconv3 = nn.Sequential(
                    nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.config.deconv_channel_num_3, 1, 3, 1, 1),  # 输出1个通道
                    )

    def forward(self, x):
        """
        前向传播: 逐步上采样并解码为深度图
        
        参数:
            x: 输入的潜在特征 (B, latent_dim, H, W)
        返回:
            深度预测结果 (B, H', W'), 值域为[0, 1]
        """
        x = self.deconv1(x)
        x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_1, mode='bilinear', align_corners=False)
        x = self.deconv2(x)
        x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_2, mode='bilinear', align_corners=False)
        x = self.deconv3(x)
        x = F.interpolate(x, size=(self.config.img_resolution[0], self.config.img_resolution[1]), mode='bilinear', align_corners=False)
        x = torch.sigmoid(x).squeeze(1)  # 使用sigmoid将深度值归一化到[0,1],并去除通道维度

        return x


class GPT(nn.Module):
    """
    基于GPT架构的多模态融合Transformer
    用于融合图像和LiDAR特征,支持速度条件输入
    """

    def __init__(self, n_embd, n_head, block_exp, n_layer, 
                    img_vert_anchors, img_horz_anchors, 
                    lidar_vert_anchors, lidar_horz_anchors,
                    seq_len, 
                    embd_pdrop, attn_pdrop, resid_pdrop, config, use_velocity=True):
        super().__init__()
        self.n_embd = n_embd  # 嵌入维度
        # 当前仅支持序列长度为1
        self.seq_len = 1
        
        # 图像和LiDAR的空间锚点数量
        self.img_vert_anchors = img_vert_anchors  # 图像垂直锚点数
        self.img_horz_anchors = img_horz_anchors  # 图像水平锚点数
        self.lidar_vert_anchors = lidar_vert_anchors  # LiDAR垂直锚点数
        self.lidar_horz_anchors = lidar_horz_anchors  # LiDAR水平锚点数
        self.config = config

        # 可学习的位置嵌入参数,覆盖图像和LiDAR的所有空间位置
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len * img_vert_anchors * img_horz_anchors + self.seq_len * lidar_vert_anchors * lidar_horz_anchors, n_embd))
        
        # 速度嵌入模块
        self.use_velocity = use_velocity
        if(use_velocity == True):
            self.vel_emb = nn.Linear(self.seq_len, n_embd)  # 将速度投影到嵌入空间

        # Dropout层用于嵌入正则化
        self.drop = nn.Dropout(embd_pdrop)

        # Transformer块序列
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, 
                        block_exp, attn_pdrop, resid_pdrop)
                        for layer in range(n_layer)])
        
        # 最终的LayerNorm层
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = self.seq_len
        # 应用权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        初始化模型权重
        - 线性层: 使用正态分布初始化权重,偏置置零
        - LayerNorm: 偏置置零,权重设为配置值
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=self.config.gpt_linear_layer_init_mean, std=self.config.gpt_linear_layer_init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_tensor, lidar_tensor, velocity):
        """
        前向传播: 融合图像和LiDAR特征
        
        参数:
            image_tensor: 图像特征张量 (B, C, H, W)
            lidar_tensor: LiDAR特征张量 (B, C, H, W)
            velocity: 自车速度 (B, seq_len)
            
        返回:
            image_tensor_out: 增强后的图像特征 (B, C, H, W)
            lidar_tensor_out: 增强后的LiDAR特征 (B, C, H, W)
        """
        
        bz = lidar_tensor.shape[0]  # 批次大小
        lidar_h, lidar_w = lidar_tensor.shape[2:4]  # LiDAR特征图尺寸
        img_h, img_w = image_tensor.shape[2:4]  # 图像特征图尺寸
        
        # 确保序列长度为1
        assert self.seq_len == 1
        
        # 将图像和LiDAR特征重塑为序列形式: (B, seq_len, C, H, W) -> (B, H*W, C)
        image_tensor = image_tensor.view(bz, self.seq_len, -1, img_h, img_w).permute(0,1,3,4,2).contiguous().view(bz, -1, self.n_embd)
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, lidar_h, lidar_w).permute(0,1,3,4,2).contiguous().view(bz, -1, self.n_embd)

        # 拼接图像和LiDAR的token嵌入
        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

        # 将速度投影到嵌入空间
        if(self.use_velocity==True):
            velocity_embeddings = self.vel_emb(velocity)  # (B, C)
            # 添加位置嵌入和速度嵌入到所有token
            x = self.drop(self.pos_emb + token_embeddings + velocity_embeddings.unsqueeze(1))  # (B, num_tokens, C)
        else:
            # 仅添加位置嵌入
            x = self.drop(self.pos_emb + token_embeddings)
            
        # 通过Transformer块处理
        x = self.blocks(x)  # (B, num_tokens, C)
        # 最终LayerNorm
        x = self.ln_f(x)  # (B, num_tokens, C)

        # 将输出重塑回原始形状
        x = x.view(bz, self.seq_len*self.img_vert_anchors*self.img_horz_anchors + self.seq_len*self.lidar_vert_anchors*self.lidar_horz_anchors, self.n_embd)

        # 分离图像和LiDAR特征,并恢复空间维度
        image_tensor_out = x[:, :self.seq_len*self.img_vert_anchors*self.img_horz_anchors, :].contiguous().view(bz * self.seq_len, -1, img_h, img_w)
        lidar_tensor_out = x[:, self.seq_len*self.img_vert_anchors*self.img_horz_anchors:, :].contiguous().view(bz * self.seq_len, -1, lidar_h, lidar_w)

        return image_tensor_out, lidar_tensor_out

        
class ImageCNN(nn.Module):
    """
    图像输入的编码器网络
    使用TIMM库中的预训练视觉架构
    
    参数:
        architecture: 使用的视觉架构名称(如resnet34, regnet等)
        normalize: 是否对输入图像进行归一化
        out_features: 输出特征维度
    """

    def __init__(self, architecture, normalize=True, out_features=512):
        super().__init__()
        self.normalize = normalize
        # 从TIMM库创建预训练模型
        self.features = timm.create_model(architecture, pretrained=True)
        self.features.fc = None  # 移除全连接层
        
        # 针对不同架构进行模块重命名,以便使用统一的代码
        if (architecture.startswith('regnet')):  # RegNet架构
            self.features.conv1 = self.features.stem.conv
            self.features.bn1  = self.features.stem.bn
            self.features.act1 = nn.Sequential()  # ReLU已包含在BatchNorm中
            self.features.maxpool =  nn.Sequential()
            self.features.layer1 =self.features.s1
            self.features.layer2 =self.features.s2
            self.features.layer3 =self.features.s3
            self.features.layer4 =self.features.s4
            self.features.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.features.head = nn.Sequential()

        elif (architecture.startswith('convnext')):  # ConvNext架构
            self.features.conv1 = self.features.stem._modules['0']
            self.features.bn1 = self.features.stem._modules['1']
            self.features.act1 = nn.Sequential()  # ConvNext的stem后没有激活函数
            self.features.maxpool = nn.Sequential()
            self.features.layer1 = self.features.stages._modules['0']
            self.features.layer2 = self.features.stages._modules['1']
            self.features.layer3 = self.features.stages._modules['2']
            self.features.layer4 = self.features.stages._modules['3']
            self.features.global_pool = self.features.head
            self.features.global_pool.flatten = nn.Sequential()
            self.features.global_pool.fc = nn.Sequential()
            self.features.head = nn.Sequential()
            
            # ConvNext没有ResNet的第0层,需要调整feature_info索引
            self.features.feature_info.append(self.features.feature_info[3])
            self.features.feature_info[3] = self.features.feature_info[2]
            self.features.feature_info[2] = self.features.feature_info[1]
            self.features.feature_info[1] = self.features.feature_info[0]

            # 重新初始化LayerNorm以匹配输出特征维度
            _tmp = self.features.global_pool.norm
            self.features.global_pool.norm = nn.LayerNorm((out_features,1,1), _tmp.eps, _tmp.elementwise_affine)


def normalize_imagenet(x):
    """
    使用ImageNet标准对输入图像进行归一化
    
    参数:
        x: 输入图像张量,像素值范围[0, 255]
        
    返回:
        归一化后的图像张量
    """
    x = x.clone()
    # 对RGB三个通道分别进行归一化
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229  # R通道
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224  # G通道
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225  # B通道
    return x


class LidarEncoder(nn.Module):
    """
    LiDAR输入的编码器网络
    使用TIMM库中的视觉架构,但修改第一层卷积以适应LiDAR输入通道数
    
    参数:
        architecture: 使用的视觉架构名称
        in_channels: LiDAR输入通道数(默认为2)
        out_features: 输出特征维度
    """

    def __init__(self, architecture, in_channels=2, out_features=512):
        super().__init__()

        # 创建模型(不使用预训练权重,因为输入通道数不同)
        self._model = timm.create_model(architecture, pretrained=False)
        self._model.fc = None

        # 针对不同架构进行模块重命名
        if (architecture.startswith('regnet')):  # RegNet架构
            self._model.conv1 = self._model.stem.conv
            self._model.bn1  = self._model.stem.bn
            self._model.act1 = nn.Sequential()
            self._model.maxpool =  nn.Sequential()
            self._model.layer1 = self._model.s1
            self._model.layer2 = self._model.s2
            self._model.layer3 = self._model.s3
            self._model.layer4 = self._model.s4
            self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self._model.head = nn.Sequential()

        elif (architecture.startswith('convnext')):  # ConvNext架构
            self._model.conv1 = self._model.stem._modules['0']
            self._model.bn1 = self._model.stem._modules['1']
            self._model.act1 = nn.Sequential()
            self._model.maxpool = nn.Sequential()
            self._model.layer1 = self._model.stages._modules['0']
            self._model.layer2 = self._model.stages._modules['1']
            self._model.layer3 = self._model.stages._modules['2']
            self._model.layer4 = self._model.stages._modules['3']
            self._model.global_pool = self._model.head
            self._model.global_pool.flatten = nn.Sequential()
            self._model.global_pool.fc = nn.Sequential()
            self._model.head = nn.Sequential()
            _tmp = self._model.global_pool.norm
            self._model.global_pool.norm = nn.LayerNorm((out_features,1,1), _tmp.eps, _tmp.elementwise_affine)

        # 修改第一层卷积以匹配LiDAR的输入通道数
        _tmp = self._model.conv1
        use_bias = (_tmp.bias != None)
        # [修改] 2026-03-18: 保存 bias 引用，在 del _tmp 之前提取，避免 del 后访问已删除对象
        bias_data = _tmp.bias if use_bias else None
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels,
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=use_bias)
        
        # 删除旧的卷积层以避免未使用的参数
        if architecture.startswith('convnext'):
          del self._model.stem._modules['0']
        elif architecture.startswith('regnet'):
          del self._model.stem.conv
        del _tmp
        # [修改] 2026-03-18: 添加 CUDA 可用性检查，避免在 CPU 环境下调用 CUDA API
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 如果有偏置,保留原始偏置
        if(use_bias):
            self._model.conv1.bias = bias_data


class SelfAttention(nn.Module):
    """
    标准的多头自注意力层
    包含Q、K、V投影和输出投影
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0  # 确保嵌入维度可以被头数整除
        
        # Q、K、V的线性投影层
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        
        # 正则化层
        self.attn_drop = nn.Dropout(attn_pdrop)  # 注意力权重的dropout
        self.resid_drop = nn.Dropout(resid_pdrop)  # 残差连接的dropout
        
        # 输出投影层
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        """
        前向传播: 计算多头自注意力
        
        参数:
            x: 输入张量 (B, T, C)
            
        返回:
            注意力输出 (B, T, C)
        """
        B, T, C = x.size()

        # 计算所有头的Q、K、V,并将头维度移到批次维度
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # 计算缩放点积注意力: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)  # 对注意力权重进行softmax
        att = self.attn_drop(att)  # 应用dropout
        
        # 加权求和: (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v
        # 重新组合所有头的输出
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影和dropout
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """
    标准的Transformer块
    包含自注意力层和前馈网络(MLP),都带有残差连接和LayerNorm
    """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)  # 自注意力前的LayerNorm
        self.ln2 = nn.LayerNorm(n_embd)  # MLP前的LayerNorm
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        
        # MLP前馈网络: 线性层 -> ReLU -> 线性层 -> Dropout
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),  # 扩展维度
            nn.ReLU(True),  # 使用ReLU而非GELU
            nn.Linear(block_exp * n_embd, n_embd),  # 恢复维度
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        """
        前向传播: Pre-LayerNorm + 残差连接
        
        参数:
            x: 输入张量 (B, T, C)
            
        返回:
            输出张量 (B, T, C)
        """
        x = x + self.attn(self.ln1(x))  # 自注意力 + 残差
        x = x + self.mlp(self.ln2(x))  # MLP + 残差

        return x
