"""
主模型模块 (model.py)

功能描述:
    定义TransFuser自动驾驶系统的主要模型组件
    包括目标检测头、感知模型和完整的自动驾驶模型
    
主要类:
    LidarCenterNetHead: 基于CenterNet的LiDAR目标检测头
        - 检测3D目标（车辆、行人等）
        - 预测中心热图、宽高、偏移、方向角、速度和刹车
    
    PerceptionModel: 感知模型
        - 融合图像和LiDAR特征
        - 支持多种backbone: transFuser, late_fusion, latentTF,
          geometric_fusion, crossvit_fusion
        - 输出BEV特征、语义分割、深度估计等
    
    LidarCenterNet: 完整的自动驾驶模型
        - 集成感知模型和检测头
        - 支持训练和推理两种模式
        - 输出路点预测、目标检测等结果

支持的Backbone:
    - transFuser: 基于Transformer的图像-LiDAR融合
    - late_fusion: 后期融合（特征级别）
    - latentTF: 潜在空间Transformer融合
    - geometric_fusion: 几何感知融合
    - crossvit_fusion: CrossViT跨模态注意力融合（新增）

依赖:
    - PyTorch: 深度学习框架
    - mmcv/mmdet: 目标检测工具库
    - PIL: 图像处理
    - OpenCV: 计算机视觉

使用示例:
    from config import GlobalConfig
    from model import LidarCenterNet
    
    config = GlobalConfig(setting='eval')
    model = LidarCenterNet(config, backbone='crossvit_fusion')
    
    # 推理
    output = model.forward(image, lidar, target_point, ego_vel)

"""

from collections import deque
import torch.nn.functional as F
import cv2

from utils import *
from transfuser import TransfuserBackbone, SegDecoder, DepthDecoder
from geometric_fusion import GeometricFusionBackbone
from late_fusion import LateFusionBackbone
from latentTF import latentTFBackbone
from crossvit_fusion import CrossViTFusionBackbone  # CrossViT融合模块（新增）
from single_modal_backbone import ImageOnlyBackbone, LidarOnlyBackbone  # 单模态Backbone（消融实验）
from no_bi_attn_crossvit import NoBiAttnCrossViTBackbone  # 消融: 移除双向注意力
from no_multiscale_crossvit import NoMultiScaleCrossViTBackbone  # 消融: 移除多尺度融合
from no_vel_crossvit import NoVelCrossViTBackbone  # 消融: 移除速度嵌入
from no_bi_ms_crossvit import NoBiMsCrossViTBackbone  # 消融: 同时移除双向注意力和多尺度融合
from std_crossvit import StdCrossViTBackbone  # 消融: 标准交叉注意力
from no_attn_crossvit import NoAttnCrossViTBackbone  # 消融: 移除跨注意力(4层MLP)
from no_attn_no_ms_crossvit import NoAttnNoMsCrossViTBackbone  # 消融: 无注意力+无多尺度
from copy import deepcopy
from point_pillar import PointPillarNet


from PIL import Image, ImageFont, ImageDraw
from torchvision import models

import torch
import torch.nn as nn
import functools

try:
    from mmcv.cnn import bias_init_with_prob, normal_init
except ImportError:
    from mmengine.model import bias_init_with_prob, normal_init

from mmcv.ops import batched_nms


def force_fp32(apply_to=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if apply_to is not None:
                new_args = list(args)
                new_kwargs = dict(kwargs)
                import inspect
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                for i, arg in enumerate(new_args[1:], 1):
                    param_name = param_names[i] if i < len(param_names) else None
                    if param_name in apply_to:
                        if isinstance(arg, torch.Tensor):
                            new_args[i] = arg.float()
                        elif isinstance(arg, (list, tuple)):
                            new_args[i] = type(arg)(
                                t.float() if isinstance(t, torch.Tensor) else t for t in arg
                            )
                for key in apply_to:
                    if key in new_kwargs:
                        val = new_kwargs[key]
                        if isinstance(val, torch.Tensor):
                            new_kwargs[key] = val.float()
                        elif isinstance(val, (list, tuple)):
                            new_kwargs[key] = type(val)(
                                t.float() if isinstance(t, torch.Tensor) else t for t in val
                            )
                with torch.amp.autocast('cuda', enabled=False):
                    return func(*new_args, **new_kwargs)
            else:
                with torch.amp.autocast('cuda', enabled=False):
                    return func(*args, **kwargs)
        return wrapper
    return decorator


try:
    from mmdet.core import multi_apply
except ImportError:
    try:
        from mmdet.utils import multi_apply
    except ImportError:
        def multi_apply(func, *args, **kwargs):
            pfunc = functools.partial(func, **kwargs) if kwargs else func
            map_results = map(pfunc, *args)
            return tuple(map(list, zip(*map_results)))

try:
    from mmdet.models import HEADS, build_loss
except ImportError:
    from mmdet.registry import MODELS as HEADS
    try:
        from mmdet.models import build_loss
    except ImportError:
        build_loss = HEADS.build

try:
    from mmdet.models.utils import gaussian_radius, gen_gaussian_target
    from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                         transpose_and_gather_feat)
except ImportError:
    try:
        from mmdet.utils.misc import gaussian_radius
    except ImportError:
        def gaussian_radius(det_size, min_overlap=0.5):
            height, width = det_size
            a1 = 1
            b1 = (height + width)
            c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
            sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
            r1 = (b1 + sq1) / 2
            a2 = 4 * min_overlap
            b2 = 2 * (height + width)
            c2 = (1 - min_overlap) * width * height
            sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
            r2 = (b2 + sq2) / 2
            a3 = 4 * min_overlap
            b3 = -2 * (height + width)
            c3 = (1 - min_overlap) * width * height
            sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
            r3 = (b3 + sq3) / 2
            return min(r1, r2, r3)
    from mmdet.models.utils.gaussian_target import (gen_gaussian_target, get_local_maximum,
                                                     get_topk_from_heatmap, transpose_and_gather_feat)

try:
    from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
    from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin
except ImportError:
    from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
    class BBoxTestMixin:
        pass


@HEADS.register_module()
class LidarCenterNetHead(BaseDenseHead, BBoxTestMixin):
    """
    基于CenterNet的LiDAR目标检测头 (Objects as Points)
    
    功能:
        使用中心点表示法检测3D目标（车辆、行人等）
        论文: https://arxiv.org/abs/1904.07850
    
    检测输出:
        - center_heatmap: 目标中心点热图 (B, num_classes, H, W)
        - wh: 目标宽高 (B, 2, H, W)
        - offset: 中心点偏移 (B, 2, H, W)
        - yaw_class: 方向角分类 (B, num_dir_bins, H, W)
        - yaw_res: 方向角残差 (B, 1, H, W)
        - velocity: 速度预测 (B, 1, H, W)
        - brake: 刹车预测 (B, 2, H, W)
    
    参数:
        in_channel (int): 输入特征图的通道数
        feat_channel (int): 中间特征图的通道数
        num_classes (int): 目标类别数（不含背景）
        loss_center_heatmap (dict): 中心热图损失配置，默认GaussianFocalLoss
        loss_wh (dict): 宽高损失配置，默认L1Loss
        loss_offset (dict): 偏移损失配置，默认L1Loss
        loss_dir_class (dict): 方向角分类损失配置，默认CrossEntropyLoss
        loss_dir_res (dict): 方向角残差损失配置，默认SmoothL1Loss
        loss_velocity (dict): 速度损失配置，默认L1Loss
        loss_brake (dict): 刹车损失配置，默认CrossEntropyLoss
        train_cfg (dict): 训练配置（包含num_dir_bins等参数）
        test_cfg (dict): 测试配置
        init_cfg (dict): 初始化配置
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 loss_dir_class=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss_dir_res=dict(type='SmoothL1Loss', loss_weight=1.0),
                 loss_velocity=dict(type='L1Loss', loss_weight=1.0),
                 loss_brake=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(LidarCenterNetHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel,
                                             num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)
        self.num_dir_bins = train_cfg.num_dir_bins
        self.yaw_class_head = self._build_head(in_channel, feat_channel, self.num_dir_bins)
        self.yaw_res_head = self._build_head(in_channel, feat_channel, 1)
        self.velocity_head = self._build_head(in_channel, feat_channel, 1)
        self.brake_head = self._build_head(in_channel, feat_channel, 2)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.loss_dir_class = build_loss(loss_dir_class)
        self.loss_dir_res = build_loss(loss_dir_res)
        self.loss_velocity = build_loss(loss_velocity)
        self.loss_brake = build_loss(loss_brake)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = train_cfg.fp16_enabled
        self.i = 0

    def _build_head(self, in_channel, feat_channel, out_channel):
        """为每个分支构建检测头。"""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """初始化检测头的权重。"""
        bias_init = bias_init_with_prob(self.train_cfg.center_net_bias_init_with_prob)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=self.train_cfg.center_net_normal_init_std)

    def forward(self, feats):
        """前向传播特征。注意CenterNet头不使用FPN。

        参数:
            feats (tuple[Tensor]): 来自上游网络的特征，每个是4D张量。

        返回:
            center_heatmap_preds (List[Tensor]): 所有层级的中心预测热图，通道数为num_classes。
            wh_preds (List[Tensor]): 所有层级的宽高预测，通道数为2。
            offset_preds (List[Tensor]): 所有层级的偏移预测，通道数为2。
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        """单层级特征的前向传播。

        参数:
            feat (Tensor): 单层级的特征。

        返回:
            center_heatmap_pred (Tensor): 中心预测热图，通道数为num_classes。
            wh_pred (Tensor): 宽高预测，通道数为2。
            offset_pred (Tensor): 偏移预测，通道数为2。
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        yaw_class_pred = self.yaw_class_head(feat)
        yaw_res_pred = self.yaw_res_head(feat)
        velocity_pred = self.velocity_head(feat)
        brake_pred = self.brake_head(feat)

        return center_heatmap_pred, wh_pred, offset_pred, yaw_class_pred, yaw_res_pred, velocity_pred, brake_pred

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds', 'yaw_class_preds', 'yaw_res_preds', 'velocity_pred', 'brake_pred'))
    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             yaw_class_preds,
             yaw_res_preds,
             velocity_preds,
             brake_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """计算检测头的损失。

        参数:
            center_heatmap_preds (list[Tensor]): 所有层级的中心预测热图，shape (B, num_classes, H, W)。
            wh_preds (list[Tensor]): 所有层级的宽高预测，shape (B, 2, H, W)。
            offset_preds (list[Tensor]): 所有层级的偏移预测，shape (B, 2, H, W)。
            gt_bboxes (list[Tensor]): 每张图像的真实边界框，shape (num_gts, 4)，格式为[tl_x, tl_y, br_x, br_y]。
            gt_labels (list[Tensor]): 每个框对应的类别索引。
            img_metas (list[dict]): 每张图像的元信息，如图像大小、缩放因子等。
            gt_bboxes_ignore (None | list[Tensor]): 指定计算损失时可忽略的边界框，默认为None。

        返回:
            dict[str, Tensor]: 包含以下组件:
                - loss_center_heatmap (Tensor): 中心热图损失。
                - loss_wh (Tensor): 宽高热图损失。
                - loss_offset (Tensor): 偏移热图损失。
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]
        yaw_class_pred = yaw_class_preds[0]
        yaw_res_pred = yaw_res_preds[0]
        velocity_pred = velocity_preds[0]
        brake_pred = brake_preds[0]

        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels, gt_bboxes_ignore,
                                                     center_heatmap_pred.shape)
        
        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        yaw_class_target = target_result['yaw_class_target']
        yaw_res_target = target_result['yaw_res_target']
        offset_target = target_result['offset_target']
        velocity_target = target_result['velocity_target']
        brake_target = target_result['brake_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']

        # 由于wh_target和offset_target的通道数为2，loss_center_heatmap的avg_factor
        # 始终是loss_wh和loss_offset的1/2
        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
        loss_wh = self.loss_wh(
            wh_pred,
            wh_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        loss_offset = self.loss_offset(
            offset_pred,
            offset_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        loss_yaw_class = self.loss_dir_class(
            yaw_class_pred,
            yaw_class_target,
            wh_offset_target_weight[:, :1, ...],
            avg_factor=avg_factor)
        loss_yaw_res = self.loss_dir_res(
            yaw_res_pred,
            yaw_res_target,
            wh_offset_target_weight[:, :1, ...],
            avg_factor=avg_factor)
        loss_velocity = self.loss_velocity(
            velocity_pred,
            velocity_target,
            wh_offset_target_weight[:, :1, ...],
            avg_factor=avg_factor)
        loss_brake = self.loss_brake(
            brake_pred,
            brake_target,
            wh_offset_target_weight[:, :1, ...],
            avg_factor=avg_factor)

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
            loss_yaw_class=loss_yaw_class,
            loss_yaw_res=loss_yaw_res,
            loss_velocity=loss_velocity,
            loss_brake=loss_brake)

    def loss_by_feat(self,
                     center_heatmap_preds,
                     wh_preds,
                     offset_preds,
                     yaw_class_preds,
                     yaw_res_preds,
                     velocity_preds,
                     brake_preds,
                     gt_bboxes,
                     gt_labels,
                     img_metas,
                     gt_bboxes_ignore=None):
        return self.loss(
            center_heatmap_preds,
            wh_preds,
            offset_preds,
            yaw_class_preds,
            yaw_res_preds,
            velocity_preds,
            brake_preds,
            gt_bboxes,
            gt_labels,
            img_metas,
            gt_bboxes_ignore)

    def angle2class(self, angle):
        """将连续角度转换为离散类别和残差。
        将连续角度转换为离散类别和从类别中心角度到当前角度的小回归数。
        参数:
            angle (torch.Tensor): 角度范围为0-2pi（或-pi~pi），
                类别中心在0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)。
        返回:
            tuple: 编码后的离散类别和残差。
        """
        angle = angle % (2 * np.pi)
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)

        angle_cls = torch.div(shifted_angle, angle_per_class, rounding_mode="trunc")
        angle_res = shifted_angle - (angle_cls * angle_per_class + angle_per_class / 2)
        return angle_cls.long(), angle_res

    def class2angle(self, angle_cls, angle_res, limit_period=True):
        """angle2class的逆函数。
        参数:
            angle_cls (torch.Tensor): 要解码的角度类别。
            angle_res (torch.Tensor): 要解码的角度残差。
            limit_period (bool): 是否将角度限制在[-pi, pi]范围内。
        返回:
            torch.Tensor: 从angle_cls和angle_res解码的角度。
        """
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        angle_center = angle_cls.float() * angle_per_class
        angle = angle_center + angle_res
        if limit_period:
            angle[angle > np.pi] -= 2 * np.pi
        return angle

    def get_targets(self, gt_bboxes, gt_labels, gt_ignores, feat_shape):
        """计算多张图像中的回归和分类目标。

        参数:
            gt_bboxes (list[Tensor]): 每张图像的真实边界框，shape (num_gts, 4)，格式为[tl_x, tl_y, br_x, br_y]。
            gt_labels (list[Tensor]): 每个框对应的类别索引。
            feat_shape (list[int]): 特征图形状，值为[B, _, H, W]。
            img_shape (list[int]): 图像形状，格式为[h, w]。

        返回:
            tuple[dict,float]: float值为平均avg_factor，dict包含以下组件:
               - center_heatmap_target (Tensor): 中心热图目标，shape (B, num_classes, H, W)。
               - wh_target (Tensor): 宽高预测目标，shape (B, 2, H, W)。
               - offset_target (Tensor): 偏移预测目标，shape (B, 2, H, W)。
               - wh_offset_target_weight (Tensor): 宽高和偏移预测的权重，shape (B, 2, H, W)。
        """
        img_h, img_w = self.train_cfg.lidar_resolution_height, self.train_cfg.lidar_resolution_width
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        yaw_class_target = gt_bboxes[-1].new_zeros([bs, 1, feat_h, feat_w]).long()
        yaw_res_target = gt_bboxes[-1].new_zeros([bs, 1, feat_h, feat_w])
        velocity_target = gt_bboxes[-1].new_zeros([bs, 1, feat_h, feat_w])
        brake_target = gt_bboxes[-1].new_zeros([bs, 1, feat_h, feat_w]).long()
 
        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[0][batch_id]
            gt_label = gt_labels[0][batch_id]
            gt_ignore = gt_ignores[0][batch_id]

            center_x = gt_bbox[:, [0]] * width_ratio
            center_y = gt_bbox[:, [1]] * width_ratio
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                if gt_ignore[j]:
                    continue

                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                
                if ctx_int < 0 or ctx_int >= feat_w or cty_int < 0 or cty_int >= feat_h:
                    continue
                
                scale_box_h = gt_bbox[j, 3] * height_ratio
                scale_box_w = gt_bbox[j, 2] * width_ratio
                
                radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.1)
                radius = max(2, int(radius))
                ind = gt_label[j].long()
                
                gen_gaussian_target(center_heatmap_target[batch_id, ind], [ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h
                
                yaw_class, yaw_res = self.angle2class(gt_bbox[j, 4])

                yaw_class_target[batch_id, 0, cty_int, ctx_int] = yaw_class
                yaw_res_target[batch_id, 0, cty_int, ctx_int] = yaw_res

                velocity_target[batch_id, 0, cty_int, ctx_int] = gt_bbox[j, 5]
                brake_target[batch_id, 0, cty_int, ctx_int] = gt_bbox[j, 6].long()
                 
                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int
                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            yaw_class_target=yaw_class_target.squeeze(1),
            yaw_res_target=yaw_res_target,
            offset_target=offset_target,
            velocity_target=velocity_target,
            brake_target=brake_target.squeeze(1),
            wh_offset_target_weight=wh_offset_target_weight)
        return target_result, avg_factor

    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   yaw_class_preds,
                   yaw_res_preds,
                   velocity_preds, 
                   brake_preds,
                   rescale=True,
                   with_nms=False):
        """将网络输出的批次转换为边界框预测。

        参数:
            center_heatmap_preds (list[Tensor]): 所有层级的中心预测热图，shape (B, num_classes, H, W)。
            wh_preds (list[Tensor]): 所有层级的宽高预测，shape (B, 2, H, W)。
            offset_preds (list[Tensor]): 所有层级的偏移预测，shape (B, 2, H, W)。
            img_metas (list[dict]): 每张图像的元信息，如图像大小、缩放因子等。
            rescale (bool): 如果为True，返回原始图像空间中的框，默认为True。
            with_nms (bool): 如果为True，返回框前先进行NMS，默认为False。

        返回:
            list[tuple[Tensor, Tensor]]: result_list中的每个元素是2元组。
                第一个元素是(n, 5)张量，5表示(tl_x, tl_y, br_x, br_y, score)，score在0到1之间。
                元组中第二个张量的shape为(n,)，每个元素表示对应框的类别标签。
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1

        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_preds[0],
            wh_preds[0],
            offset_preds[0],
            yaw_class_preds[0],
            yaw_res_preds[0],
            velocity_preds[0], 
            brake_preds[0],
            k=self.train_cfg.top_k_center_keypoints,
            kernel=self.train_cfg.center_net_max_pooling_kernel)

        if with_nms:
            det_results = []
            for (det_bboxes, det_labels) in zip(batch_det_bboxes,
                                                batch_labels):
                det_bbox, det_label = self._bboxes_nms(det_bboxes, det_labels,
                                                       self.test_cfg)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(bs) for bs in zip(batch_det_bboxes, batch_labels)
            ]
        return det_results

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       yaw_class_pred,
                       yaw_res_pred,
                       velocity_pred,
                       brake_pred,
                       k=100,
                       kernel=3):
        """将输出转换为检测原始边界框预测。

        参数:
            center_heatmap_pred (Tensor): 中心预测热图，shape (B, num_classes, H, W)。
            wh_pred (Tensor): 宽高预测，shape (B, 2, H, W)。
            offset_pred (Tensor): 偏移预测，shape (B, 2, H, W)。
            img_shape (list[int]): 图像形状，格式为[h, w]。
            k (int): 从热图中获取top k中心关键点，默认100。
            kernel (int): 提取局部最大像素的最大池化核，默认3。

        返回:
            tuple[torch.Tensor]: CenterNetHead的解码输出，包含以下张量:
              - batch_bboxes (Tensor): 每个框的坐标，shape (B, k, 5)。
              - batch_topk_labels (Tensor): 每个框的类别，shape (B, k)。
        """
        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        yaw_class = transpose_and_gather_feat(yaw_class_pred, batch_index)
        yaw_res = transpose_and_gather_feat(yaw_res_pred, batch_index)
        velocity = transpose_and_gather_feat(velocity_pred, batch_index)
        brake = transpose_and_gather_feat(brake_pred, batch_index)
        brake = torch.argmax(brake, -1)
        velocity = velocity[..., 0]

        # 将类别+残差转换为偏航角
        yaw_class = torch.argmax(yaw_class, -1)
        yaw = self.class2angle(yaw_class, yaw_res.squeeze(2))
        # 速度
        
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]

        ratio = 4.

        batch_bboxes = torch.stack([topk_xs, topk_ys, wh[..., 0], wh[..., 1], yaw, velocity, brake], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        batch_bboxes[:, :, :4] *= ratio

        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels

        out_bboxes, keep = batched_nms(bboxes[:, :4].contiguous(),
                                       bboxes[:, -1].contiguous(), labels,
                                       cfg.nms_cfg)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class LidarCenterNet(nn.Module):
    """
    完整的自动驾驶感知与规划模型 (LidarCenterNet)
    
    功能:
        集成多模态感知（图像+LiDAR）和路点规划的端到端自动驾驶模型
        支持多种融合backbone，输出路点预测和3D目标检测结果
    
    架构:
        1. 感知backbone: 融合图像和LiDAR特征
        2. BEV预测头: 预测鸟瞰图语义
        3. 目标检测头: 基于CenterNet的3D目标检测
        4. 路点预测: GRU网络预测未来路点
        5. 可选: 语义分割和深度估计解码器
    
    参数:
        config (GlobalConfig): 全局配置对象
        device (torch.device): 运行设备（CPU或GPU）
        backbone (str): 融合backbone类型，可选:
            - 'transFuser': 基于Transformer的融合
            - 'late_fusion': 后期特征融合
            - 'geometric_fusion': 几何感知融合
            - 'latentTF': 潜在空间Transformer融合
            - 'crossvit_fusion': CrossViT跨模态注意力融合
        image_architecture (str): 图像编码器架构，默认'resnet34'
        lidar_architecture (str): LiDAR编码器架构，默认'resnet18'
        use_velocity (bool): 是否使用速度信息，默认True
    
    输出:
        推理模式: (路点预测, 目标检测结果, BEV预测, ...)
        训练模式: (路点预测, 各种损失, ...)
    
    示例:
        >>> config = GlobalConfig(setting='eval')
        >>> model = LidarCenterNet(config, device='cuda', backbone='crossvit_fusion')
        >>> output = model.forward(image, lidar, target_point, ego_vel)
    """

    def __init__(self, config, device, backbone, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        """
        初始化LidarCenterNet模型
        
        参数:
            config (GlobalConfig): 全局配置对象，包含所有超参数
            device (torch.device): 运行设备
            backbone (str): 融合backbone类型
            image_architecture (str): 图像编码器架构，默认'resnet34'
            lidar_architecture (str): LiDAR编码器架构，默认'resnet18'
            use_velocity (bool): 是否使用速度信息，默认True
        """
        super().__init__()
        self.device = device
        self.config = config
        self.pred_len = config.pred_len                          # 预测路点数量
        self.use_target_point_image = config.use_target_point_image  # 是否使用目标点图像
        self.gru_concat_target_point = config.gru_concat_target_point  # GRU是否拼接目标点
        self.use_point_pillars = config.use_point_pillars        # 是否使用Point Pillars

        # 如果使用Point Pillars，初始化Point Pillar网络
        if(self.use_point_pillars == True):
            self.point_pillar_net = PointPillarNet(config.num_input, config.num_features,
                                                   min_x = config.min_x, max_x = config.max_x,
                                                   min_y = config.min_y, max_y = config.max_y,
                                                   pixels_per_meter = int(config.pixels_per_meter),
                                                  )

        self.backbone = backbone  # 保存backbone类型名称

        # 根据backbone类型初始化对应的融合模型
        if(backbone == 'transFuser'):
            # TransFuser: 基于Transformer的图像-LiDAR融合
            self._model = TransfuserBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif(backbone == 'late_fusion'):
            # LateFusion: 后期特征融合（分别编码后融合）
            self._model = LateFusionBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif(backbone == 'geometric_fusion'):
            # GeometricFusion: 几何感知的跨模态融合
            self._model = GeometricFusionBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'latentTF'):
            # LatentTF: 潜在空间Transformer融合
            self._model = latentTFBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'crossvit_fusion'):
            # CrossViTFusion: 基于CrossViT的跨模态注意力融合（新增）
            self._model = CrossViTFusionBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'image_only'):
            # ImageOnly: 仅图像模态（消融实验）
            self._model = ImageOnlyBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'lidar_only'):
            # LidarOnly: 仅LiDAR模态（消融实验）
            self._model = LidarOnlyBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'no_bi_attn'):
            # 消融: 移除双向注意力 (仅LiDAR→图像单向)
            self._model = NoBiAttnCrossViTBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'no_multiscale'):
            # 消融: 移除多尺度融合 (仅Layer4融合)
            self._model = NoMultiScaleCrossViTBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'no_velocity'):
            # 消融: 移除速度嵌入
            self._model = NoVelCrossViTBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'no_bi_ms'):
            # 消融: 同时移除双向注意力和多尺度融合
            self._model = NoBiMsCrossViTBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'std_crossvit'):
            # 消融: 标准交叉注意力 (用nn.MultiheadAttention)
            self._model = StdCrossViTBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'no_attn'):
            # 消融: 移除跨注意力 (4层纯MLP，无跨模态交互)
            self._model = NoAttnCrossViTBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'no_attn_no_ms'):
            # 消融: 无注意力+无多尺度 (仅Layer4 MLP)
            self._model = NoAttnNoMsCrossViTBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        else:
            raise ValueError("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF, crossvit_fusion, image_only, lidar_only, no_bi_attn, no_multiscale, no_velocity, no_bi_ms, std_crossvit, no_attn, no_attn_no_ms")

        # 多任务学习：语义分割和深度估计解码器
        if config.multitask:
            self.seg_decoder   = SegDecoder(self.config,   self.config.perception_output_features).to(self.device)
            self.depth_decoder = DepthDecoder(self.config, self.config.perception_output_features).to(self.device)

        channel = config.channel  # 基础通道数（默认64）

        # BEV预测头：预测鸟瞰图语义（3类：背景、道路、车辆）
        self.pred_bev = nn.Sequential(
                            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(channel, 3, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        ).to(self.device)

        # 目标检测头：基于CenterNet的3D目标检测
        self.head = LidarCenterNetHead(channel, channel, 1, train_cfg=config).to(self.device)
        self.i = 0  # 调试计数器

        # 路点预测网络：将感知特征映射到路点预测
        self.join = nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        ).to(self.device)

        self.decoder = nn.GRUCell(input_size=4 if self.gru_concat_target_point else 2, # 2 represents x,y coordinate
                                  hidden_size=self.config.gru_hidden_size).to(self.device)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(self.config.gru_hidden_size, 3).to(self.device)

        # PID控制器
        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

    def forward_gru(self, z, target_point):
        z = self.join(z)
    
        output_wp = list()
        
        # GRU的初始输入变量
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(z.device)

        target_point = target_point.clone()
        target_point[:, 1] *= -1
        
        # 自回归生成输出路点
        for _ in range(self.pred_len):
            if self.gru_concat_target_point:
                x_in = torch.cat([x, target_point], dim=1)
            else:
                x_in = x
            
            z = self.decoder(x_in, z)
            dx = self.output(z)
            
            x = dx[:,:2] + x
            
            output_wp.append(x[:,:2])
            
        pred_wp = torch.stack(output_wp, dim=1)

        # 在车辆坐标系中预测路点，然后转换到LiDAR坐标系，因为GT路点在LiDAR坐标系中
        pred_wp[:, :, 0] = pred_wp[:, :, 0] - self.config.lidar_pos[0]
            
        pred_brake = None
        steer = None
        throttle = None
        brake = None

        return pred_wp, pred_brake, steer, throttle, brake

    def control_pid(self, waypoints, velocity, is_stuck):
        """
        使用PID控制器预测车辆控制信号
        
        功能:
            根据预测的路点和当前速度，计算转向、油门和刹车控制信号
            使用两个独立的PID控制器分别控制转向和速度
        
        参数:
            waypoints (Tensor): 预测的路点序列，shape (1, pred_len, 2)
                - 坐标系: 车辆坐标系（x向前，y向左）
            velocity (Tensor): 当前车速（米/秒）
            is_stuck (bool): 是否处于卡住状态
                - True: 使用默认速度强制前进
        
        返回:
            steer (float): 转向控制信号，范围[-1, 1]
                - 负值: 向左转
                - 正值: 向右转
            throttle (float): 油门控制信号，范围[0, 1]
            brake (bool): 是否刹车
        
        算法:
            1. 计算期望速度（相邻路点距离 * 2）
            2. 如果卡住，使用默认速度
            3. 判断是否需要刹车（速度过低或超速）
            4. 速度PID控制器计算油门
            5. 转向PID控制器计算转向角
        """
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()
        # 训练时将路点转换到LiDAR坐标系，控制时需要转换回来
        waypoints[:, 0] += self.config.lidar_pos[0]

        speed = velocity[0].data.cpu().numpy()

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0

        if is_stuck:
            desired_speed = np.array(self.config.default_speed) # 默认速度14.4 km/h

        brake = ((desired_speed < self.config.brake_speed) or ((speed / desired_speed) > self.config.brake_ratio))

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
        throttle = throttle if not brake else 0.0
        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if (speed < 0.01):
            angle = 0.0  # 不移动时不希望角度误差在积分中累积
        if brake:
            angle = 0.0
        
        steer = self.turn_controller.step(angle)

        steer = np.clip(steer, -1.0, 1.0) # 有效转向值在[-1,1]范围内

        return steer, throttle, brake
    
    def forward_ego(self, rgb, lidar_bev, target_point, target_point_image, ego_vel, bev_points=None, cam_points=None, save_path=None, expert_waypoints=None,
                    stuck_detector=0, forced_move=False, num_points=None, rgb_back=None, debug=False):
        """
        自车推理前向传播（评估/部署模式）
        
        功能:
            在评估模式下执行完整的感知和规划推理
            输出路点预测和目标检测结果
        
        参数:
            rgb (Tensor): RGB图像输入，shape (B, C, H, W)
            lidar_bev (Tensor): LiDAR BEV特征，shape (B, C, H, W)
            target_point (Tensor): 目标点坐标，shape (B, 2)
            target_point_image (Tensor): 目标点图像，shape (B, 1, H, W)
            ego_vel (Tensor): 自车速度，shape (B, 1)
            bev_points (Tensor): BEV点云坐标（geometric_fusion使用）
            cam_points (Tensor): 相机点云坐标（geometric_fusion使用）
            save_path (str): 调试图像保存路径
            expert_waypoints (Tensor): 专家路点（用于可视化对比）
            stuck_detector (int): 卡住检测计数器
            forced_move (bool): 是否强制移动
            num_points (Tensor): 每帧点云点数（Point Pillars使用）
            rgb_back (Tensor): 后视相机图像
            debug (bool): 是否开启调试模式
        
        返回:
            pred_wp (Tensor): 预测路点，shape (B, pred_len, 2)
            bboxes (list): 检测到的3D边界框列表
            pred_bev (Tensor): BEV语义预测
            steer (float): 转向控制信号
            throttle (float): 油门控制信号
            brake (bool): 刹车信号
        """
        # 如果使用Point Pillars，先将点云转换为BEV特征
        if(self.use_point_pillars == True):
            lidar_bev = self.point_pillar_net(lidar_bev, num_points)
            lidar_bev = torch.rot90(lidar_bev, -1, dims=(2, 3))  # 旋转以保持与体素化方法的一致性

        # 如果使用目标点图像，将其拼接到LiDAR BEV特征
        if self.use_target_point_image:
            lidar_bev = torch.cat((lidar_bev, target_point_image), dim=1)

        # 根据backbone类型调用对应的前向传播
        if (self.backbone == 'transFuser'):
            # TransFuser: 标准图像-LiDAR融合
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'late_fusion'):
            # LateFusion: 后期特征融合
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'geometric_fusion'):
            # GeometricFusion: 需要额外的几何点云信息
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel, bev_points, cam_points)
        elif (self.backbone == 'latentTF'):
            # LatentTF: 潜在空间Transformer融合
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'crossvit_fusion'):
            # CrossViTFusion: 跨模态注意力融合（新增）
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'image_only'):
            # ImageOnly: 仅图像模态（消融实验）
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'lidar_only'):
            # LidarOnly: 仅LiDAR模态（消融实验）
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'no_bi_attn'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'no_multiscale'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'no_velocity'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'no_bi_ms'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'std_crossvit'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'no_attn'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'no_attn_no_ms'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        else:
            raise ValueError("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF, crossvit_fusion, image_only, lidar_only, no_bi_attn, no_multiscale, no_velocity, no_bi_ms, std_crossvit, no_attn, no_attn_no_ms")

        pred_wp, _, _, _, _ = self.forward_gru(fused_features, target_point)

        preds = self.head([features[0]])
        results = self.head.get_bboxes(preds[0], preds[1], preds[2], preds[3], preds[4], preds[5], preds[6])
        bboxes, _ = results[0]

        # 根据预测置信度过滤边界框
        bboxes = bboxes[bboxes[:, -1] > self.config.bb_confidence_threshold]
        rotated_bboxes = []
        for bbox in bboxes.detach().cpu().numpy():
            bbox = self.get_bbox_local_metric(bbox)
            rotated_bboxes.append(bbox)

        self.i += 1
        if debug and self.i % 2 == 0 and not (save_path is None):
            pred_bev = self.pred_bev(features[0])
            pred_bev = F.interpolate(pred_bev, (self.config.bev_resolution_height, self.config.bev_resolution_width), mode='bilinear', align_corners=True)
            pred_semantic = self.seg_decoder(image_features_grid)
            pred_depth = self.depth_decoder(image_features_grid)

            self.visualize_model_io(save_path, self.i, self.config, rgb, lidar_bev, target_point,
                            pred_wp, pred_bev, pred_semantic, pred_depth, bboxes, self.device,
                            gt_bboxes=None, expert_waypoints=expert_waypoints, stuck_detector=stuck_detector, forced_move=forced_move)


        return pred_wp, rotated_bboxes

    def forward(self, rgb, lidar_bev, ego_waypoint, target_point, target_point_image, ego_vel, bev, label, depth, semantic, num_points=None, save_path=None, bev_points=None, cam_points=None):
        """
        训练前向传播（计算所有损失）
        
        功能:
            在训练模式下执行完整的前向传播，计算所有损失函数
            包括路点损失、BEV损失、目标检测损失、语义分割损失和深度损失
        
        参数:
            rgb (Tensor): RGB图像输入，shape (B, C, H, W)
            lidar_bev (Tensor): LiDAR BEV特征，shape (B, C, H, W)
            ego_waypoint (Tensor): 真实路点（GT），shape (B, pred_len, 2)
            target_point (Tensor): 目标点坐标，shape (B, 2)
            target_point_image (Tensor): 目标点图像，shape (B, 1, H, W)
            ego_vel (Tensor): 自车速度，shape (B, 1)
            bev (Tensor): BEV语义标签，shape (B, H, W)
            label (Tensor): 目标检测标签，shape (B, N, 8)
            depth (Tensor): 深度图标签，shape (B, 1, H, W)
            semantic (Tensor): 语义分割标签，shape (B, H, W)
            num_points (Tensor): 每帧点云点数（Point Pillars使用）
            save_path (str): 调试图像保存路径
            bev_points (Tensor): BEV点云坐标（geometric_fusion使用）
            cam_points (Tensor): 相机点云坐标（geometric_fusion使用）
        
        返回:
            loss (dict): 包含所有损失的字典:
                - loss_wp: 路点预测损失（L1损失）
                - loss_bev: BEV语义分割损失（交叉熵）
                - loss_center_heatmap: 目标检测中心热图损失
                - loss_wh: 目标检测宽高损失
                - loss_offset: 目标检测偏移损失
                - loss_yaw_class: 方向角分类损失
                - loss_yaw_res: 方向角残差损失
                - loss_velocity: 速度预测损失
                - loss_brake: 刹车预测损失
                - loss_depth: 深度估计损失（多任务时）
                - loss_semantic: 语义分割损失（多任务时）
        """
        loss = {}  # 初始化损失字典

        # 如果使用Point Pillars，先将点云转换为BEV特征
        if(self.use_point_pillars == True):
            lidar_bev = self.point_pillar_net(lidar_bev, num_points)
            lidar_bev = torch.rot90(lidar_bev, -1, dims=(2, 3))  # 旋转以保持与体素化方法的一致性

        # 如果使用目标点图像，将其拼接到LiDAR BEV特征
        if self.use_target_point_image:
            lidar_bev = torch.cat((lidar_bev, target_point_image), dim=1)

        # 根据backbone类型调用对应的前向传播
        if (self.backbone == 'transFuser'):
            # TransFuser: 标准图像-LiDAR融合
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'late_fusion'):
            # LateFusion: 后期特征融合
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'geometric_fusion'):
            # GeometricFusion: 需要额外的几何点云信息
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel, bev_points, cam_points)
        elif (self.backbone == 'latentTF'):
            # LatentTF: 潜在空间Transformer融合
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'crossvit_fusion'):
            # CrossViTFusion: 跨模态注意力融合
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'image_only'):
            # ImageOnly: 仅图像模态（消融实验）
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'lidar_only'):
            # LidarOnly: 仅LiDAR模态（消融实验）
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'no_bi_attn'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'no_multiscale'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'no_velocity'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'no_bi_ms'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'std_crossvit'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'no_attn'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'no_attn_no_ms'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        else:
            raise ValueError("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF, crossvit_fusion, image_only, lidar_only, no_bi_attn, no_multiscale, no_velocity, no_bi_ms, std_crossvit, no_attn, no_attn_no_ms")

        # GRU路点预测
        pred_wp, _, _, _, _ = self.forward_gru(fused_features, target_point)

        # 预测BEV鸟瞰图语义（3类：背景、道路、车辆）
        pred_bev = self.pred_bev(features[0])
        pred_bev = F.interpolate(pred_bev, (self.config.bev_resolution_height, self.config.bev_resolution_width), mode='bilinear', align_corners=True)

        # 计算BEV损失（加权交叉熵，车辆类别权重更高）
        # image_only模式下关闭BEV损失：图像特征(前视)与BEV标签(鸟瞰)视角不匹配，无法有效学习
        if self.backbone == 'image_only':
            loss_bev = torch.tensor(0.0, device=pred_bev.device, requires_grad=True)
        else:
            weight = torch.from_numpy(np.array([1., 1., 3.])).to(dtype=torch.float32, device=pred_bev.device)
            loss_bev = F.cross_entropy(pred_bev, bev, weight=weight).mean()

        # 计算路点损失（L1损失）
        loss_wp = torch.mean(torch.abs(pred_wp - ego_waypoint))
        loss.update({
            "loss_wp": loss_wp,
            "loss_bev": loss_bev
        })

        # 目标检测前向传播
        preds = self.head([features[0]])

        # 准备目标检测标签
        gt_labels = torch.zeros_like(label[:, :, 0])  # 所有目标都是同一类别（车辆）
        gt_bboxes_ignore = label.sum(dim=-1) == 0.    # 全零标签表示无效目标
        
        # 计算目标检测损失
        loss_bbox = self.head.loss(preds[0], preds[1], preds[2], preds[3], preds[4], preds[5], preds[6],
                                [label], gt_labels=[gt_labels], gt_bboxes_ignore=[gt_bboxes_ignore], img_metas=None)
        
        loss.update(loss_bbox)

        # 多任务学习：语义分割和深度估计
        if self.config.multitask:
            pred_semantic = self.seg_decoder(image_features_grid)
            pred_depth = self.depth_decoder(image_features_grid)
            # lidar_only模式下关闭语义分割和深度估计损失：LiDAR特征(鸟瞰)与图像视角标签不匹配
            if self.backbone == 'lidar_only':
                loss_semantic = torch.tensor(0.0, device=pred_semantic.device, requires_grad=True)
                loss_depth = torch.tensor(0.0, device=pred_depth.device, requires_grad=True)
            else:
                loss_semantic = self.config.ls_seg * F.cross_entropy(pred_semantic, semantic).mean()
                loss_depth = self.config.ls_depth * F.l1_loss(pred_depth, depth).mean()
            loss.update({
                "loss_depth": loss_depth,
                "loss_semantic": loss_semantic
            })
        else:
            # 不使用多任务时，损失设为0
            loss.update({
                "loss_depth": torch.zeros_like(loss_wp),
                "loss_semantic": torch.zeros_like(loss_wp)
            })

        self.i += 1
        if ((self.config.debug == True) and (self.i % self.config.train_debug_save_freq == 0) and (save_path != None)):
            with torch.no_grad():
                results = self.head.get_bboxes(preds[0], preds[1], preds[2], preds[3], preds[4], preds[5], preds[6])
                bboxes, _ = results[0]
                bboxes = bboxes[bboxes[:, -1] > self.config.bb_confidence_threshold]
                self.visualize_model_io(save_path, self.i, self.config, rgb, lidar_bev, target_point,
                                   pred_wp, pred_bev, pred_semantic, pred_depth, bboxes, self.device,
                                   gt_bboxes=label, expert_waypoints=ego_waypoint, stuck_detector=0, forced_move=False)

        return loss


    # 将坐标系转换为x向前y向右，车辆中心为原点
    # 单位从像素转换为米
    def get_bbox_local_metric(self, bbox):
        x, y, w, h, yaw, speed, brake, confidence = bbox

        w = w / self.config.bounding_box_divisor / self.config.pixels_per_meter # 采集数据时乘以2，加载标签时乘以8
        h = h / self.config.bounding_box_divisor / self.config.pixels_per_meter # 采集数据时乘以2，加载标签时乘以8

        T = get_lidar_to_bevimage_transform()
        T_inv = np.linalg.inv(T)

        center = np.array([x,y,1.0])

        center_old_coordinate_sys = T_inv @ center

        center_old_coordinate_sys = center_old_coordinate_sys + np.array(self.config.lidar_pos)

        # 转换到标准CARLA右手坐标系
        center_old_coordinate_sys[1] =  -center_old_coordinate_sys[1]

        bbox = np.array([[-h, -w, 1],
                         [-h,  w, 1],
                         [ h,  w, 1],
                         [ h, -w, 1],
                         [ 0,  0, 1],
                         [ 0, h * speed * 0.5, 1]])

        R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw),  np.cos(yaw), 0],
                      [0,                      0, 1]])

        for point_index in range(bbox.shape[0]):
            bbox[point_index] = R @ bbox[point_index]
            bbox[point_index] = bbox[point_index] + np.array([center_old_coordinate_sys[0], center_old_coordinate_sys[1],0])

        return bbox, brake, confidence

    # 这个方法有所不同
    def get_rotated_bbox(self, bbox):
        x, y, w, h, yaw, speed, brake =  bbox

        bbox = np.array([[h,   w, 1],
                         [h,  -w, 1],
                         [-h, -w, 1],
                         [-h,  w, 1],
                         [0, 0, 1],
                         [-h * speed * 0.5, 0, 1]])
        bbox[:, :2] /= self.config.bounding_box_divisor
        bbox[:, :2] = bbox[:, [1, 0]]

        c, s = np.cos(yaw), np.sin(yaw)
        # 使用y x因为坐标系已改变
        r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

        bbox = r1_to_world @ bbox.T
        bbox = bbox.T

        return bbox, brake

    def draw_bboxes(self, bboxes, image, color=(255, 255, 255), brake_color=(0, 0, 255)):
        idx = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]]
        for bbox, brake in bboxes:
            bbox = bbox.astype(np.int32)[:, :2]
            for s, e in idx:
                if brake >= self.config.draw_brake_threshhold:
                    color = brake_color
                else:
                    color = color
                # 刹车时仍有较高速度
                cv2.line(image, tuple(bbox[s]), tuple(bbox[e]), color=color, thickness=1)
        return image


    def draw_waypoints(self, label, waypoints, image, color = (255, 255, 255)):
        waypoints = waypoints.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        for bbox, points in zip(label, waypoints):
            x, y, w, h, yaw, speed, brake =  bbox
            c, s = np.cos(yaw), np.sin(yaw)
            # 使用y x因为坐标系已改变
            r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

            # 转换到图像空间
            # 需要像LiDAR点一样对y分量取负
            # 直接在图像坐标系中构建点
            # LiDAR坐标系: 前方+x, 右方+y
            #            x
            #            +
            #            |
            #            |
            #            |---------+y
            #
            # 图像坐标系: ---------> x
            #            |
            #            |
            #            +
            #            y

            points[:, 0] *= -1
            points = points * self.config.pixels_per_meter
            points = points[:, [1, 0]]
            points = np.concatenate((points, np.ones_like(points[:, :1])), axis=-1)

            points = r1_to_world @ points.T
            points = points.T

            points_to_draw = []
            for point in points[:, :2]:
                points_to_draw.append(point.copy())
                point = point.astype(np.int32)
                cv2.circle(image, tuple(point), radius=3, color=color, thickness=3)
        return image


    def draw_target_point(self, target_point, image, color = (255, 255, 255)):
        target_point = target_point.copy()

        target_point[1] += self.config.lidar_pos[0]
        point = target_point * self.config.pixels_per_meter
        point[1] *= -1
        point[1] = self.config.lidar_resolution_width - point[1] #Might be LiDAR height
        point[0] += int(self.config.lidar_resolution_height / 2.0) #Might be LiDAR width
        point = point.astype(np.int32)
        point = np.clip(point, 0, 512)
        cv2.circle(image, tuple(point), radius=5, color=color, thickness=3)
        return image

    def visualize_model_io(self, save_path, step, config, rgb, lidar_bev, target_point,
                        pred_wp, pred_bev, pred_semantic, pred_depth, bboxes, device,
                        gt_bboxes=None, expert_waypoints=None, stuck_detector=0, forced_move=False):
        font = ImageFont.load_default()
        i = 0 # 如果有批次图像，只可视化第一张
        if config.multitask:
            classes_list = config.classes_list
            converter = np.array(classes_list)

            depth_image = pred_depth[i].detach().cpu().numpy()

            indices = np.argmax(pred_semantic.detach().cpu().numpy(), axis=1)
            semantic_image = converter[indices[i, ...], ...].astype('uint8')

            ds_image = np.stack((depth_image, depth_image, depth_image), axis=2)
            ds_image = (ds_image * 255).astype(np.uint8)
            ds_image = np.concatenate((ds_image, semantic_image), axis=0)
            ds_image = cv2.resize(ds_image, (640, 256))
            ds_image = np.concatenate([ds_image, np.zeros_like(ds_image[:50])], axis=0)

        images = np.concatenate(list(lidar_bev.detach().cpu().numpy()[i][:2]), axis=1)
        images = (images * 255).astype(np.uint8)
        images = np.stack([images, images, images], axis=-1)
        images = np.concatenate([images, np.zeros_like(images[:50])], axis=0)

        # 绘制GT边界框
        if (not (gt_bboxes is None)):
            rotated_bboxes_gt = []
            for bbox in gt_bboxes.detach().cpu().numpy()[i]:
                bbox = self.get_rotated_bbox(bbox)
                rotated_bboxes_gt.append(bbox)
            images = self.draw_bboxes(rotated_bboxes_gt, images, color=(0, 255, 0), brake_color=(0, 255, 128))

        rotated_bboxes = []
        for bbox in bboxes.detach().cpu().numpy():
            bbox = self.get_rotated_bbox(bbox[:7])
            rotated_bboxes.append(bbox)
        images = self.draw_bboxes(rotated_bboxes, images, color=(255, 0, 0), brake_color=(0, 255, 255))

        label = torch.zeros((1, 1, 7)).to(device)
        label[:, -1, 0] = 128.
        label[:, -1, 1] = 256.

        if not expert_waypoints is None:
            images = self.draw_waypoints(label[0], expert_waypoints[i:i+1], images, color=(0, 0, 255))

        images = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, 2:]), images, color=(255, 255, 255)) # 辅助路点（白色）
        images = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, :2]), images, color=(255, 0, 0))     # 前两个相关路点（蓝色）

        # 绘制目标点
        images = self.draw_target_point(target_point[i].detach().cpu().numpy(), images)

        # 卡住状态文字
        images = Image.fromarray(images)
        draw = ImageDraw.Draw(images)
        draw.text((10, 0), "stuck detector:   %04d" % (stuck_detector), font=font)
        draw.text((10, 30), "forced move:      %s" % (" True" if forced_move else "False"), font=font,
                  fill=(255, 0, 0, 255) if forced_move else (255, 255, 255, 255))
        images = np.array(images)

        bev = pred_bev[i].detach().cpu().numpy().argmax(axis=0) / 2.
        bev = np.stack([bev, bev, bev], axis=2) * 255.
        bev_image = bev.astype(np.uint8)
        bev_image = cv2.resize(bev_image, (256, 256))
        bev_image = np.concatenate([bev_image, np.zeros_like(bev_image[:50])], axis=0)

        if not expert_waypoints is None:
            bev_image = self.draw_waypoints(label[0], expert_waypoints[i:i+1], bev_image, color=(0, 0, 255))

        bev_image = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, 2:]), bev_image, color=(255, 255, 255))
        bev_image = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, :2]), bev_image, color=(255, 0, 0))

        bev_image = self.draw_target_point(target_point[i].detach().cpu().numpy(), bev_image)

        if (not (expert_waypoints is None)):
            aim = expert_waypoints[i:i + 1, :2].detach().cpu().numpy()[0].mean(axis=0)
            expert_angle = np.degrees(np.arctan2(aim[1], aim[0] + self.config.lidar_pos[0]))

            aim = pred_wp[i:i + 1, :2].detach().cpu().numpy()[0].mean(axis=0)
            ego_angle = np.degrees(np.arctan2(aim[1], aim[0] + self.config.lidar_pos[0]))
            angle_error = normalize_angle_degree(expert_angle - ego_angle)

            bev_image = Image.fromarray(bev_image)
            draw = ImageDraw.Draw(bev_image)
            draw.text((0, 0), "Angle error:        %.2f°" % (angle_error), font=font)

        bev_image = np.array(bev_image)

        rgb_image = rgb[i].permute(1, 2, 0).detach().cpu().numpy()[:, :, [2, 1, 0]]
        rgb_image = cv2.resize(rgb_image, (1280 + 128, 320 + 32))
        assert (config.multitask)
        images = np.concatenate((bev_image, images, ds_image), axis=1)

        images = np.concatenate((rgb_image, images), axis=0)

        cv2.imwrite(str(save_path + ("/%d.png" % (step // 2))), images)