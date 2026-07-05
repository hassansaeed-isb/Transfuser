"""
评估专用模型模块 (model_eval.py)

本文件是 model.py 的评估专用副本，移除了所有 mmcv/mmdet 依赖。
仅保留推理/评估所需的功能，移除了训练专用的损失计算代码。

与 model.py 的主要差异:
    1. 移除 mmcv/mmdet/mmengine 所有 import
    2. 用纯 PyTorch 实现 get_local_maximum、get_topk_from_heatmap、transpose_and_gather_feat
    3. 用 torchvision.ops.nms 替换 mmcv.ops.batched_nms
    4. LidarCenterNetHead 不再继承 BaseDenseHead/BBoxTestMixin，不使用 @HEADS.register_module()
    5. 移除训练专用方法 (loss, get_targets, loss_by_feat)
    6. 移除 build_loss，评估时不需要损失函数
    7. init_weights 使用简化实现，不依赖 mmcv 的 bias_init_with_prob/normal_init

依赖:
    - PyTorch: 深度学习框架
    - torchvision: 仅使用 nms 操作
    - PIL: 图像处理
    - OpenCV: 计算机视觉
"""

from collections import deque
import torch.nn.functional as F
import cv2

from utils import *
from transfuser import TransfuserBackbone, SegDecoder, DepthDecoder
from geometric_fusion import GeometricFusionBackbone
from late_fusion import LateFusionBackbone
from latentTF import latentTFBackbone
from crossvit_fusion import CrossViTFusionBackbone
from single_modal_backbone import ImageOnlyBackbone, LidarOnlyBackbone
from no_bi_attn import NoBiAttnCrossViTBackbone
from no_multiscale import NoMultiScaleCrossViTBackbone
from no_vel import NoVelCrossViTBackbone
from no_bi_ms import NoBiMsCrossViTBackbone
from std_crossvit import StdCrossViTBackbone
from copy import deepcopy
from point_pillar import PointPillarNet

from PIL import Image, ImageFont, ImageDraw
from torchvision import models

import torch
import torch.nn as nn
import functools
import math


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


def multi_apply(func, *args, **kwargs):
    pfunc = functools.partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def _get_local_maximum(heat, kernel=3):
    """提取热图的局部最大值（纯 PyTorch 实现，替代 mmdet 的 get_local_maximum）。

    对热图进行 max_pool2d 后与原始热图比较，仅保留局部最大值位置。

    参数:
        heat (Tensor): 输入热图，shape (B, C, H, W)
        kernel (int): 最大池化核大小，默认3

    返回:
        Tensor: 局部最大值热图，非局部最大值位置被置零
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel_size=kernel, stride=1, padding=pad)
    eq_mask = (heat == hmax).float()
    return heat * eq_mask


def _get_topk_from_heatmap(scores, k=100):
    """从热图中获取 top-k 个关键点（纯 PyTorch 实现，替代 mmdet 的 get_topk_from_heatmap）。

    将热图展平后取 top-k 分数及其索引，再转换为空间坐标。

    参数:
        scores (Tensor): 输入热图，shape (B, C, H, W)
        k (int): 取前 k 个关键点，默认100

    返回:
        tuple: (batch_scores, batch_index, batch_topk_labels, topk_ys, topk_xs)
            - batch_scores: top-k 分数，shape (B, k)
            - batch_index: top-k 在展平特征图中的索引，shape (B, k)
            - batch_topk_labels: top-k 的类别标签，shape (B, k)
            - topk_ys: top-k 的 y 坐标，shape (B, k)
            - topk_xs: top-k 的 x 坐标，shape (B, k)
    """
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), k)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), k)
    topk_labels = (topk_ind / k).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, k)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, k)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, k)

    return topk_score, topk_inds, topk_labels, topk_ys, topk_xs


def _transpose_and_gather_feat(feat, ind):
    """转置并按索引聚合特征（纯 PyTorch 实现，替代 mmdet 的 transpose_and_gather_feat）。

    将特征图从 (B, C, H, W) 转换为 (B, C, H*W)，再按索引收集。

    参数:
        feat (Tensor): 输入特征，shape (B, C, H, W)
        ind (Tensor): 索引张量，shape (B, K)

    返回:
        Tensor: 聚合后的特征，shape (B, K, C)
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _gather_feat(feat, ind, mask=None):
    """按索引收集特征。

    参数:
        feat (Tensor): 输入特征，shape (B, N, C)
        ind (Tensor): 索引张量，shape (B, K)
        mask (Tensor, optional): 可选掩码

    返回:
        Tensor: 收集后的特征，shape (B, K, C)
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _batched_nms(boxes, scores, idxs, iou_threshold):
    """批量非极大值抑制（使用 torchvision.ops.nms 实现，替代 mmcv.ops.batched_nms）。

    对每个类别分别执行 NMS，通过偏移坐标避免跨类别抑制。

    参数:
        boxes (Tensor): 边界框，shape (N, 4)，格式 [x1, y1, x2, y2]
        scores (Tensor): 置信度分数，shape (N,)
        idxs (Tensor): 类别标签，shape (N,)
        iou_threshold (float): IoU 阈值

    返回:
        tuple: (keep_boxes, keep_indices)
            - keep_boxes: NMS 后保留的边界框
            - keep_indices: 保留的索引
    """
    from torchvision.ops import nms

    if boxes.numel() == 0:
        return torch.empty((0, 4), dtype=boxes.dtype, device=boxes.device), \
               torch.empty((0,), dtype=torch.long, device=boxes.device)

    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return boxes[keep], keep


class LidarCenterNetHead(nn.Module):
    """基于CenterNet的LiDAR目标检测头（评估专用，无 mmcv/mmdet 依赖）

    检测输出:
        - center_heatmap: 目标中心点热图 (B, num_classes, H, W)
        - wh: 目标宽高 (B, 2, H, W)
        - offset: 中心点偏移 (B, 2, H, W)
        - yaw_class: 方向角分类 (B, num_dir_bins, H, W)
        - yaw_res: 方向角残差 (B, 1, H, W)
        - velocity: 速度预测 (B, 1, H, W)
        - brake: 刹车预测 (B, 2, H, W)
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 train_cfg=None,
                 test_cfg=None):
        super(LidarCenterNetHead, self).__init__()
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)
        self.num_dir_bins = train_cfg.num_dir_bins
        self.yaw_class_head = self._build_head(in_channel, feat_channel, self.num_dir_bins)
        self.yaw_res_head = self._build_head(in_channel, feat_channel, 1)
        self.velocity_head = self._build_head(in_channel, feat_channel, 1)
        self.brake_head = self._build_head(in_channel, feat_channel, 2)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = train_cfg.fp16_enabled
        self.i = 0

    def _build_head(self, in_channel, feat_channel, out_channel):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """初始化检测头的权重（简化版，不依赖 mmcv）。"""
        bias_init = -math.log((1 - self.train_cfg.center_net_bias_init_with_prob) /
                              self.train_cfg.center_net_bias_init_with_prob)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=self.train_cfg.center_net_normal_init_std)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        yaw_class_pred = self.yaw_class_head(feat)
        yaw_res_pred = self.yaw_res_head(feat)
        velocity_pred = self.velocity_head(feat)
        brake_pred = self.brake_head(feat)

        return center_heatmap_pred, wh_pred, offset_pred, yaw_class_pred, yaw_res_pred, velocity_pred, brake_pred

    def angle2class(self, angle):
        angle = angle % (2 * np.pi)
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        angle_cls = torch.div(shifted_angle, angle_per_class, rounding_mode="trunc")
        angle_res = shifted_angle - (angle_cls * angle_per_class + angle_per_class / 2)
        return angle_cls.long(), angle_res

    def class2angle(self, angle_cls, angle_res, limit_period=True):
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        angle_center = angle_cls.float() * angle_per_class
        angle = angle_center + angle_res
        if limit_period:
            angle[angle > np.pi] -= 2 * np.pi
        return angle

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
            for (det_bboxes, det_labels) in zip(batch_det_bboxes, batch_labels):
                det_bbox, det_label = self._bboxes_nms(det_bboxes, det_labels, self.test_cfg)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(bs) for bs in zip(batch_det_bboxes, batch_labels)
            ]
        return det_results

    @force_fp32(apply_to=('center_heatmap_pred', 'wh_pred', 'offset_pred',
                           'yaw_class_pred', 'yaw_res_pred', 'velocity_pred', 'brake_pred'))
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
        center_heatmap_pred = _get_local_maximum(center_heatmap_pred, kernel=kernel)

        topk_score, batch_index, batch_topk_labels, topk_ys, topk_xs = \
            _get_topk_from_heatmap(center_heatmap_pred, k=k)

        wh = _transpose_and_gather_feat(wh_pred, batch_index)
        offset = _transpose_and_gather_feat(offset_pred, batch_index)
        yaw_class = _transpose_and_gather_feat(yaw_class_pred, batch_index)
        yaw_res = _transpose_and_gather_feat(yaw_res_pred, batch_index)
        velocity = _transpose_and_gather_feat(velocity_pred, batch_index)
        brake = _transpose_and_gather_feat(brake_pred, batch_index)
        brake = torch.argmax(brake, -1)
        velocity = velocity[..., 0]

        yaw_class = torch.argmax(yaw_class, -1)
        yaw = self.class2angle(yaw_class, yaw_res.squeeze(2))

        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]

        ratio = 4.

        batch_bboxes = torch.stack([topk_xs, topk_ys, wh[..., 0], wh[..., 1], yaw, velocity, brake], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, topk_score[..., None]), dim=-1)
        batch_bboxes[:, :, :4] *= ratio

        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels

        out_bboxes, keep = _batched_nms(bboxes[:, :4].contiguous(),
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
    """完整的自动驾驶感知与规划模型（评估专用，无 mmcv/mmdet 依赖）"""

    def __init__(self, config, device, backbone, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__()
        self.device = device
        self.config = config
        self.pred_len = config.pred_len
        self.use_target_point_image = config.use_target_point_image
        self.gru_concat_target_point = config.gru_concat_target_point
        self.use_point_pillars = config.use_point_pillars

        if(self.use_point_pillars == True):
            self.point_pillar_net = PointPillarNet(config.num_input, config.num_features,
                                                   min_x = config.min_x, max_x = config.max_x,
                                                   min_y = config.min_y, max_y = config.max_y,
                                                   pixels_per_meter = int(config.pixels_per_meter),
                                                  )

        self.backbone = backbone

        if(backbone == 'transFuser'):
            self._model = TransfuserBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif(backbone == 'late_fusion'):
            self._model = LateFusionBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif(backbone == 'geometric_fusion'):
            self._model = GeometricFusionBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'latentTF'):
            self._model = latentTFBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'crossvit_fusion'):
            self._model = CrossViTFusionBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'image_only'):
            self._model = ImageOnlyBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'lidar_only'):
            self._model = LidarOnlyBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'no_bi_attn'):
            self._model = NoBiAttnCrossViTBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'no_multiscale'):
            self._model = NoMultiScaleCrossViTBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'no_velocity'):
            self._model = NoVelCrossViTBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'no_bi_ms'):
            self._model = NoBiMsCrossViTBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'std_crossvit'):
            self._model = StdCrossViTBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        else:
            raise ValueError("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF, crossvit_fusion, image_only, lidar_only, no_bi_attn, no_multiscale, no_velocity, no_bi_ms, std_crossvit")

        if config.multitask:
            self.seg_decoder   = SegDecoder(self.config,   self.config.perception_output_features).to(self.device)
            self.depth_decoder = DepthDecoder(self.config, self.config.perception_output_features).to(self.device)

        channel = config.channel

        self.pred_bev = nn.Sequential(
                            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(channel, 3, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        ).to(self.device)

        self.head = LidarCenterNetHead(channel, channel, 1, train_cfg=config).to(self.device)
        self.i = 0

        self.join = nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        ).to(self.device)

        self.decoder = nn.GRUCell(input_size=4 if self.gru_concat_target_point else 2,
                                  hidden_size=self.config.gru_hidden_size).to(self.device)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(self.config.gru_hidden_size, 3).to(self.device)

        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

    def forward_gru(self, z, target_point):
        z = self.join(z)

        output_wp = list()

        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(z.device)

        target_point = target_point.clone()
        target_point[:, 1] *= -1

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

        pred_wp[:, :, 0] = pred_wp[:, :, 0] - self.config.lidar_pos[0]

        pred_brake = None
        steer = None
        throttle = None
        brake = None

        return pred_wp, pred_brake, steer, throttle, brake

    def control_pid(self, waypoints, velocity, is_stuck):
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()
        waypoints[:, 0] += self.config.lidar_pos[0]

        speed = velocity[0].data.cpu().numpy()

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0

        if is_stuck:
            desired_speed = np.array(self.config.default_speed)

        brake = ((desired_speed < self.config.brake_speed) or ((speed / desired_speed) > self.config.brake_ratio))

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
        throttle = throttle if not brake else 0.0
        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if (speed < 0.01):
            angle = 0.0
        if brake:
            angle = 0.0

        steer = self.turn_controller.step(angle)

        steer = np.clip(steer, -1.0, 1.0)

        return steer, throttle, brake

    def forward_ego(self, rgb, lidar_bev, target_point, target_point_image, ego_vel, bev_points=None, cam_points=None, save_path=None, expert_waypoints=None,
                    stuck_detector=0, forced_move=False, num_points=None, rgb_back=None, debug=False):
        if(self.use_point_pillars == True):
            lidar_bev = self.point_pillar_net(lidar_bev, num_points)
            lidar_bev = torch.rot90(lidar_bev, -1, dims=(2, 3))

        if self.use_target_point_image:
            lidar_bev = torch.cat((lidar_bev, target_point_image), dim=1)

        if (self.backbone == 'transFuser'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'late_fusion'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'geometric_fusion'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel, bev_points, cam_points)
        elif (self.backbone == 'latentTF'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'crossvit_fusion'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'image_only'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'lidar_only'):
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
        else:
            raise ValueError("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF, crossvit_fusion, image_only, lidar_only, no_bi_attn, no_multiscale, no_velocity, no_bi_ms, std_crossvit")

        pred_wp, _, _, _, _ = self.forward_gru(fused_features, target_point)

        preds = self.head([features[0]])
        results = self.head.get_bboxes(preds[0], preds[1], preds[2], preds[3], preds[4], preds[5], preds[6])
        bboxes, _ = results[0]

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

    def get_bbox_local_metric(self, bbox):
        x, y, w, h, yaw, speed, brake, confidence = bbox

        w = w / self.config.bounding_box_divisor / self.config.pixels_per_meter
        h = h / self.config.bounding_box_divisor / self.config.pixels_per_meter

        T = get_lidar_to_bevimage_transform()
        T_inv = np.linalg.inv(T)

        center = np.array([x,y,1.0])

        center_old_coordinate_sys = T_inv @ center

        center_old_coordinate_sys = center_old_coordinate_sys + np.array(self.config.lidar_pos)

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
                cv2.line(image, tuple(bbox[s]), tuple(bbox[e]), color=color, thickness=1)
        return image


    def draw_waypoints(self, label, waypoints, image, color = (255, 255, 255)):
        waypoints = waypoints.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        for bbox, points in zip(label, waypoints):
            x, y, w, h, yaw, speed, brake =  bbox
            c, s = np.cos(yaw), np.sin(yaw)
            r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

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
        point[1] = self.config.lidar_resolution_width - point[1]
        point[0] += int(self.config.lidar_resolution_height / 2.0)
        point = point.astype(np.int32)
        point = np.clip(point, 0, 512)
        cv2.circle(image, tuple(point), radius=5, color=color, thickness=3)
        return image

    def visualize_model_io(self, save_path, step, config, rgb, lidar_bev, target_point,
                        pred_wp, pred_bev, pred_semantic, pred_depth, bboxes, device,
                        gt_bboxes=None, expert_waypoints=None, stuck_detector=0, forced_move=False):
        font = ImageFont.load_default()
        i = 0
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

        images = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, 2:]), images, color=(255, 255, 255))
        images = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, :2]), images, color=(255, 0, 0))

        images = self.draw_target_point(target_point[i].detach().cpu().numpy(), images)

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
