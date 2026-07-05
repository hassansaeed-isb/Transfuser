"""
致谢: Tianwei Yin
从LAV仓库复制

"""

from torch import nn
import torch


def _scatter_max(src, index, dim=0):
    num_groups = index.max().item() + 1
    out_shape = list(src.shape)
    out_shape[dim] = num_groups
    output = torch.full(out_shape, float('-inf'), device=src.device, dtype=src.dtype)
    index_expand = index.unsqueeze(1).expand_as(src) if dim == 0 else index
    output.scatter_reduce_(dim, index_expand, src, reduce='amax', include_self=False)
    return output


def _scatter_mean(src, index, dim=0):
    num_groups = index.max().item() + 1
    out_shape = list(src.shape)
    out_shape[dim] = num_groups
    sum_output = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
    count_output = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
    index_expand = index.unsqueeze(1).expand_as(src) if dim == 0 else index
    sum_output.scatter_add_(dim, index_expand, src)
    ones = torch.ones_like(src)
    count_output.scatter_add_(dim, index_expand, ones)
    return sum_output / count_output.clamp(min=1)


class DynamicPointNet(nn.Module):
    def __init__(self, num_input=9, num_features=[32,32]):
        super().__init__()

        L = []
        for num_feature in num_features:
            L += [
                nn.Linear(num_input, num_feature),
                nn.BatchNorm1d(num_feature),
                nn.ReLU(inplace=True),
            ]

            num_input = num_feature

        self.net = nn.Sequential(*L)
        
    def forward(self, points, inverse_indices):
        """
        TODO: 支持多层
        """
        feat = self.net(points)
        feat_max = _scatter_max(feat, inverse_indices, dim=0)
        # feat_max = scatter_max(points, inverse_indices, dim=0)[0]  # 备用方案
        return feat_max


class PointPillarNet(nn.Module):
    def __init__(self, num_input=9, num_features=[32,32], 
        min_x=-10, max_x=70,
        min_y=-40, max_y=40,
        pixels_per_meter=4):

        super().__init__()
        self.point_net = DynamicPointNet(num_input, num_features)

        self.nx = (max_x-min_x) * pixels_per_meter
        self.ny = (max_y-min_y) * pixels_per_meter
        self.min_x = min_x 
        self.min_y = min_y 
        self.max_x = max_x 
        self.max_y = max_y 
        self.pixels_per_meter = pixels_per_meter

    def decorate(self, points, unique_coords, inverse_indices):
        dtype = points.dtype
        x_centers = unique_coords[inverse_indices][:, 2:3].to(dtype) / self.pixels_per_meter + self.min_x 
        y_centers = unique_coords[inverse_indices][:, 1:2].to(dtype) / self.pixels_per_meter + self.min_y 

        xyz = points[:, :3]

        points_cluster = xyz - _scatter_mean(xyz, inverse_indices, dim=0)[inverse_indices]

        points_xp = xyz[:, :1] - x_centers 
        points_yp = xyz[:, 1:2] - y_centers

        features = torch.cat([points, points_cluster, points_xp, points_yp], dim=-1)
        return features 

    def grid_locations(self, points):
        keep = (points[:, 0] >= self.min_x) & (points[:, 0] < self.max_x) & \
            (points[:, 1] >= self.min_y) & (points[:, 1] < self.max_y)
        points = points[keep, :]

        # 可视化（调试用）
        #import open3d as o3d
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(points[...,:3].detach().cpu().numpy())
        #o3d.visualization.draw_geometries([pcd])


        coords = (points[:, [0, 1]] - torch.tensor([self.min_x, self.min_y], 
            device=points.device)) * self.pixels_per_meter
        coords = coords.long()

        return points, coords 

    def pillar_generation(self, points, coords):
        unique_coords, inverse_indices = coords.unique(return_inverse=True, dim=0)
        decorated_points = self.decorate(points, unique_coords, inverse_indices)

        return decorated_points, unique_coords, inverse_indices

    def scatter_points(self, features, coords, batch_size ):
        canvas = torch.zeros(batch_size, features.shape[1], self.ny, self.nx, dtype=features.dtype, device=features.device)
        canvas[coords[:, 0], :, torch.clamp(self.ny-1-coords[:, 1],0,self.ny-1), torch.clamp(coords[:, 2],0,self.nx-1)] = features
        return canvas 

    def forward(self, lidar_list, num_points):
        batch_size = len(lidar_list)
        with torch.no_grad():
            coords = [] 
            filtered_points = [] 
            for batch_id, points in enumerate(lidar_list):
                points = points[:num_points[batch_id]]
                points, grid_yx= self.grid_locations(points)

                # 批次索引
                grid_byx = torch.nn.functional.pad(grid_yx, 
                    (1, 0), mode='constant', value=batch_id)

                coords.append(grid_byx)
                filtered_points.append(points)

            # batch_size, grid_y, grid_x（批次大小，网格y，网格x）
            coords = torch.cat(coords, dim=0)
            filtered_points = torch.cat(filtered_points, dim=0)

            decorated_points, unique_coords, inverse_indices = self.pillar_generation(filtered_points, coords)

        features = self.point_net(decorated_points, inverse_indices)

        return self.scatter_points(features, unique_coords, batch_size)
