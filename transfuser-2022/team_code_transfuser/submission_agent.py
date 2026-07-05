import os
import json
from copy import deepcopy

import cv2
import carla
from PIL import Image
from collections import deque

import torch
import numpy as np
import math

from leaderboard.autoagents import autonomous_agent
from model_eval import LidarCenterNet
from config import GlobalConfig
from data import lidar_to_histogram_features, draw_target_point, lidar_bev_cam_correspondences

from shapely.geometry import Polygon

import itertools
import pathlib
SAVE_PATH = os.environ.get('SAVE_PATH')

if not SAVE_PATH:
    SAVE_PATH = None
else:
    pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

def get_entry_point():
    return 'HybridAgent'


class HybridAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, route_index=None):
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.initialized = False

        args_file = open(os.path.join(path_to_conf_file, 'args.txt'), 'r')
        self.args = json.load(args_file)
        args_file.close()

        # 设置机器以避免加载文件
        self.config = GlobalConfig(setting='eval')

        if ('sync_batch_norm' in self.args):
            self.config.sync_batch_norm = bool(self.args['sync_batch_norm'])
        if ('use_point_pillars' in self.args):
            self.config.use_point_pillars = self.args['use_point_pillars']
        if ('n_layer' in self.args):
            self.config.n_layer = self.args['n_layer']
        if ('use_target_point_image' in self.args):
            self.config.use_target_point_image = bool(self.args['use_target_point_image'])
        if ('use_velocity' in self.args):
            use_velocity = bool(self.args['use_velocity'])
        else:
            use_velocity = True

        if ('image_architecture' in self.args):
            image_architecture = self.args['image_architecture']
        else:
            image_architecture = 'resnet34'

        if ('lidar_architecture' in self.args):
            lidar_architecture = self.args['lidar_architecture']
        else:
            lidar_architecture = 'resnet18'

        if ('backbone' in self.args):
            self.backbone = self.args['backbone']  # 选项: 'geometric_fusion', 'transFuser', 'late_fusion', 'latentTF'
        else:
            self.backbone = 'transFuser'  # 选项: 'geometric_fusion', 'transFuser', 'late_fusion', 'latentTF'

        self.gps_buffer = deque(maxlen=self.config.gps_buffer_max_len) # 存储最近x个更新的GPS信号
        self.ego_model = EgoModel(dt=self.config.carla_frame_rate) # 用于GPS去噪的自行车模型

        self.bb_buffer = deque(maxlen=1)
        self.lidar_pos = self.config.lidar_pos  # LiDAR位置的x, y, z坐标
        self.iou_treshold_nms = self.config.iou_treshold_nms # 边界框预测非极大值抑制使用的IoU阈值


        # 加载模型文件
        self.nets = []
        self.model_count = 0 # 统计集成中的模型数量
        for file in os.listdir(path_to_conf_file):
            if file.endswith(".pth"):
                self.model_count += 1
                print(os.path.join(path_to_conf_file, file))
                net = LidarCenterNet(self.config, 'cuda', self.backbone, image_architecture, lidar_architecture, use_velocity)
                if(self.config.sync_batch_norm == True):
                    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net) # 模型使用同步批归一化训练，需要转换否则参数加载不正确
                # [修改] 2026-03-18: PyTorch 2.0+ 中 torch.load 需要显式指定 weights_only 参数
                # weights_only=False 保持向后兼容，允许加载包含非 tensor 对象的 checkpoint
                state_dict = torch.load(os.path.join(path_to_conf_file, file), map_location='cuda:0', weights_only=False)
                state_dict = {k[7:]: v for k, v in state_dict.items()} # 移除分布式训练产生的.module前缀，如果评估非DDP训练的模型请删除此行
                net.load_state_dict(state_dict, strict=False)
                net.cuda()
                net.eval()
                self.nets.append(net)


        self.stuck_detector = 0
        self.forced_move = 0

        self.use_lidar_safe_check = True
        self.aug_degrees = [0] # 测试时数据增强，未使用，只增强0度
        self.steer_damping = self.config.steer_damping
        self.rgb_back = None # 用于调试



    def _init(self):
        self._route_planner = RoutePlanner(self.config.route_planner_min_distance, self.config.route_planner_max_distance)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        sensors = [
                    {
                        'type': 'sensor.camera.rgb',
                        'x': self.config.camera_pos[0], 'y': self.config.camera_pos[1], 'z':self.config.camera_pos[2],
                        'roll': self.config.camera_rot_0[0], 'pitch': self.config.camera_rot_0[1], 'yaw': self.config.camera_rot_0[2],
                        'width': self.config.camera_width, 'height': self.config.camera_height, 'fov': self.config.camera_fov,
                        'id': 'rgb_front'
                        },
                    {
                        'type': 'sensor.camera.rgb',
                        'x': self.config.camera_pos[0], 'y': self.config.camera_pos[1], 'z':self.config.camera_pos[2],
                        'roll': self.config.camera_rot_1[0], 'pitch': self.config.camera_rot_1[1], 'yaw': self.config.camera_rot_1[2],
                        'width': self.config.camera_width, 'height': self.config.camera_height, 'fov': self.config.camera_fov,
                        'id': 'rgb_left'
                        },
                    {
                        'type': 'sensor.camera.rgb',
                        'x': self.config.camera_pos[0], 'y': self.config.camera_pos[1], 'z':self.config.camera_pos[2],
                        'roll': self.config.camera_rot_2[0], 'pitch': self.config.camera_rot_2[1], 'yaw': self.config.camera_rot_2[2],
                        'width': self.config.camera_width, 'height': self.config.camera_height, 'fov': self.config.camera_fov,
                        'id': 'rgb_right'
                        },
                    {
                        'type': 'sensor.other.imu',
                        'x': 0.0, 'y': 0.0, 'z': 0.0,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'sensor_tick': self.config.carla_frame_rate,
                        'id': 'imu'
                        },
                    {
                        'type': 'sensor.other.gnss',
                        'x': 0.0, 'y': 0.0, 'z': 0.0,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'sensor_tick': 0.01,
                        'id': 'gps'
                        },
                    {
                        'type': 'sensor.speedometer',
                        'reading_frequency': self.config.carla_fps,
                        'id': 'speed'
                        }
                    ]
        if(SAVE_PATH != None): # 用于可视化的调试相机
            sensors.append({
                            'type': 'sensor.camera.rgb',
                            'x': -4.5, 'y': 0.0, 'z':2.3,
                            'roll': 0.0, 'pitch': -15.0, 'yaw': 0.0,
                            'width': 960, 'height': 480, 'fov': 100,
                            'id': 'rgb_back'
                            })

        if (self.backbone != 'latentTF'):  # LiDAR方法
            sensors.append({
                            'type': 'sensor.lidar.ray_cast',
                            'x': self.lidar_pos[0], 'y': self.lidar_pos[1], 'z': self.lidar_pos[2],
                            'roll': self.config.lidar_rot[0], 'pitch': self.config.lidar_rot[1], 'yaw': self.config.lidar_rot[2],
                            'id': 'lidar'
                           })

        return sensors

    def tick(self, input_data):
        rgb = []
        for pos in ['left', 'front', 'right']:
            rgb_cam = 'rgb_' + pos
            rgb_pos = cv2.cvtColor(input_data[rgb_cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            rgb_pos = self.scale_crop(Image.fromarray(rgb_pos), self.config.scale, self.config.img_width, self.config.img_width, self.config.img_resolution[0], self.config.img_resolution[0])
            rgb.append(rgb_pos)
        rgb = np.concatenate(rgb, axis=1)

        if(SAVE_PATH != None): # 用于可视化的调试相机
            # 不需要缓冲，始终使用最新的
            self.rgb_back = input_data["rgb_back"][1][:, :, :3]

        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        if (np.isnan(compass) == True): # CARLA 0.9.10偶尔在指南针中发送NaN值
            compass = 0.0

        result = {
                'rgb': rgb,
                'gps': gps,
                'speed': speed,
                'compass': compass,
                }

        if (self.backbone != 'latentTF'):
            lidar = input_data['lidar'][1][:, :3]
            result['lidar'] = lidar

        pos = self._get_position(result)
        result['gps'] = pos

        self.gps_buffer.append(pos)
        denoised_pos = np.average(self.gps_buffer, axis=0)

        waypoint_route = self._route_planner.run_step(denoised_pos)
        next_wp, next_cmd = waypoint_route[1] if len(waypoint_route) > 1 else waypoint_route[0]
        result['next_command'] = next_cmd.value

        theta = compass + np.pi/2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
            ])

        local_command_point = np.array([next_wp[0]-denoised_pos[0], next_wp[1]-denoised_pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result['target_point'] = tuple(local_command_point)

        return result

    @torch.inference_mode() # torch_no_grad的更快版本
    def run_step(self, input_data, timestamp):
        self.step += 1

        if not self.initialized:
            self._init()
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            self.control = control        

        # 每步都需要运行以进行GPS去噪
        tick_data = self.tick(input_data)

        # 重复动作两次以确保LiDAR数据可用性
        if self.step % self.config.action_repeat == 1:
            self.update_gps_buffer(self.control, tick_data['compass'], tick_data['speed'])
            return self.control

        # 准备图像输入
        image = self.prepare_image(tick_data)

        num_points = None
        if(self.backbone == 'latentTF'): # 仅图像方法
            lidar_bev = torch.zeros((1, 2, self.config.lidar_resolution_width, self.config.lidar_resolution_height)).to('cuda', dtype=torch.float32) # 虚拟数据
        else:
            # 准备LiDAR输入
            if (self.config.use_point_pillars == True):
                lidar_cloud = deepcopy(input_data['lidar'][1])
                lidar_cloud[:, 1] *= -1  # invert
                lidar_bev = [torch.tensor(lidar_cloud).to('cuda', dtype=torch.float32)]
                num_points = [torch.tensor(len(lidar_cloud)).to('cuda', dtype=torch.int32)]
            else:
                lidar_bev = self.prepare_lidar(tick_data)

        
        # 准备目标位置输入
        target_point_image, target_point = self.prepare_goal_location(tick_data)

        # 准备速度输入
        gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32) # 控制器使用
        velocity = gt_velocity.reshape(1, 1) # transfuser使用

        # 解除卡住状态
        is_stuck = False
        # 除以2因为每隔一帧处理一次
        # 1100 = 55秒 * 20帧/秒，移动1.5秒 = 30帧来解除卡住
        if(self.stuck_detector > self.config.stuck_threshold and self.forced_move < self.config.creep_duration):
            print("Detected agent being stuck. Move for frame: ", self.forced_move)
            is_stuck = True
            self.forced_move += 1


        # 前向传播
        with torch.no_grad():
            pred_wps = []
            bounding_boxes = []
            for i in range(self.model_count):
                rotated_bb = []
                if (self.backbone == 'transFuser'):
                    pred_wp, _ = self.nets[i].forward_ego(image, lidar_bev, target_point, target_point_image, velocity,
                                                          num_points=num_points, save_path=SAVE_PATH, stuck_detector=self.stuck_detector,
                                                          forced_move=is_stuck, debug=self.config.debug, rgb_back=self.rgb_back)
                elif (self.backbone == 'late_fusion'):
                    pred_wp, _ = self.nets[i].forward_ego(image, lidar_bev, target_point, target_point_image, velocity, num_points=num_points)
                elif (self.backbone == 'geometric_fusion'):
                    bev_points = list()
                    cam_points = list()

                    curr_bev_points, curr_cam_points = lidar_bev_cam_correspondences(deepcopy(tick_data['lidar']), lidar_bev, image, self.step, False)
                    bev_points.append(torch.from_numpy(curr_bev_points).unsqueeze(0))
                    cam_points.append(torch.from_numpy(curr_cam_points).unsqueeze(0))

                    bev_points = bev_points[0].long().to('cuda', dtype=torch.int64)
                    cam_points = cam_points[0].long().to('cuda', dtype=torch.int64)
                    pred_wp, _ = self.nets[i].forward_ego(image, lidar_bev, target_point, target_point_image, velocity, bev_points, cam_points, num_points=num_points)
                elif (self.backbone == 'latentTF'):
                    pred_wp, rotated_bb = self.nets[i].forward_ego(image, lidar_bev, target_point, target_point_image, velocity, num_points=num_points)
                elif (self.backbone == 'crossvit_fusion'):
                    pred_wp, _ = self.nets[i].forward_ego(image, lidar_bev, target_point, target_point_image, velocity, num_points=num_points)
                elif (self.backbone == 'image_only'):
                    pred_wp, _ = self.nets[i].forward_ego(image, lidar_bev, target_point, target_point_image, velocity, num_points=num_points)
                elif (self.backbone == 'lidar_only'):
                    pred_wp, _ = self.nets[i].forward_ego(image, lidar_bev, target_point, target_point_image, velocity, num_points=num_points)
                else:
                    raise ValueError("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF, crossvit_fusion, image_only, lidar_only")

                pred_wps.append(pred_wp)
                bounding_boxes.append(rotated_bb)

        bbs_vehicle_coordinate_system = self.non_maximum_suppression(bounding_boxes, self.iou_treshold_nms)

        self.bb_buffer.append(bbs_vehicle_coordinate_system)
        self.pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0) # 对集成模型的预测取平均

        # 转换到局部坐标系
        pred_wp_transformed = []
        for i, degree in enumerate(self.aug_degrees):
            rad = np.deg2rad(degree)
            degree_matrix = np.array([[np.cos(rad), np.sin(rad)],
                                [-np.sin(rad), np.cos(rad)]])
            # 求逆
            degree_matrix = degree_matrix.T
            cur_pred_wp = self.pred_wp[i].detach().cpu().numpy()
            transformed_wp = (degree_matrix @ cur_pred_wp.T).T
            pred_wp_transformed.append(transformed_wp)

        self.pred_wp = np.stack(pred_wp_transformed, axis=0)
        self.pred_wp = torch.median(torch.from_numpy(self.pred_wp).to('cuda', dtype=torch.float32), dim=0, keepdims=True)[0]

        if (self.backbone == 'latentTF'):
            safety_box = []
            if(self.bb_detected_in_front_of_vehicle(gt_velocity) == True):
                safety_box.append(True)
        else:
            # 安全检查
            safety_box = deepcopy(tick_data['lidar'])
            safety_box[:, 1] *= -1  # 取反

            # z轴
            safety_box      = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
            safety_box      = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]

            # y轴
            safety_box      = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
            safety_box      = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]

            # x轴
            safety_box      = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
            safety_box      = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]

        steer, throttle, brake = self.nets[0].control_pid(self.pred_wp, gt_velocity, is_stuck)
        
        if is_stuck and self.forced_move==1: # 解除卡住时初始帧不转向
            steer = 0.0

        # 转向调制
        if brake or is_stuck:
            steer *= self.steer_damping
        if(gt_velocity < 0.1): # 0.1是判断车辆停止的任意低阈值
            self.stuck_detector += 1
        elif(gt_velocity > 0.1 and is_stuck == False):
            self.stuck_detector = 0
            self.forced_move    = 0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        # 安全控制器，当车辆正前方有障碍物时停车
        if self.use_lidar_safe_check:
            emergency_stop = (len(safety_box) > 0) # 检查列表是否为空
            if ((emergency_stop == True) and (is_stuck == True)):  # 只在解除卡住时使用安全框
                print("Detected object directly in front of the vehicle. Stopping. Step:", self.step)
                control.steer = float(steer)
                control.throttle = float(0.0)
                control.brake = float(True)
                # 将覆盖卡住检测器，如果卡在交通中我们确实想等待

        self.control = control

        self.update_gps_buffer(self.control, tick_data['compass'], tick_data['speed'])
        return control

    def bb_detected_in_front_of_vehicle(self, ego_speed):
        if (len(self.bb_buffer) < 1):  # We only start after we have 4 time steps.
            return False

        collision_predicted = False

        # 标准自车的尺寸
        extent_x = self.config.ego_extent_x
        extent_y = self.config.ego_extent_y
        extent_z = self.config.ego_extent_z
        extent = carla.Vector3D(extent_x, extent_y, extent_z)

        # 安全框
        bremsweg = ((ego_speed.cpu().numpy().item() * 3.6) / 10.0) ** 2 / 2.0  # 紧急制动的制动距离公式
        safety_x = np.clip(bremsweg + 1.0, a_min=2.0, a_max=4.0)  # 加一米是车辆本身

        center_safety_box = carla.Location(x=safety_x, y=0.0, z=1.0)

        safety_bounding_box = carla.BoundingBox(center_safety_box, extent)
        safety_bounding_box.rotation = carla.Rotation(0.0,0.0,0.0)

        for bb in self.bb_buffer[-1]:
            bb_orientation = self.get_bb_yaw(bb)
            bb_extent_x = 0.5 * np.sqrt((bb[3, 0] - bb[0, 0]) ** 2 + (bb[3, 1] - bb[0, 1]) ** 2)
            bb_extent_y = 0.5 * np.sqrt((bb[0, 0] - bb[1, 0]) ** 2 + (bb[0, 1] - bb[1, 1]) ** 2)
            bb_extent_z = 1.0  # 给定任意高度，不影响结果
            loc_local = carla.Location(bb[4,0], bb[4,1], 0.0)
            extent_det = carla.Vector3D(bb_extent_x, bb_extent_y, bb_extent_z)
            bb_local = carla.BoundingBox(loc_local, extent_det)
            bb_local.rotation = carla.Rotation(0.0, np.rad2deg(bb_orientation).item(), 0.0)

            if (self.check_obb_intersection(safety_bounding_box, bb_local) == True):
                collision_predicted = True

        return collision_predicted

    def non_maximum_suppression(self, bounding_boxes, iou_treshhold):
        filtered_boxes = []
        bounding_boxes = np.array(list(itertools.chain.from_iterable(bounding_boxes)), dtype=np.object)

        if(bounding_boxes.size == 0): #If no bounding boxes are detected can't do NMS
            return filtered_boxes


        confidences_indices = np.argsort(bounding_boxes[:, 2])
        while (len(confidences_indices) > 0):
            idx = confidences_indices[-1]
            current_bb = bounding_boxes[idx, 0]
            filtered_boxes.append(current_bb)
            confidences_indices = confidences_indices[:-1] # 从列表中移除最后一个元素

            if(len(confidences_indices) == 0):
                break

            for idx2 in deepcopy(confidences_indices):
                if(self.iou_bbs(current_bb, bounding_boxes[idx2, 0]) > iou_treshhold): # 从列表中移除边界框
                    confidences_indices = confidences_indices[confidences_indices != idx2]

        return filtered_boxes

    def update_gps_buffer(self, control, theta, speed):
        yaw = np.array([(theta - np.pi/2.0)])
        speed = np.array([speed])
        action = np.array(np.stack([control.steer, control.throttle, control.brake], axis=-1))

        # 更新GPS位置
        for i in range(len(self.gps_buffer)):
            loc =self.gps_buffer[i]
            loc_temp = np.array([loc[1], -loc[0]]) # 自行车模型使用不同的坐标系
            next_loc_tmp, _, _ = self.ego_model.forward(loc_temp, yaw, speed, action)
            next_loc = np.array([-next_loc_tmp[1], next_loc_tmp[0]])
            self.gps_buffer[i] = next_loc

        return None

    def get_bb_yaw(self, box):
        location_2 = box[2]
        location_3 = box[3]
        location_4 = box[4]
        center_top = (0.5 * (location_3 - location_2)) + location_2
        vector_top = center_top - location_4
        rotation_yaw = np.arctan2(vector_top[1], vector_top[0])

        return rotation_yaw

    def prepare_image(self, tick_data):
        image = Image.fromarray(tick_data['rgb'])
        image_degrees = []
        for degree in self.aug_degrees:
            crop_shift = degree / 60 * self.config.img_width
            rgb = torch.from_numpy(self.shift_x_scale_crop(image, scale=self.config.scale, crop=self.config.img_resolution, crop_shift=crop_shift)).unsqueeze(0)
            image_degrees.append(rgb.to('cuda', dtype=torch.float32))
        image = torch.cat(image_degrees, dim=0)
        return image

    def iou_bbs(self, bb1, bb2):
        a = Polygon([(bb1[0,0], bb1[0,1]), (bb1[1,0], bb1[1,1]), (bb1[2,0], bb1[2,1]), (bb1[3,0], bb1[3,1])])
        b = Polygon([(bb2[0,0], bb2[0,1]), (bb2[1,0], bb2[1,1]), (bb2[2,0], bb2[2,1]), (bb2[3,0], bb2[3,1])])
        intersection_area = a.intersection(b).area
        union_area = a.union(b).area
        iou = intersection_area / union_area
        return iou
    
    
    def dot_product(self, vector1, vector2):
        return (vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z)

    def cross_product(self, vector1, vector2):
        return carla.Vector3D(x=vector1.y * vector2.z - vector1.z * vector2.y, y=vector1.z * vector2.x - vector1.x * vector2.z, z=vector1.x * vector2.y - vector1.y * vector2.x)

    def get_separating_plane(self, rPos, plane, obb1, obb2):
        ''' 检查是否存在分离平面
        rPos Vec3
        plane Vec3
        obb1  边界框
        obb2 边界框
        '''
        return (abs(self.dot_product(rPos, plane)) > (abs(self.dot_product((obb1.rotation.get_forward_vector() * obb1.extent.x), plane)) +
                                                      abs(self.dot_product((obb1.rotation.get_right_vector()   * obb1.extent.y), plane)) +
                                                      abs(self.dot_product((obb1.rotation.get_up_vector()      * obb1.extent.z), plane)) +
                                                      abs(self.dot_product((obb2.rotation.get_forward_vector() * obb2.extent.x), plane)) +
                                                      abs(self.dot_product((obb2.rotation.get_right_vector()   * obb2.extent.y), plane)) +
                                                      abs(self.dot_product((obb2.rotation.get_up_vector()      * obb2.extent.z), plane)))
                )
    
    def check_obb_intersection(self, obb1, obb2):
        RPos = obb2.location - obb1.location
        return not(self.get_separating_plane(RPos, obb1.rotation.get_forward_vector(), obb1, obb2) or
                   self.get_separating_plane(RPos, obb1.rotation.get_right_vector(),   obb1, obb2) or
                   self.get_separating_plane(RPos, obb1.rotation.get_up_vector(),      obb1, obb2) or
                   self.get_separating_plane(RPos, obb2.rotation.get_forward_vector(), obb1, obb2) or
                   self.get_separating_plane(RPos, obb2.rotation.get_right_vector(),   obb1, obb2) or
                   self.get_separating_plane(RPos, obb2.rotation.get_up_vector(),      obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_forward_vector()), obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_right_vector()),   obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_up_vector()),      obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector()  , obb2.rotation.get_forward_vector()), obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector()  , obb2.rotation.get_right_vector()),   obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector()  , obb2.rotation.get_up_vector()),      obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector()     , obb2.rotation.get_forward_vector()), obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector()     , obb2.rotation.get_right_vector()),   obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector()     , obb2.rotation.get_up_vector()),      obb1, obb2))



    def prepare_lidar(self, tick_data):
        lidar_transformed = deepcopy(tick_data['lidar']) 
        lidar_transformed[:, 1] *= -1  # invert
        lidar_transformed = torch.from_numpy(lidar_to_histogram_features(lidar_transformed)).unsqueeze(0)
        lidar_transformed_degrees = [lidar_transformed.to('cuda', dtype=torch.float32)]
        lidar_bev = torch.cat(lidar_transformed_degrees[::-1], dim=1)
        return lidar_bev

    def prepare_goal_location(self, tick_data):
        tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
                                            torch.FloatTensor([tick_data['target_point'][1]])]
        target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

        target_point_image_degrees = []
        target_point_degrees = []
        for degree in self.aug_degrees:
            rad = np.deg2rad(degree)
            degree_matrix = np.array([[np.cos(rad), np.sin(rad)],
                                [-np.sin(rad), np.cos(rad)]])

            current_target_point = (degree_matrix @ target_point[0].cpu().numpy().reshape(2, 1)).T

            target_point_image = draw_target_point(current_target_point[0])
            target_point_image = torch.from_numpy(target_point_image)[None].to('cuda', dtype=torch.float32)
            target_point_image_degrees.append(target_point_image)
            target_point_degrees.append(torch.from_numpy(current_target_point))

        target_point_image = torch.cat(target_point_image_degrees, dim=0)
        target_point = torch.cat(target_point_degrees, dim=0).to('cuda', dtype=torch.float32)

        return target_point_image, target_point

    def scale_crop(self, image, scale=1, start_x=0, crop_x=None, start_y=0, crop_y=None):
        (width, height) = (image.width // scale, image.height // scale)
        if scale != 1:
            image = image.resize((width, height))
        if crop_x is None:
            crop_x = width
        if crop_y is None:
            crop_y = height
            
        image = np.asarray(image)
        cropped_image = image[start_y:start_y+crop_y, start_x:start_x+crop_x]
        return cropped_image

    def shift_x_scale_crop(self, image, scale, crop, crop_shift=0):
        crop_h, crop_w = crop
        (width, height) = (int(image.width // scale), int(image.height // scale))
        im_resized = image.resize((width, height))
        image = np.array(im_resized)
        start_y = height//2 - crop_h//2
        start_x = width//2 - crop_w//2
        
        # 只在x方向移动
        start_x += int(crop_shift // scale)
        cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
        cropped_image = np.transpose(cropped_image, (2,0,1))
        return cropped_image

    def destroy(self):
        del self.nets

# 取自LBC
class RoutePlanner(object):
    def __init__(self, min_distance, max_distance):
        self.saved_route = deque()
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.is_last = False

        self.mean = np.array([0.0, 0.0]) # 用于carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945]) # 用于carla 9.10

    def set_route(self, global_plan, gps=False):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean

            self.route.append((pos, cmd))

    def run_step(self, gps):
        if len(self.route) <= 2:
            self.is_last = True
            return self.route

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        return self.route

    def save(self):
        self.saved_route = deepcopy(self.route)

    def load(self):
        self.route = self.saved_route
        self.is_last = False

# 取自World on Rails
class EgoModel():
    def __init__(self, dt=1./4):
        self.dt = dt
        
        # 运动学自行车模型，参数来自World on Rails的调优结果
        self.front_wb    = -0.090769015
        self.rear_wb     = 1.4178275

        self.steer_gain  = 0.36848336
        self.brake_accel = -4.952399
        self.throt_accel = 0.5633837

    def forward(self, locs, yaws, spds, acts):
        # 运动学自行车模型，参数来自World on Rails的调优结果
        steer = acts[..., 0:1].item()
        throt = acts[..., 1:2].item()
        brake = acts[..., 2:3].astype(np.uint8)

        if (brake):
            accel = self.brake_accel
        else:
            accel = self.throt_accel * throt

        wheel = self.steer_gain * steer

        beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
        yaws = yaws.item()
        spds = spds.item()
        next_locs_0 = locs[0].item() + spds * math.cos(yaws + beta) * self.dt
        next_locs_1 = locs[1].item() + spds * math.sin(yaws + beta) * self.dt
        next_yaws = yaws + spds / self.rear_wb * math.sin(beta) * self.dt
        next_spds = spds + accel * self.dt
        next_spds = next_spds * (next_spds > 0.0)  # 快速ReLU

        next_locs = np.array([next_locs_0, next_locs_1])
        next_yaws = np.array(next_yaws)
        next_spds = np.array(next_spds)

        return next_locs, next_yaws, next_spds
