"""
全局配置模块 (GlobalConfig)

功能描述:
    定义TransFuser自动驾驶项目的所有配置参数
    包括数据处理、传感器参数、模型架构、训练设置等
    
主要类:
    GlobalConfig: 全局配置类，包含所有超参数和设置
    
配置分类:
    1. 数据时序/分辨率配置: 图像和LiDAR的输入输出尺寸
    2. 传感器安装参数: 相机和LiDAR的位置和朝向
    3. 数据增强/调试配置: 训练时的数据增强策略
    4. 激光雷达离散化配置: Point Pillars相关参数
    5. 模型主干网络配置: 网络架构和检测头参数
    6. 语义分割类别配置: 类别定义和颜色映射
    7. 优化配置: 学习率和损失权重
    8. GPT/CrossViT编码器配置: Transformer参数
    9. 控制器配置: PID控制器参数
    
使用示例:
    from config import GlobalConfig
    
    # 训练时使用所有城镇数据
    config = GlobalConfig(root_dir='/path/to/data', setting='all')
    
    # 使用Town02和Town05作为验证集
    config = GlobalConfig(root_dir='/path/to/data', setting='02_05_withheld')
    
    # 评估时不需要训练数据
    config = GlobalConfig(setting='eval')
    
    # 访问配置参数
    print(config.img_resolution)  # (160, 704)
    print(config.crossvit_blocks)  # 2

"""

import os

class GlobalConfig:
    """
    全局配置类 - 定义TransFuser项目的所有超参数
    
    功能:
        集中管理所有配置参数，包括数据处理、模型架构、训练策略等
        支持通过构造函数参数动态设置数据路径和训练设置
    
    参数:
        root_dir (str): 训练数据的根目录路径
        setting (str): 训练设置，可选值:
            - 'all': 使用所有城镇数据训练，无验证集
            - '02_05_withheld': 保留Town02和Town05作为验证集
            - 'eval': 评估模式，不需要训练数据
        **kwargs: 额外的配置参数，会覆盖默认值
    
    属性:
        train_data (list): 训练数据路径列表
        val_data (list): 验证数据路径列表
        train_towns (list): 训练城镇列表
        val_towns (list): 验证城镇列表
    
    示例:
        >>> config = GlobalConfig(root_dir='/data', setting='all')
        >>> print(config.img_resolution)
        (160, 704)
        >>> print(config.backbone)
        'transFuser'
    """
    
    # =========================================================================
    # 1. 数据时序/分辨率配置
    # =========================================================================
    
    seq_len = 1         # 输入时序长度（单帧输入，不使用历史帧）
    # 图像和激光雷达分别使用单帧的时序长度
    img_seq_len = 1     # 图像输入时序长度
    lidar_seq_len = 1   # LiDAR输入时序长度
    pred_len = 4        # 预测未来路点的数量（预测4个未来路点）
    scale = 1           # 图像预处理缩放因子（1=原始尺寸，2=放大2倍）
    
    # 图像预处理后的分辨率 (H, W)
    # 三个相机拼接后的宽度: 960*3/4 = 720, 但实际使用704
    img_resolution = (160, 704)  # 图像预处理后的分辨率，高度H=160，宽度W=704
    img_width = 320     # 单个相机图像宽度（与scale匹配），scale=1->320, scale=2->640
    
    # LiDAR点云体素化后的网格分辨率
    lidar_resolution_width  = 256  # LiDAR BEV图像宽度（像素）
    lidar_resolution_height = 256  # LiDAR BEV图像高度（像素）
    pixels_per_meter = 8.0         # 像素/米，即1像素=0.125米

    # =========================================================================
    # 2. 传感器安装参数
    # =========================================================================
    
    # LiDAR传感器安装参数
    lidar_pos = [1.3, 0.0, 2.5]    # LiDAR安装位置 [x, y, z]（米）
                                    # x=1.3: 车辆前方1.3米
                                    # y=0.0: 车辆中心线
                                    # z=2.5: 离地2.5米
    lidar_rot = [0.0, 0.0, -90.0]  # LiDAR旋转角 [Roll, Pitch, Yaw]（度）
                                    # Yaw=-90: 绕z轴旋转-90度

    # 相机传感器安装参数
    camera_pos = [1.3, 0.0, 2.3]   # 相机安装位置 [x, y, z]（米）
    camera_width = 960              # 相机原始图像宽度（像素）
    camera_height = 480             # 相机原始图像高度（像素）
    camera_fov = 120                # 相机视场角FOV（度）
    
    # 三个相机的旋转角（Roll, Pitch, Yaw，单位度）
    camera_rot_0 = [0.0, 0.0, 0.0]    # 相机0: 正前方
    camera_rot_1 = [0.0, 0.0, -60.0]  # 相机1: 左前方60度
    camera_rot_2 = [0.0, 0.0, 60.0]   # 相机2: 右前方60度
    
    # BEV（鸟瞰图）损失的分辨率
    # 注意: 如果宽高不对称，需要检查宽高是否互换
    bev_resolution_width  = 160  # BEV损失上采样后的宽度（像素）
    bev_resolution_height = 160  # BEV损失上采样后的高度（像素）

    # =========================================================================
    # 3. 数据增强/调试配置
    # =========================================================================
    
    use_target_point_image = False  # 是否将目标点渲染到LiDAR BEV图像中
    gru_concat_target_point = True  # GRU路点预测时是否拼接目标点特征
    augment = True                  # 是否使用数据增强（旋转、翻转等）
    inv_augment_prob = 0.1          # 不启用增强的概率（实际增强概率=1-0.1=0.9）
    aug_max_rotation = 20           # 数据增强的最大旋转角度（度）
    debug = True                    # 调试模式：可视化模型输入输出并保存到save_path
    sync_batch_norm = False         # 是否启用同步批归一化（多GPU训练时使用）
    train_debug_save_freq = 50      # 训练时调试文件保存频率（每50步保存一次）

    bb_confidence_threshold = 0.3   # 边界框置信度阈值（高于0.3才认为是有效检测）

    # =========================================================================
    # 4. 激光雷达离散化配置（仅Point Pillars使用）
    # =========================================================================
    
    # 激光雷达离散化配置，仅用于Point Pillars
    use_point_pillars = False       # 是否使用Point Pillars激光雷达处理方法
                                    # False: 使用体素化方法（默认）
                                    # True: 使用Point Pillars方法
    max_lidar_points = 40000        # 激光雷达最大点数限制
    
    # 激光雷达点云的空间范围（米）
    min_x = -16   # x轴最小值（车辆后方16米）
    max_x = 16    # x轴最大值（车辆前方16米）
    min_y = -32   # y轴最小值（车辆前方32米）
    max_y = 0     # y轴最大值（车辆位置）
    
    num_input = 9           # Point Pillars输入特征维度（每个点的特征数）
    num_features = [32, 32] # Point Pillars各层的特征维度

    # =========================================================================
    # 5. 模型主干网络/检测头配置
    # =========================================================================
    
    # 主干网络类型
    # 可选: 'transFuser', 'late_fusion', 'latentTF', 'geometric_fusion', 'crossvit_fusion'
    backbone = 'transFuser'

    # CenterNet检测头参数
    num_dir_bins = 12                   # 方向角分箱数量（将360度分成12个区间）
    fp16_enabled = False                # 是否启用FP16混合精度训练
    center_net_bias_init_with_prob = 0.1  # 偏置初始化概率（用于focal loss）
    center_net_normal_init_std = 0.001  # 正态分布初始化标准差
    top_k_center_keypoints = 100        # 保留的Top-K中心点数量
    center_net_max_pooling_kernel = 3   # 最大池化核大小（用于NMS）
    channel = 64                        # CenterNet基础通道数

    bounding_box_divisor = 2.0          # 边界框宽高缩放因子（数据采集时的缩放）
                                        # 注意: 后续数据编辑需修复，移除该参数
    draw_brake_threshhold = 0.5         # 刹车可视化阈值（刹车值>0.5时用刹车颜色绘制）

    # Waypoint GRU路点预测网络
    gru_hidden_size = 64  # GRU隐藏层维度

    # =========================================================================
    # 6. 语义分割/类别配置
    # =========================================================================
    
    num_class = 7  # 语义分割类别数量
    
    # 类别ID到RGB颜色的映射（用于可视化）
    classes = {
        0: [0, 0, 0],       # 未标注 (黑色)
        1: [0, 0, 255],     # 车辆 (蓝色)
        2: [128, 64, 128],  # 道路 (紫灰色)
        3: [255, 0, 0],     # 红灯 (红色)
        4: [0, 255, 0],     # 行人 (绿色)
        5: [157, 234, 50],  # 道路线 (黄绿色)
        6: [255, 255, 255], # 人行道 (白色)
    }
    
    # 类别颜色列表（BGR格式，用于OpenCV可视化）
    classes_list = [
        [0, 0, 0],       # 未标注 (黑色)
        [255, 0, 0],     # 车辆 (蓝色，BGR格式)
        [128, 64, 128],  # 道路 (紫灰色)
        [0, 0, 255],     # 红灯 (红色，BGR格式)
        [0, 255, 0],     # 行人 (绿色)
        [50, 234, 157],  # 道路线 (黄绿色，BGR格式)
        [255, 255, 255], # 人行道 (白色)
    ]
    
    # CARLA语义标签到项目类别ID的转换表
    # 索引: CARLA原始标签ID
    # 值: 项目内部类别ID (0=未标注, 1=车辆, 2=道路, 3=红灯, 4=行人, 5=道路线, 6=人行道)
    converter = [
        0,  # 0: unlabeled (未标注)
        0,  # 1: building (建筑物) -> 未标注
        0,  # 2: fence (围栏) -> 未标注
        0,  # 3: other (其他) -> 未标注
        4,  # 4: pedestrian (行人) -> 行人
        0,  # 5: pole (电线杆) -> 未标注
        5,  # 6: road line (道路线) -> 道路线
        2,  # 7: road (道路) -> 道路
        6,  # 8: sidewalk (人行道) -> 人行道
        0,  # 9: vegetation (植被) -> 未标注
        1,  # 10: vehicle (车辆) -> 车辆
        0,  # 11: wall (墙壁) -> 未标注
        0,  # 12: traffic sign (交通标志) -> 未标注
        0,  # 13: sky (天空) -> 未标注
        0,  # 14: ground (地面) -> 未标注
        0,  # 15: bridge (桥梁) -> 未标注
        0,  # 16: rail track (铁轨) -> 未标注
        0,  # 17: guard rail (护栏) -> 未标注
        0,  # 18: traffic light (交通灯) -> 未标注
        0,  # 19: static (静态物体) -> 未标注
        0,  # 20: dynamic (动态物体) -> 未标注
        0,  # 21: water (水) -> 未标注
        0,  # 22: terrain (地形) -> 未标注
        3,  # 23: red light (红灯) -> 红灯
        3,  # 24: yellow light (黄灯) -> 红灯（视为红灯处理）
        0,  # 25: green light (绿灯) -> 未标注
        0,  # 26: stop sign (停车标志) -> 未标注
        5,  # 27: stop line marking (停车线标记) -> 道路线
    ]

    # =========================================================================
    # 7. 优化配置
    # =========================================================================
    
    lr = 1e-4       # 初始学习率
    multitask = True  # 是否使用多任务学习（同时训练语义分割和深度估计）
    ls_seg   = 1.0  # 语义分割损失权重
    ls_depth = 10.0 # 深度估计损失权重（较大权重因为深度值范围更大）

    # 卷积编码器的锚点配置（用于Transformer的位置编码）
    img_vert_anchors = 5        # 图像特征垂直方向的锚点数量
    img_horz_anchors = 20 + 2   # 图像特征水平方向的锚点数量（+2为额外token）
    lidar_vert_anchors = 8      # LiDAR特征垂直方向的锚点数量
    lidar_horz_anchors = 8      # LiDAR特征水平方向的锚点数量
    
    # 总锚点数量（用于Transformer序列长度计算）
    img_anchors = img_vert_anchors * img_horz_anchors      # 图像总锚点数 = 5*22 = 110
    lidar_anchors = lidar_vert_anchors * lidar_horz_anchors  # LiDAR总锚点数 = 8*8 = 64

    # 详细损失函数列表（用于日志记录和权重控制）
    detailed_losses = ['loss_wp',           # 路点预测损失
                       'loss_bev',          # BEV语义分割损失
                       'loss_depth',        # 深度估计损失
                       'loss_semantic',     # 语义分割损失
                       'loss_center_heatmap',  # 目标检测中心热图损失
                       'loss_wh',           # 目标检测宽高损失
                       'loss_offset',       # 目标检测偏移损失
                       'loss_yaw_class',    # 目标检测方向角分类损失
                       'loss_yaw_res',      # 目标检测方向角残差损失
                       'loss_velocity',     # 速度预测损失
                       'loss_brake']        # 刹车预测损失
    
    # 各损失函数的权重（与detailed_losses一一对应）
    detailed_losses_weights = [1.0,  # loss_wp: 路点损失权重
                               1.0,  # loss_bev: BEV损失权重
                               1.0,  # loss_depth: 深度损失权重
                               1.0,  # loss_semantic: 语义损失权重
                               0.2,  # loss_center_heatmap: 中心热图损失权重
                               0.2,  # loss_wh: 宽高损失权重
                               0.2,  # loss_offset: 偏移损失权重
                               0.2,  # loss_yaw_class: 方向角分类损失权重
                               0.2,  # loss_yaw_res: 方向角残差损失权重
                               0.0,  # loss_velocity: 速度损失权重（当前禁用）
                               0.0]  # loss_brake: 刹车损失权重（当前禁用）

    # 感知分支输出特征维度
    perception_output_features = 512  # 感知分支输出的特征向量维度
    bev_features_chanels = 64         # BEV特征金字塔的通道数
    bev_upsample_factor = 2           # BEV特征上采样因子

    # 反卷积层通道数（用于BEV特征解码）
    deconv_channel_num_1 = 128  # 第一个反卷积层的通道数
    deconv_channel_num_2 = 64   # 第二个反卷积层的通道数
    deconv_channel_num_3 = 32   # 第三个反卷积层的通道数

    # 反卷积层缩放因子（控制特征图尺寸变化）
    deconv_scale_factor_1 = 8  # 第一层后特征图尺寸放大8倍
    deconv_scale_factor_2 = 4  # 第二层后特征图尺寸放大4倍

    # =========================================================================
    # 8. 运行时配置（控制器和仿真器参数）
    # =========================================================================
    
    gps_buffer_max_len = 100    # 历史GPS测量值的缓冲区大小（保留最近100帧）
    carla_frame_rate = 1.0 / 20.0  # CARLA仿真器帧率（毫秒）= 50ms/帧
    carla_fps = 20              # 仿真器帧率（帧/秒）
    
    iou_treshold_nms = 0.2      # 边界框NMS的IoU阈值（用于集成预测的后处理）
    steer_damping = 0.5         # 刹车时方向盘阻尼系数（刹车时转向乘以此系数）
    
    route_planner_min_distance = 7.5   # 路线规划器的最小距离（米）
    route_planner_max_distance = 50.0  # 路线规划器的最大距离（米）
    
    action_repeat = 2           # 网络动作重复次数（=2因为LiDAR帧率是仿真器的一半）
    stuck_threshold = 1100/action_repeat  # 触发蠕行控制器的帧数阈值
    creep_duration = 30 / action_repeat   # 蠕行前进的持续帧数

    # 安全框尺寸（用于碰撞检测，坐标系以车辆为中心）
    safety_box_z_min = -2.0   # 安全框z轴最小值（米）
    safety_box_z_max = -1.05  # 安全框z轴最大值（米）

    safety_box_y_min = -3.0   # 安全框y轴最小值（米）
    safety_box_y_max = 0.0    # 安全框y轴最大值（米）

    safety_box_x_min = -1.066  # 安全框x轴最小值（米）
    safety_box_x_max = 1.066   # 安全框x轴最大值（米）

    # 自车（ego vehicle）的尺寸（半长/半宽/半高，米）
    ego_extent_x = 2.4508416652679443  # 自车x方向半长（前后方向）
    ego_extent_y = 1.0641621351242065  # 自车y方向半宽（左右方向）
    ego_extent_z = 0.7553732395172119  # 自车z方向半高（上下方向）

    # =========================================================================
    # 9. GPT/Transformer编码器配置
    # =========================================================================
    
    # GPT Encoder - 用于TransFuser和LatentTF的Transformer参数
    n_embd = 512        # 嵌入维度（Transformer的特征维度）
    block_exp = 4       # MLP扩展比例（FFN中间层维度 = n_embd * block_exp）
    n_layer = 8         # Transformer层数
    n_head = 4          # 多头注意力的头数
    n_scale = 4         # 多尺度融合的尺度数量
    embd_pdrop = 0.1    # 嵌入层Dropout概率
    resid_pdrop = 0.1   # 残差连接Dropout概率
    attn_pdrop = 0.1    # 注意力权重Dropout概率
    
    # GPT线性层初始化参数
    gpt_linear_layer_init_mean = 0.0   # 线性层权重初始化的正态分布均值
    gpt_linear_layer_init_std  = 0.02  # 线性层权重初始化的正态分布标准差
    gpt_layer_norm_init_weight = 1.0   # LayerNorm层的初始权重值
    
    # CrossViT Encoder - 用于CrossViT融合模块的参数
    crossvit_blocks = 2         # 每个尺度的CrossViT块数量
    crossvit_mlp_ratio = 4.0    # CrossViT中MLP的扩展比例
    crossvit_qkv_bias = True    # CrossViT的QKV投影是否使用偏置

    # =========================================================================
    # 10. PID控制器配置
    # =========================================================================
    
    # 转向PID控制器参数
    turn_KP = 1.25  # 转向比例增益
    turn_KI = 0.75  # 转向积分增益
    turn_KD = 0.3   # 转向微分增益
    turn_n = 20     # 转向控制器缓冲区大小（历史误差数量）

    # 速度PID控制器参数
    speed_KP = 5.0  # 速度比例增益
    speed_KI = 0.5  # 速度积分增益
    speed_KD = 1.0  # 速度微分增益
    speed_n = 20    # 速度控制器缓冲区大小

    default_speed = 4.0  # 蠕行时的默认速度（米/秒）

    # 油门和刹车控制参数
    max_throttle = 0.75   # 数据集中油门信号的上限值
    brake_speed = 0.4     # 触发刹车的速度阈值（低于此速度时刹车）
    brake_ratio = 1.1     # 触发刹车的速度比例（实际速度/期望速度 > 此值时刹车）
    clip_delta = 0.25     # 纵向控制器速度输入的最大变化量
    clip_throttle = 0.75  # 控制器允许的最大油门值

    def __init__(self, root_dir='', setting='all', **kwargs):
        """
        初始化全局配置
        
        功能:
            根据指定的训练设置，自动扫描数据目录并构建训练/验证数据路径列表
        
        参数:
            root_dir (str): 训练数据的根目录路径
                目录结构应为: root_dir/town_name/scenario_name/
            setting (str): 训练设置，可选值:
                - 'all': 使用所有城镇数据训练，第一个城镇的数据也用于验证
                - '02_05_withheld': 保留Town02和Town05作为验证集，其余用于训练
                - 'eval': 评估模式，不需要训练数据（跳过数据扫描）
            **kwargs: 额外的配置参数，会覆盖类的默认属性值
        
        属性设置:
            self.root_dir: 数据根目录
            self.train_towns: 训练城镇列表
            self.val_towns: 验证城镇列表
            self.train_data: 训练数据路径列表
            self.val_data: 验证数据路径列表
        
        示例:
            >>> config = GlobalConfig(root_dir='/data/carla', setting='all')
            >>> print(len(config.train_data))  # 训练数据数量
            >>> print(len(config.val_data))    # 验证数据数量
        """
        self.root_dir = root_dir
        
        if (setting == 'all'):
            # 使用所有城镇数据训练，无专门的验证集
            # 第一个城镇的数据同时用于验证（仅用于监控训练进度）
            self.train_towns =[town for town in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, town))]  # 获取所有城镇文件夹
            self.val_towns = [self.train_towns[0]]         # 使用第一个城镇作为验证
            self.train_data, self.val_data = [], []
            
            # 扫描训练数据
            for town in self.train_towns:
                root_files = os.listdir(os.path.join(self.root_dir, town))  # 获取城镇内的场景文件夹
                for file in root_files:
                    # 只添加文件夹（场景），跳过文件
                    if not os.path.isfile(os.path.join(self.root_dir, file)):
                        self.train_data.append(os.path.join(self.root_dir, town, file))
            
            # 扫描验证数据（与训练数据相同，仅用于监控）
            for town in self.val_towns:
                root_files = os.listdir(os.path.join(self.root_dir, town))
                for file in root_files:
                    if not os.path.isfile(os.path.join(self.root_dir, file)):
                        self.val_data.append(os.path.join(self.root_dir, town, file))

        elif (setting == '02_05_withheld'):
            # 保留Town02和Town05作为验证集，其余城镇用于训练
            print("Skip Town02 and Town05")
            self.train_towns = [town for town in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, town))]   # 所有城镇（包括02和05）
            self.val_towns = self.train_towns              # Town02和05会在下面自动筛选
            self.train_data, self.val_data = [], []
            
            # 扫描训练数据（排除Town02和Town05）
            for town in self.train_towns:
                root_files = os.listdir(os.path.join(self.root_dir, town))
                for file in root_files:
                    # 跳过Town02和Town05的数据（保留为测试集）
                    if ((file.find('Town02') != -1) or (file.find('Town05') != -1)):
                        continue
                    if not os.path.isfile(os.path.join(self.root_dir, file)):
                        print("Train Folder: ", file)
                        self.train_data.append(os.path.join(self.root_dir, town, file))
            
            # 扫描验证数据（只使用Town02和Town05）
            for town in self.val_towns:
                root_files = os.listdir(os.path.join(self.root_dir, town))
                for file in root_files:
                    # 只使用Town02和Town05的数据
                    if ((file.find('Town02') == -1) and (file.find('Town05') == -1)):
                        continue
                    if not os.path.isfile(os.path.join(self.root_dir, file)):
                        print("Val Folder: ", file)
                        self.val_data.append(os.path.join(self.root_dir, town, file))
                        
        elif (setting == 'eval'):
            # 评估模式：不需要训练数据，跳过数据扫描
            pass
        else:
            print("Error: Selected setting: ", setting, " does not exist.")

        # 允许通过kwargs动态覆盖任何配置参数
        for k, v in kwargs.items():
            setattr(self, k, v)
