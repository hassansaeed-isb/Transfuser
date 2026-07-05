# CrossViT适配快速修复指南

## 修复清单

### ✅ 必须修复 (P0 - CRITICAL)

#### 1. model.py - 添加CrossViT backbone支持 (3处)

**位置1**: 第572行后
```python
elif (backbone == 'crossvit_fusion'):
    self._model = CrossViTFusionBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
```

**位置2**: 第703行后
```python
elif (self.backbone == 'crossvit_fusion'):
    features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
```

**位置3**: 第752行后
```python
elif (self.backbone == 'crossvit_fusion'):
    features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
```

**同时更新3处错误信息**，在末尾添加 `, crossvit_fusion`

---

#### 2. train.py - 更新帮助信息

**位置**: 第49行
```python
help='Which Fusion backbone to use. Options: transFuser, late_fusion, latentTF, geometric_fusion, crossvit_fusion')
```

---

### ⚠️ 应该修复 (P1 - WARNING)

#### 3. config.py - 修复缩进 (第182-198行)

将所有tab字符替换为4个空格:
```python
    # GPT Encoder
    n_embd = 512
    block_exp = 4
    ...
    
    # CrossViT Encoder
    crossvit_blocks = 2
    crossvit_mlp_ratio = 4.0
    crossvit_qkv_bias = True
```

---

#### 4. requirements.txt - 添加版本号

**位置**: 第94行
```
open3d==0.16.0
```

---

## 快速验证

修复后运行:
```bash
# 测试CrossViT导入
python -c "from crossvit_fusion import CrossViTFusionBackbone; print('✓ CrossViT导入成功')"

# 测试模型初始化
python -c "from model import LidarCenterNet; from config import GlobalConfig; c = GlobalConfig(); m = LidarCenterNet(c, 'cuda', 'crossvit_fusion', 'resnet34', 'resnet18', True); print('✓ 模型初始化成功')"

# 测试示例脚本
python example_crossvit_usage.py

# 查看训练帮助
python train.py --help | grep crossvit
```

---

## 使用CrossViT训练

```bash
python train.py \
    --id crossvit_exp1 \
    --backbone crossvit_fusion \
    --batch_size 4 \
    --lr 1e-4
```

---

详细信息请参考: [`crossvit_adaptation_fix_plan.md`](crossvit_adaptation_fix_plan.md)
