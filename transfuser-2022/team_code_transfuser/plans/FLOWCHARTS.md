# CrossViT适配修复流程图

## 问题诊断流程

```mermaid
graph TD
    A[开始: 使用CrossViT训练] --> B{能否导入CrossViTFusionBackbone?}
    B -->|否| C[检查crossvit_fusion.py是否存在]
    B -->|是| D{model.py中是否注册了crossvit_fusion?}
    
    C --> C1[问题: 文件缺失]
    
    D -->|否| E[问题1: model.py未注册backbone]
    D -->|是| F{train.py帮助信息是否包含crossvit_fusion?}
    
    F -->|否| G[问题2: train.py帮助信息缺失]
    F -->|是| H{config.py缩进是否正确?}
    
    H -->|否| I[问题3: config.py缩进错误]
    H -->|是| J{requirements.txt版本号是否完整?}
    
    J -->|否| K[问题4: requirements.txt缺少版本]
    J -->|是| L[所有问题已修复]
    
    E --> M[修复: 添加3处backbone分支]
    G --> N[修复: 更新帮助文本]
    I --> O[修复: 替换tab为空格]
    K --> P[修复: 添加open3d版本号]
    
    M --> Q[验证: 测试模型初始化]
    N --> Q
    O --> Q
    P --> Q
    
    Q --> R{所有测试通过?}
    R -->|是| S[完成: 可以正常训练]
    R -->|否| T[回到问题诊断]
    
    style E fill:#ff6b6b
    style G fill:#ff6b6b
    style I fill:#ffd93d
    style K fill:#ffd93d
    style S fill:#6bcf7f
```

## 修复实施流程

```mermaid
graph LR
    A[阶段1: 准备] --> B[阶段2: P0修复]
    B --> C[阶段3: P1修复]
    C --> D[阶段4: 验证]
    D --> E[阶段5: 完成]
    
    B1[修复model.py<br/>3处backbone注册] --> B
    B2[修复train.py<br/>帮助信息] --> B
    
    C1[修复config.py<br/>缩进问题] --> C
    C2[修复requirements.txt<br/>版本号] --> C
    
    D1[运行测试用例] --> D
    D2[回归测试] --> D
    
    style B fill:#ff6b6b
    style C fill:#ffd93d
    style D fill:#6bcf7f
    style E fill:#6bcf7f
```

## 文件依赖关系

```mermaid
graph TD
    A[train.py] --> B[model.py]
    A --> C[config.py]
    A --> D[data.py]
    
    B --> E[crossvit_fusion.py]
    B --> F[transfuser.py]
    B --> G[late_fusion.py]
    B --> H[latentTF.py]
    B --> I[geometric_fusion.py]
    
    E --> J[timm库]
    E --> K[torch]
    
    B --> C
    E --> C
    
    L[requirements.txt] -.定义依赖.-> J
    L -.定义依赖.-> K
    L -.定义依赖.-> M[open3d]
    
    style E fill:#95e1d3
    style B fill:#f38181
    style C fill:#ffd93d
    style L fill:#aa96da
```

## CrossViT集成架构

```mermaid
graph TB
    subgraph 输入层
        A1[RGB图像<br/>3x160x704]
        A2[LiDAR BEV<br/>2x256x256]
        A3[速度<br/>1维]
    end
    
    subgraph 编码器
        B1[ImageCNN<br/>ResNet/RegNet]
        B2[LidarEncoder<br/>ResNet/RegNet]
    end
    
    subgraph CrossViT融合
        C1[Layer1 + CrossViT1]
        C2[Layer2 + CrossViT2]
        C3[Layer3 + CrossViT3]
        C4[Layer4 + CrossViT4]
    end
    
    subgraph 输出
        D1[FPN特征金字塔]
        D2[图像特征网格]
        D3[融合全局特征]
    end
    
    A1 --> B1
    A2 --> B2
    A3 --> C1
    A3 --> C2
    A3 --> C3
    A3 --> C4
    
    B1 --> C1
    B2 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
    
    C4 --> D1
    C4 --> D2
    C4 --> D3
    
    style C1 fill:#95e1d3
    style C2 fill:#95e1d3
    style C3 fill:#95e1d3
    style C4 fill:#95e1d3
```

## 修复优先级矩阵

```mermaid
quadrantChart
    title 问题严重程度 vs 修复难度
    x-axis 修复难度低 --> 修复难度高
    y-axis 严重程度低 --> 严重程度高
    
    quadrant-1 高优先级
    quadrant-2 立即修复
    quadrant-3 低优先级
    quadrant-4 中优先级
    
    model.py backbone注册: [0.3, 0.95]
    train.py帮助信息: [0.1, 0.85]
    config.py缩进: [0.4, 0.6]
    requirements.txt版本: [0.2, 0.5]
```

## 测试验证流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant T as train.py
    participant M as model.py
    participant C as CrossViTFusionBackbone
    participant D as 数据加载器
    
    U->>T: python train.py --backbone crossvit_fusion
    T->>M: 创建LidarCenterNet模型
    M->>M: 检查backbone参数
    
    alt backbone == 'crossvit_fusion'
        M->>C: 初始化CrossViTFusionBackbone
        C-->>M: 返回模型实例
        M-->>T: 模型创建成功
    else backbone不支持
        M-->>T: 抛出异常
        T-->>U: 错误: backbone不存在
    end
    
    T->>D: 加载训练数据
    D-->>T: 返回batch数据
    
    loop 每个训练步骤
        T->>M: forward(rgb, lidar, velocity)
        M->>C: 前向传播
        C-->>M: 返回features
        M-->>T: 返回loss
        T->>T: 反向传播和优化
    end
    
    T-->>U: 训练完成
```

---

## 使用说明

这些流程图帮助理解:
1. **问题诊断流程**: 如何识别和定位问题
2. **修复实施流程**: 按什么顺序修复
3. **文件依赖关系**: 各文件之间的关系
4. **CrossViT架构**: 数据如何在模型中流动
5. **优先级矩阵**: 哪些问题最紧急
6. **测试验证**: 如何验证修复是否成功
