# MSO

**面向多机器人探索的轻量局部地图预测与融合**

[English](README.md)

MSO（Make Sense at Once）是一个面向二维室内多机器人探索的研究系统。它将
轻量局部占据地图预测器与成对地图配准结合，使机器人在没有预先已知全局相对位姿
的情况下利用预测结构。

本仓库对应论文 **“Lightweight Local Map Prediction and Fusion for Multirobot
Exploration”**。当前版本包含预测器网络、ROS 2 推理节点、launch 文件以及训练
相关模型组件。完整归档版本还将提供论文结果所对应的冻结评估工具、配置和版本化
模型权重。

## 方法概览

- 342K 参数的蒸馏模型在机器人端完成局部占据地图预测。
- 观测、未知和占据单元以三通道局部地图表示。
- ROS 2 节点发布局部预测地图和累计全局预测地图。
- 机器人交换地图时执行成对配准；预测单元是临时信息，后续传感器观测具有更高
  优先级。
- 预测仅参与前沿排序，路径生成和碰撞检查仍受真实观测自由空间约束。

论文证据范围限定在结构化室内环境，不主张已经证明跨建筑泛化、通信中断鲁棒性
或超出已测试机器人规模的真机扩展能力。

## 仓库结构

```text
MSO/
├── launch/
│   └── sensemap.launch.py       # ROS 2 预测器 launch 文件
├── resource/
│   └── sensemap                 # ament 包标记
├── sensemap/
│   ├── explore_model/
│   │   ├── SenseMapNet.py       # 学生与教师网络
│   │   ├── critic_model.py      # 对抗判别器
│   │   ├── dataset.py           # 占据地图数据加载器
│   │   ├── ffc.py               # Fast Fourier Convolution 模块
│   │   ├── lightning_model.py   # 训练封装
│   │   ├── main_unet_gan.py     # 研究训练入口
│   │   └── train_gan_new.py     # 蒸馏训练循环
│   └── predict_map.py           # ROS 2 推理节点
├── package.xml
├── setup.cfg
└── setup.py
```

## 当前发布范围

本仓库正在整理为版本化科研软件归档。下表明确区分当前已包含内容和待归档内容，
避免将部分代码误认为完整实验包。

| 组件 | 当前状态 |
|---|---|
| 预测器网络定义 | 已包含 |
| ROS 2 预测节点和 launch 文件 | 已包含 |
| 数据加载器及训练相关模块 | 已包含 |
| 论文所用训练权重 | 正在准备归档版本 |
| 完整成对配准与部署代码 | 可供编辑和审稿人核查；正在准备归档版本 |
| 冻结评估清单、事件日志和绘图脚本 | 可供编辑和审稿人核查；正在准备归档版本 |
| KTH、HouseExpo 和 MRPB 第三方数据 | 请从原始数据提供方获取 |
| 真机 rosbag 与处理后源数据 | 可供投稿核查；正在准备公开归档 |

## 环境要求

部署节点基于 ROS 2 和 Python 3，通常需要：

- ROS 2，以及 `rclpy`、`tf2_ros`、`geometry_msgs`、`nav_msgs`
- 与目标 CPU 或 CUDA 平台匹配的 PyTorch
- NumPy
- OpenCV
- scikit-learn

训练还需要 Pillow、Matplotlib、tqdm 和 PyTorch Lightning。论文所用的精确软
硬件环境将在归档版本中冻结。NVIDIA Jetson 平台应安装与当前 JetPack 对应的
PyTorch，不建议直接使用通用 wheel。

## 构建 ROS 2 包

```bash
mkdir -p ~/mso_ws/src
cd ~/mso_ws/src
git clone git@github.com:SenseLabRobo4111/MSO.git
cd ~/mso_ws

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

模型权重不提交到 Git。启动时必须传入绝对路径：

```bash
ros2 launch sensemap sensemap.launch.py \
  robot_id:=0 \
  model_path:=/absolute/path/to/mso_distilled.ckpt \
  architecture:=deconv
```

也可以直接启动节点：

```bash
ros2 run sensemap sensemap_predictor --ros-args \
  -p robot_id:=0 \
  -p model_path:=/absolute/path/to/mso_distilled.ckpt \
  -p architecture:=deconv
```

## 模型权重格式

推理节点支持两种格式：

1. 含 `state_dict` 的 PyTorch Lightning checkpoint，其中生成器键以 `gen.`
   开头；
2. `DistillMapNet` 的普通 state dictionary。

`model_path` 是必填参数。若省略路径或权重与所选蒸馏网络结构不一致，节点会立即
报错。默认 `deconv` 版本包含 342,771 个参数，对应论文报告的 342K 模型；兼容
旧 checkpoint 时可显式选择 `bilinear` 版本。

## ROS 2 接口

当 `robot_id:=N` 时，节点使用以下接口。

| 方向 | 名称 | 类型 | 用途 |
|---|---|---|---|
| 订阅 | `/robot_N/map` | `nav_msgs/msg/OccupancyGrid` | 已观测占据地图 |
| 发布 | `/robot_N/predicted_map` | `nav_msgs/msg/OccupancyGrid` | 当前局部预测 |
| 发布 | `/robot_N/predicted_map_global` | `nav_msgs/msg/OccupancyGrid` | 累计预测地图 |
| 发布 | `/robot_N/way_point` | `geometry_msgs/msg/PoseStamped` | 可选前沿目标接口 |
| 坐标变换 | `global_map` 到 `robot_N/base_link` | TF2 | 放置局部裁剪所需机器人位姿 |

默认预测周期为 1 秒。话题名和坐标系名称与论文中的部署命名保持一致。

## 数据目录格式

`MapReconstructionDataset` 要求每个样本使用独立目录：

```text
dataset_root/
├── train/
│   ├── sample_000001/
│   │   ├── obs_0.png
│   │   └── local_map_0.png
│   └── ...
└── test/
    └── ...
```

- `obs_0.png` 为三通道观测地图；
- `local_map_0.png` 为二值占据目标；
- 图像以最近邻方式缩放到 256 × 256 像素。

数据准备必须保留评估所用的固定划分清单。论文不会将现有 floorplan-disjoint
划分解释为 building-disjoint 泛化证据。

## 复现与结果记录

复现论文结果时，应保留准确的 checkpoint、数据划分、随机种子、规划配置、地图
分辨率、坐标系约定和评估掩码。配准实验还应保存估计与参考变换、门控决定、拒绝
原因和事件级编号。仅保留聚合曲线不足以审计 false accept、false reject 或位姿
误差。

## 引用

论文目前处于审稿阶段。软件归档记录和论文 DOI 发布后，请引用对应正式版本。在
此之前，请引用本仓库并注明固定 commit hash，以保证代码版本可追溯。

## 数据与代码可用性

编辑和审稿所需的自定义代码及处理后数据可向通讯作者获取。发表前将建立带持久
标识符的版本化归档。受数据许可限制，部分第三方原始数据不能在本仓库重复分发，
应从 KTH、HouseExpo 和 MRPB 的原始来源获取。

## 联系方式

论文和科研软件相关问题请联系通讯作者 Fei Qiao：`qiaofei@tsinghua.edu.cn`。

## 许可

当前仓库版本尚未授予软件使用许可。归档版本将注明最终批准的许可，以及第三方
组件适用的限制。
