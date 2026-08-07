# MSO

**面向资源受限室内多机器人探索的预测式系统**

[English](README.md)

MSO（Make Sense at Once）是一个面向二维室内多机器人探索的研究系统，结合轻量局部占据地图预测、成对地图配准和观测约束规划。预测结构可用于配准和目标排序，但碰撞检查与持久地图更新仍以真实观测为准。

本仓库对应论文 **“A Predictive System for Multirobot Indoor Exploration under Resource Constraints”**。当前版本公开网络结构和 ROS 2 推理接口，但不是论文训练与全部实验的完整复现归档。

## 证据边界

- 部署的学生模型包含 342,771 个可训练参数。
- 输入为 256×256 的障碍、未知、自由三通道局部栅格。
- 输出为未知区域的占据概率。
- 新传感器观测会覆盖持久地图中的预测内容。
- 预测可参与目标排序；路径生成与碰撞检查使用实测占据地图。
- 成对配准面向没有预先已知机器人间相对位姿的相遇事件。

论文证据范围限于结构化二维室内环境，不证明跨建筑泛化、通信中断鲁棒性、错误提交后的在线恢复，或超出受控双机器人实物场景的规模扩展能力。

## 当前发布范围

| 组件 | 当前状态 |
|---|---|
| 学生与教师网络定义 | 已包含 |
| ROS 2 预测节点与启动文件 | 已包含 |
| 数据加载器 | 仅作为格式参考 |
| 论文历史训练程序 | 未包含 |
| 历史数据划分与 checkpoint 选择记录 | 未包含 |
| 论文 checkpoint | 未包含 |
| 部署版成对配准程序 | 未包含 |
| 离线变换评估器与事件记录 | 随论文送审数据包提供，不在本仓库 |
| 实物 rosbag | 文件过大，未放入本仓库 |
| KTH、HouseExpo、MRPB 原始数据 | 请从原提供方获取 |

`lightning_model.py`、`main_unet_gan.py` 和 `train_gan_new.py` 是历史研究原型。它们的默认参数与论文不同，依赖本仓库未跟踪的模块，也没有实现论文所述的完整四项蒸馏训练流程。因此不能把这些文件当作复现论文训练结果的标准入口。

## 运行环境

ROS 节点需要 Python 3、ROS 2、PyTorch、NumPy、OpenCV 和 scikit-learn；ROS 依赖包括 `rclpy`、`tf2_ros`、`geometry_msgs` 和 `nav_msgs`。当前版本没有提供锁定的依赖环境。

论文部署环境为 Ubuntu 22.04、ROS 2 Humble 和 NVIDIA Jetson AGX Orin。Jetson 上应安装与 JetPack 版本匹配的 PyTorch。

## 构建与启动

```bash
mkdir -p ~/mso_ws/src
cd ~/mso_ws/src
git clone git@github.com:SenseLabRobo4111/MSO.git
cd ~/mso_ws

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

当前版本不包含模型权重。启动时需传入绝对路径：

```bash
ros2 launch sensemap sensemap.launch.py \
  robot_id:=0 \
  model_path:=/absolute/path/to/mso_distilled.ckpt \
  architecture:=deconv
```

也可直接启动节点：

```bash
ros2 run sensemap sensemap_predictor --ros-args \
  -p robot_id:=0 \
  -p model_path:=/absolute/path/to/mso_distilled.ckpt \
  -p architecture:=deconv
```

默认 `deconv` 网络对应论文中的 342,771 参数学生模型。兼容的历史权重可显式选择 `architecture:=bilinear`。若路径为空或权重与结构不匹配，程序会直接报错。

## ROS 2 接口

当 `robot_id:=N` 时：

| 方向 | 名称 | 类型 | 用途 |
|---|---|---|---|
| 订阅 | `/robot_N/map` | `nav_msgs/msg/OccupancyGrid` | 实测占据地图 |
| 发布 | `/robot_N/predicted_map` | `nav_msgs/msg/OccupancyGrid` | 当前局部预测 |
| 发布 | `/robot_N/predicted_map_global` | `nav_msgs/msg/OccupancyGrid` | 累积预测地图 |
| 发布 | `/robot_N/way_point` | `geometry_msgs/msg/PoseStamped` | 可选前沿目标 |
| 坐标变换 | `global_map` 到 `robot_N/base_link` | TF2 | 局部裁剪所需位姿 |

推理回调每秒执行一次并刷新局部与全局预测。规划器可以更频繁地读取最近缓存，但读取频率不等于推理频率。

## 数据加载格式

`MapReconstructionDataset` 要求每个样本使用独立目录：

```text
dataset_root/
|-- train/
|   `-- sample_000001/
|       |-- obs_0.png
|       `-- local_map_0.png
`-- test/
    `-- ...
```

`obs_0.png` 为三通道观测，`local_map_0.png` 为二值占据目标。图像以最近邻方式缩放到 256×256。

上述目录名称仅描述历史加载器。当前版本没有样本级 floorplan 划分、独立 validation 划分，也没有证据证明 held-out evaluation 划分在历史模型选择过程中完全未被使用。

## 复现要求

完整训练归档仍需要准确的数据划分、教师与学生权重、checkpoint 选择规则、全部损失模块、标准训练入口、随机种子、评估掩码和锁定环境；这些材料不在当前版本中。仅有聚合曲线也不能重建逐 seed 不确定性。

## 引用与联系

在论文 DOI 和正式软件归档发布前，请引用实际使用的 Git commit。论文会明确记录送审版本。

通讯作者：Fei Qiao，`qiaofei@tsinghua.edu.cn`。

## 许可

当前仓库版本尚未授予软件使用许可。未来归档版本需明确最终许可及第三方组件限制。
