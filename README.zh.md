# MSO

**面向资源受限室内多机器人探索的预测式系统**

[English](README.md)

MSO（Make Sense at Once）是一个面向二维室内多机器人探索的研究系统，结合轻量局部占据地图预测、成对地图配准和观测约束规划。预测结构可以辅助配准和目标排序，但碰撞检查与持久地图更新仍以真实观测为准。

本仓库对应论文 **“Design and Audit of Resource-Constrained Predictive Multirobot Exploration”**。当前版本包含可运行的 ROS 2 预测节点、明确标注为“重建”的训练流程、对历史双机器人配准数据的离线审计，以及用于未来实物团队采集的工具。它不是论文所有结果的完整历史复现包。

## 首先阅读

| 组件 | 是否包含 | 证据状态 |
|---|---:|---|
| 342,771 参数学生网络与教师网络定义 | 是 | 架构已核验 |
| ROS 2 预测节点与启动文件 | 是 | 运行接口可用 |
| 论文历史训练器与准确的 5,385/1,356 划分 | 否 | 未找回 |
| 论文 checkpoint 与模型选择记录 | 否 | 未找回 |
| 两份恢复出的生成器候选权重 | 是 | 与架构兼容，但不能认定为论文 checkpoint |
| 重建训练与评估流程 | 是 | 前向重建，不是历史复现 |
| 保留数据归档清单 | 是 | 8,304 条样本与源文件哈希；不是论文划分 |
| 历史双机器人配准回放 | 是 | 离线负面审计；不是部署验证 |
| 被动 N=2/3/5 采集协议 | 是 | 仅为采集工具；没有新的 N=3 或 N=5 实物结果 |
| 实物 rosbag | 否 | 本地清单已校验；受控审稿链接尚未配置 |

现有证据仅覆盖结构化二维室内环境，不能证明跨建筑泛化、通信中断运行、错误地图提交后的在线恢复，或超出受控双机器人场景的实物规模扩展。

## 目录结构

```text
MSO/
|-- experiments/
|   `-- physical_team/       # 被动 N=2/3/5 采集协议
|-- registration_replay/     # 历史双机器人数据的离线审计
|-- repro_reconstructed/     # 明确标注的训练重建与候选权重
|-- launch/
|   |-- sensemap.launch.py
|   |-- physical_team_capture.launch.py
|   |-- physical_team_3.launch.py
|   `-- physical_team_5.launch.py
|-- sensemap/
|   |-- explore_model/
|   |-- capture_event_logger.py
|   `-- predict_map.py
|-- test/
|-- package.xml
|-- setup.cfg
`-- setup.py
```

## ROS 2 运行

运行环境需要 Python 3、ROS 2、PyTorch、NumPy、OpenCV 和 scikit-learn，ROS 依赖已写入 `package.xml`。论文报告的部署环境为 Ubuntu 22.04、ROS 2 Humble 和 NVIDIA Jetson AGX Orin。Jetson 上应使用与已安装 JetPack 版本兼容的 PyTorch。

```bash
mkdir -p ~/mso_ws/src
cd ~/mso_ws/src
git clone git@github.com:SenseLabRobo4111/MSO.git
cd ~/mso_ws

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

当前版本不包含论文 checkpoint。启动时必须显式提供兼容的 checkpoint 或 state dictionary：

```bash
ros2 launch sensemap sensemap.launch.py \
  robot_id:=0 \
  model_path:=/absolute/path/to/student_weights.ckpt \
  architecture:=deconv
```

也可直接启动：

```bash
ros2 run sensemap sensemap_predictor --ros-args \
  -p robot_id:=0 \
  -p model_path:=/absolute/path/to/student_weights.ckpt \
  -p architecture:=deconv
```

默认 `deconv` 模型包含 342,771 个可训练参数。路径为空或权重与架构不匹配时会直接失败。`repro_reconstructed/` 中的恢复候选仅为研究工件，不能当作论文表格对应 checkpoint 使用。

### ROS 2 接口

当 `robot_id:=N` 时：

| 方向 | 名称 | 类型 | 用途 |
|---|---|---|---|
| 订阅 | `/robot_N/map` | `nav_msgs/msg/OccupancyGrid` | 实测占据地图 |
| 发布 | `/robot_N/predicted_map` | `nav_msgs/msg/OccupancyGrid` | 当前局部预测 |
| 发布 | `/robot_N/predicted_map_global` | `nav_msgs/msg/OccupancyGrid` | 累积预测 |
| 发布 | `/robot_N/way_point` | `geometry_msgs/msg/PoseStamped` | 可选前沿目标 |
| 坐标变换 | `global_map` 到 `robot_N/base_link` | TF2 | 局部裁剪使用的位姿 |

推理回调每秒运行一次。规划器可以更频繁地读取最近缓存，但读取频率不等于推理频率。

## 重建训练包

[`repro_reconstructed/`](repro_reconstructed/README.md) 提供：

- 统一的重建训练与评估入口；
- 两份生成器候选权重及源文件溯源；
- 对现存 6,841/1,463 目录标签生成的 8,304 条样本清单；
- 每个样本的源文件哈希与处理后样本对哈希；
- 必须显式指定 building、floorplan 或 scene 等语义分组的划分工具，并拒绝跨划分语义泄漏和完全重复样本；
- 锁定的重建依赖与工件校验工具。

使用前执行：

```bash
python3 repro_reconstructed/tools/verify_artifacts.py
python3 -m pytest repro_reconstructed/tests -q
```

现存数据归档不是论文报告的 5,385/1,356 划分。旧目录中的 worker 前缀表示并行数据生成进程，不代表建筑或楼层；工具会拒绝把 worker 前缀当成语义划分依据。

旧文件 `lightning_model.py`、`main_unet_gan.py` 和 `train_gan_new.py` 仍属于历史研究原型，依赖缺失模块，也没有实现论文完整训练流程。任何新实验都应使用重建入口并明确标注为重建实验。

## 配准离线审计

[`registration_replay/`](registration_replay/README.md) 对历史双机器人实物 bag 进行离线回放，重建旧仿射行为，并与单位尺度刚体估计器进行对照，同时记录结构检查和位姿参考检查。该审计暴露了历史配准证据中的失败模式，不能证明部署时的在线位姿精度、误接受/误拒绝率、回滚或恢复能力。

可复用的成对管理器默认关闭地图提交。任何未来提交路径都必须在新采集的事件级变换和独立位姿参考上单独验证。不得把该离线回放描述成成功的在线配准实验。

## 实物团队采集协议

[`experiments/physical_team/`](experiments/physical_team/README.md) 包含被动记录器、数据模式、预检、N=2/3/5 启动示例以及单次/整组诊断。它不包含同步三机器人或五机器人实物数据。诊断报告固定写入 `claim_authorized=false`；这些工具用于采集与检查，不是论文结论认证器。

## 数据加载格式

历史 `MapReconstructionDataset` 要求每个样本使用独立目录：

```text
dataset_root/
|-- train/
|   `-- sample_000001/
|       |-- obs_0.png
|       `-- local_map_0.png
`-- test/
    `-- ...
```

`obs_0.png` 为三通道观测，`local_map_0.png` 为二值占据目标；图像以最近邻方式缩放到 256×256。该目录结构只描述旧加载器，不能证明跨建筑划分，也不能证明评估集在历史模型选择中从未被使用。

## 校验

非 ROS 证据包可运行：

```bash
python3 repro_reconstructed/tools/verify_artifacts.py
python3 -m pytest registration_replay/tests repro_reconstructed/tests -q
```

ROS 包测试还需要常规 ROS 2 ament 测试插件。

## 引用与联系

在论文 DOI 和正式软件归档发布前，请引用实际使用的 Git commit。

通讯作者：Fei Qiao，`qiaofei@tsinghua.edu.cn`。

## 许可

MSO 作者拥有权利的源码与文档采用 BSD 3-Clause 许可证，详见 `LICENSE`。已标明的 FFC 实现与 ROS 2 测试模板继续适用其 Apache-2.0 上游条款。BSD 许可证不自动扩大恢复 checkpoint、外部数据集或原始 rosbag 的再利用权利；具体边界见 `THIRD_PARTY_NOTICES.md` 与 `REVIEW_ACCESS_TERMS.md`。
