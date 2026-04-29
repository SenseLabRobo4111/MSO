**🌐 [English](README.md) · [中文](README.zh.md)**

---

# MSO — Make Sense at Once

> **Make Sense at Once: Lightweight Map Prediction and Fusion for Efficient Multi-Robot Exploration**
> CoRL 2026 投稿。

本仓库存放 `sensemap` ROS 2 包：CoRL 2026 论文中使用的轻量化 342K 参数局部
地图预测器（FFC + GAN + 知识蒸馏）、训练流程，以及多机器人探索部署代码。

## 仓库结构

```
sensemap/
├── explore_model/
│   ├── SenseMapNet.py           # 学生 / 教师网络定义
│   ├── critic_model.py          # 对抗判别器
│   ├── ffc.py                   # Fast Fourier Convolution 模块
│   ├── dataset.py               # KTH / HouseExpo 训练数据集
│   ├── lightning_model.py       # PyTorch Lightning 包装
│   ├── main_unet_gan.py         # 训练入口
│   └── train_gan_new.py         # 知识蒸馏训练
└── predict_map.py               # ROS 2 推理节点
launch/                          # ROS 2 launch 文件
test/                            # ament 测试
package.xml, setup.py, setup.cfg # ROS 2 ament_python 包描述
```

LaTeX 论文源代码、绘图脚本、真机 rosbag 分析、结果 CSV 在另一处工作目录
（`corl_2026/`、`Exp/`），暂未同步到本仓库。Checkpoint 录用后再发布。

---

# Rebuttal / 修订待办清单

> 仅记录尚未完成的 rebuttal / 修订项；已完成的写在论文里，不在此重复。

标记说明:
- `🚧 BLOCKED` — 依赖此 simulator / 外部工作 / 决策
- `🟡 OPTIONAL` — 锦上添花，赶时间可跳
- `🟢 READY` — 现在就能做

## 高优先级

### A2. MRPB 多机扩展（N ∈ {2, 3, 5}） · `🚧 BLOCKED`

**只跑 full MSO**（`ours_multi_ours_orb`），在两个新 team 规模（N=3、N=5）
下跑。沿用论文 Sec 4.3 同 **10 个 MRPB 场景**，**每个 (scene, N) 跑 5 个
seed = 每个 N 50 runs**。

不重跑 frontier baseline 在 N=3/5 上：frontier-multi 不共享 predicted map，
纯靠几何覆盖扩展，N=3/5 曲线相比已有 N=2 基线没有额外信息。appendix 图把
full MSO 在 N=2/3/5 与 frontier-multi N=2（已有数据）对照即可。

总数：
- N=2：复用主论文 CSV，**不用重跑**（已在 `Exp/4.3/all_metrics/ours_multi_ours_orb.csv`）
- N=3：10 scenes × 5 seeds = **50 runs**（新跑）
- N=5：10 scenes × 5 seeds = **50 runs**（新跑，**审稿人主线诉求**）

**新工作量 100 runs**。如果 N=5 算力或通信带宽吃不消，退到 N ∈ {2, 3}（新
工作量减为 50 runs）；但 N=5 是审稿人明确点名的"5+ robot scaling"，强烈
建议保留。

**文件命名约定**。每个 (scene, seed, N) cell 输出一个 CSV，列必须是
`step, coverage, predicted_map_quality, topological_understanding`，路径：

```
Exp/4.3_v2/scaling/ours_multi_ours_orb_N<n>_scene<i>_seed<j>.csv
```

`<n>` ∈ `{3, 5}`，`<i>` ∈ `0..9`，`<j>` ∈ `0..4`。同 (scene, seed) 跨不同
N 的 run 必须从同一组 paired 起点出发（起点应由 (scene, seed) **确定性**
生成），保证曲线可比。

**验收**

- [ ] simulator 支持 `--robot_count={3, 5}`（如果没有就扩展）
- [ ] simulator 支持 `--seed=<int>` 且对起点采样、planner tie-breaking、
      sensor 噪声都生效（同一 seed 下跨不同 `--robot_count` 必须确定性
      一致）
- [ ] 烟雾测试：1 scene × 1 seed × N ∈ {3, 5} = 2 个 CSV 落到正确路径，
      然后再启动完整 100-run sweep
- [ ] 100 个预期 CSV 全部落到 `Exp/4.3_v2/scaling/...`（任一缺失就阻塞绘
      图——`plot_scaling_multirobot.py` 会检测到并报错）
- [ ] 完成后告诉我 **"A2 done"** —— 我会写
      `plot_scaling_multirobot.py`，渲染
      `figs/appendix/scaling_multirobot.png`（3 条 MSO N=2/3/5 曲线 + 1 条
      frontier N=2 基准），把 Appendix `app:scaling` 的占位换成真实图，并
      用实际 seed 数和 std 更新 caption

**审稿人钩子**。Reviewer #3 明确要求 3+ robot 结果，并把"scaling beyond
two agents"列为 open question。

---

### A3. KTH benchmark · `🚧 BLOCKED`

在 **5 个大型 KTH 场景**（训练 held-out，predictor 没见过具体布局但见过同
分布的 KTH 数据）上跑 **3 个 method**：`ours_multi_ours_orb`、
`nearest-multi-our-orb`、`ours_multi_ours_orb_nomerge`，每个 method × 5
seeds。两机器人 team 与主 MRPB 实验对齐。

总数：5 × 5 × 3 = **75 runs**。

**文件命名约定**：
```
Exp/4.3_v2/kth/<method>_scene<i>_seed<j>.csv
```
`<method>` ∈ `{ours_multi_ours_orb, nearest-multi-our-orb,
ours_multi_ours_orb_nomerge}`，`<i>` ∈ `0..4`，`<j>` ∈ `0..4`。

KTH 是 **in-distribution**（predictor 训练时见过 KTH 布局统计但没见过这些
具体测试场景）；appendix 已经写明，不要混作 held-out 强卖。

**验收**

- [ ] 选定 5 个具体 KTH 场景，列在 `Exp/4.3_v2/kth/SCENES.txt`（每行一个
      路径）
- [ ] 烟雾测试：1 scene × 1 seed × 3 methods → 3 个 CSV 落到正确路径
- [ ] 75 个预期 CSV 全部落到 `Exp/4.3_v2/kth/...`
- [ ] 完成后告诉我 **"A3 done"** —— 我会写 `plot_kth_curves.py`，渲染
      `figs/appendix/kth_curves.png`，替换 Appendix `app:kth` 占位，更新
      caption

**审稿人钩子**。把仿真 benchmark 从 1 个扩到 2 个，回应"3 scenes 太少"
的批评。

---

### A5. 单机真机部署 MapEx / IG-Hector / UPEN / MSO · `🚧 BLOCKED`

在 Sec 4.4 的同一物理平台（Jetson AGX Orin 32 GB、Livox Mid-360 LiDAR、
自建全向底盘）上，**单机器人**部署 4 个 baseline：MapEx、IG-Hector、UPEN、
MSO。在 **现有 3 个 arena 中选一个**（6.75 × 4.05 m），每个 method 跑 2 个
trial，共 **8 次部署**。每次部署同时录制 **第三视角视频**（无人机或手持）
和 **rosbag2**，必录话题见下表。

> **公平性约束 —— 跨方法起点严格对齐**
> 选定该 arena 上的两个起点 **pose A**（所有方法的 trial 1 用）和 **pose B**
> （所有方法的 trial 2 用）。**4 个方法的 trial 1 必须从同一个物理 pose A
> 出发，trial 2 必须从同一个物理 pose B 出发**，否则 coverage / 时间差异会
> 被审稿人质疑是起点运气。地面用胶带做标记，pose A 和 pose B 的 $(x, y, \theta)$
> 写到 `notes.md` 里；每个 trial 启动时实测的 $t=0$ pose 也单独记到该 trial
> 的 `start_pose.json`。

**4 个方法**

| Method | 来源仓库 / 模块 | Predicted-map 话题 |
|---|---|---|
| **MapEx** | upstream LaMa-based predictor (single-robot variant) | `/mapex/predicted_map_global` |
| **IG-Hector** | classical info-gain frontier (no learned predictor) | n/a |
| **UPEN** | uncertainty-based predictor | `/upen/predicted_map_global` |
| **MSO** | this repo, `predict_map.py` | `/mso/predicted_map_global` |

**录制目录结构**
```
Exp/realworld_single/
├── arena{1|2|3}/
│   ├── mapex_trial1/       (rosbag 目录 + video.mp4)
│   ├── mapex_trial2/
│   ├── ighector_trial1/
│   ├── ighector_trial2/
│   ├── upen_trial1/
│   ├── upen_trial2/
│   ├── mso_trial1/
│   └── mso_trial2/
└── notes.md                (pose A / pose B、每 trial 时长、异常等)
```

每个 `<method>_trial<i>/` 目录包含：
- 一个 rosbag2 store（`metadata.yaml` + `rosbag2_*.db3`）
- 一个 `video.mp4`，时间和 `metadata.yaml` 的 `starting_time` 对齐（视频
  第一帧做一个明显的"开始"手势方便对齐）
- 一个 `start_pose.json` 记录 $t=0$ 时机器人的 $(x, y, \theta)$

**rosbag 必录话题**

ROS 2 Humble，`sqlite3` 存储，单文件不切分（`--max-bag-duration 0`）。可以
写到 `record.launch.py` 里或直接 `ros2 bag record`：

| Topic | 类型 | 频率 | 用途 |
|---|---|---|---|
| `/odom`（或 `/robot_0/odom`）| `nav_msgs/msg/Odometry` | 50 Hz | 轨迹重建 |
| `/livox/lidar`（或原始 LiDAR）| `sensor_msgs/msg/PointCloud2` | ≥10 Hz | sensor 回归 / 重放 |
| `/map`（或 `/robot_0/map`）| `nav_msgs/msg/OccupancyGrid` | ≥1 Hz | 观测 occupancy 真值 |
| `<method>/predicted_map_global` | `nav_msgs/msg/OccupancyGrid` | ≥0.5 Hz | 仅 prediction 类方法（MapEx / UPEN / MSO） |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | ≥10 Hz | 控制器行为公平性 |
| `/way_point` 或 `/goal` | `geometry_msgs/msg/PoseStamped` | 事件驱动 | frontier / planner 输出 |
| `/tf`、`/tf_static` | `tf2_msgs/msg/TFMessage` | 高频 | 任何后处理对齐都要 |
| `/explored_volume`（如果有）| `std_msgs/msg/Float32` | ≥1 Hz | 覆盖率遥测 |
| `/explored_areas`（如果有）| `sensor_msgs/msg/PointCloud2` | ≥1 Hz | 覆盖率可视化 |
| `/rosout` | `rcl_interfaces/msg/Log` | 事件 | crash / warning 取证 |

各方法额外:
- **MSO**：`/mso/feature_packet`（稀疏 ORB 描述符，事件驱动），如果部署
  节点发布的话
- **MapEx**：`/mapex/uncertainty` 如果有
- **UPEN**：`/upen/uncertainty` 如果有

**终止条件**：每个 trial 跑到机器人报告覆盖率 ≥ 95% 或 180 s 上限，先到先
终止。把实际墙钟时长写到 `notes.md`。

**验收**

- [ ] `Exp/realworld_single/arena<X>/` 下 8 个目录全部建好（4 method × 2 trial）
- [ ] 每个目录包含：1 个 rosbag2 store、1 个 `video.mp4`、1 个
      `start_pose.json`
- [ ] **跨方法起点对齐**：4 个 `*_trial1` 目录的 `start_pose.json` 报告的
      $(x, y, \theta)$ 互相误差在 ±5 cm / ±3° 以内；4 个 `*_trial2` 同样
- [ ] 烟雾测试任选 1 个 trial 用 `py -3 corl_2026/measure_rosbag_bandwidth.py`
      解析 rosbag，确认上表所有必录话题 `count > 0`
- [ ] `notes.md` 记录 pose A、pose B、每 trial 时长、最终覆盖率、异常
- [ ] 完成后告诉我 **"A5 done"** —— 我会：
  1. 扩展 `measure_rosbag_bandwidth.py` 支持新目录布局
  2. 在 Sec 4.4（或 Appendix `app:single-baselines`）加单机真机对比段落，
     展示 4 个方法的 coverage / 时间 / 最终地图准确率柱状图
  3. 把每个方法视频里挑一帧代表性截图嵌进图里

**审稿人钩子**。直接回应 Reviewer #2 "real-world 只对 frontier baseline，
太弱"——把 MapEx / IG-Hector / UPEN 拿到同一台 Jetson Orin 上跑，是最强的
deployment-parity 论据。

---

## 中等优先级

### A4. MSO 仅融合变体 · `🚧 BLOCKED + 🟡 OPTIONAL`

关掉 predictor 用 observed-only 地图喂 fusion，跑 N=2 × 10 scenes × 5 seeds
= 50 runs（与 A2 同网格）。需要 simulator 支持 `--predictor=none`（或等价
的构造参数绕过 predictor 把 observed map 直接送 fusion）。改动若超过 1 小时
就跳过，标 future work，不能阻塞发版。

**输出路径**：
```
Exp/4.3_v2/ablation/ours_multi_fusion_only_scene<i>_seed<j>.csv
```

**验收**

- [ ] simulator 接好 `--predictor=none`（或等价方式）让 fusion 吃 observed-
      only occupancy
- [ ] 50 个 CSV 落到上述路径
- [ ] 或明确告知"infeasible"，我会在 Limitations 段加一行说明

---

### P4-2. 拓扑 / 墙连通性指标 · `🟢 READY 🟡 OPTIONAL`

对 held-out KTH/HouseExpo 测试集（1,356 样本）的每张 predicted map 算
三个指标：

1. **骨架化 IoU** —— Zhang–Suen 细化 obstacle 通道，与 ground-truth 骨架
   求 IoU。
2. **连通分量差** —— prediction 的 obstacle CC 数减去 GT 的 CC 数（带
   符号；越接近 0 越好）。
3. **门洞保留率** —— GT 中 1-pixel 宽门洞像素，在 prediction 中保持可通行
   的比例。

作为 PSNR/SSIM/LPIPS/FID/KID 的补充加进 `tab:quantitative-results(a)` 三列。
per-sample dump 到 `Exp/4.1_topology/<method>_topology.csv`。

**我需要你确认**：predictor checkpoint 和测试集 tensor 本地能否取到（或者
告诉我位置），然后告诉我可以开工，我写脚本（约 1-2 小时）。

**验收**

- [ ] 确认 predictor checkpoint 路径
- [ ] 确认 KTH/HouseExpo 测试集 tensor 位置
- [ ] 让我开始写指标脚本
- [ ] Tab 2(a) 多出三列且不破坏 `\arraystretch` / 页面排版

**审稿人钩子**。Reviewer #6 指出 FID/KID 不直接反映下游 robot 行为。

---

## 低优先级

### P4-4. Rebuttal 回应信 · `🚧 BLOCKED on phase decision`

对原始 review 中 7 条具体审稿人问题逐条回应（fusion 成功率、hallucination
敏感性、predicted free-space 安全性、通信带宽、robot 规模、MapEx/IG-Hector
真机缺席、真机统计报告）。**仅** rebuttal 阶段需要，camera-ready 修订就不写。

**验收**

- [ ] 确认阶段：rebuttal 还是 camera-ready
- [ ] 如果是 rebuttal，把审稿人原文给我，我按 CoRL 字数限制（一般每 reviewer
      5,000 字符，meta 2,000 字符）起草

---

## 提交前自检

- [ ] `pdflatex example.tex && bibtex example && pdflatex × 2` 没有错误，
      没有 `Citation undefined` warning
- [ ] 主文 ≤ 8 页（Sec 1 → Sec 6 Conclusion）
- [ ] A2/A3 数据落地后，`app:scaling` 和 `app:kth` 的占位图都换成真实
      `\includegraphics`
- [ ] `fig:exploration-overview` 的 caption 写明"averaged over 50 runs per
      method"
- [ ] `tab:ros-topics-measured` 的数字与最新 rosbag 实测一致（重跑
      `py -3 measure_rosbag_bandwidth.py` 验证）
- [ ] PDF 里没有残留的红色 `\textcolor{red}{\textbf{[TODO ...]}}` 标记

---

# 已完成

<details>
<summary>点击展开</summary>

### Phase 0 —— 纯写作修正
- 全文 `robust → improved`（fusion 主张降级）
- `without global alignment → without pre-known global relative poses`
- Sec 1 contributions 重复 `Finally` 去重
- Tab 2(a) 加 MACs/FLOPs 定义注脚
- "MSO predictor" vs "MSO system" 全文消歧
- Sec 4.2 Results 重排：per-pair Dice mean (0.510 → 0.588, +15.3%) 升为
  主指标，threshold-based SR 改作辅助
- Sec 1 contributions 改写为 "local structural prior + uncertainty-aware
  decentralized fusion" 统一原则

### Phase 1 —— Fusion 论证
- 新增 `app:fusion-dice`（4 点分析：高 threshold / Dice 分布上移 / matcher
  上限可达 / 端到端是操作指标）
- 新增 `app:fusion-failures`（3 类 failure mode：hallucinated rooms / 低
  overlap / 对称 layout）

### Phase 2A —— 50 seed 标注
- per-seed CSV 数据丢失；caption + appendix 诚实标注 "averaged over 50 runs
  per method"，per-seed std ≈ ±0.02–0.05

### Phase 4 —— 其他
- 真机 10 个 rosbag 实测 ROS 2 带宽 → appendix `tab:ros-topics-measured`
- 8 页限制核实（Sec 1 → Sec 6 正好填到 1–8 页）
- 曲线美化：统一 alpha=0.78、round caps、字号上调
- `app:scaling` 和 `app:kth` 已加 appendix 脚手架等 A2/A3 数据
- `plot_exploration_csv_results.py` 升级为读 per-seed CSV + 渲染 mean ± std
  band（当前走 fallback 单 CSV 路径）

</details>

---

# 引用

```bibtex
@inproceedings{mso2026,
  title={Make Sense at Once: Lightweight Map Prediction and Fusion for Efficient Multi-Robot Exploration},
  author={Anonymous},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2026}
}
```
