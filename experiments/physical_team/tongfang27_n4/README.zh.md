# 同方 27 楼四台 SenseBeetle 实机实验

[English full protocol](README.md)

## 分支与入口

本实验唯一的规范实机分支是：

```text
sensebeetle-n4-tongfang27
```

实机准备审查、预实验记录、正式 N=4 运行提交和后续审计修正都必须从该
分支开始：

```bash
git fetch origin
git switch --track origin/sensebeetle-n4-tongfang27
```

当前立即任务标识为 `tf27_n4_mso_three_run_v1`。`main` 是论文与通用
软件评审分支；论文汇总、历史准备和已经合并的模型修正分支都不是本次
实机运行的权威入口。

## 当前状态与证据边界

当前仓库只提供前瞻性协议和采集前门禁，没有同方 27 楼 N=4 实机结果：

- `collection_ready: false`
- `results_status: not_collected`
- `claim_authorized: false`

单元测试、合成输入、离线重放、历史双机 bag 和模型前向验证都不能替代
本次四车实机数据。在线完整链、GT/独立位姿参考、全 topic 录包、四路
视频、同步验证、集成演练和逐次运行审计未全部通过前，不得启动正式采集。

## 立即执行的三次 MSO 自主探索

四台 SenseBeetle（`robot_0` 至 `robot_3`）同时在冻结的同方 27 楼办公室
可达 ROI 内自主探索。四台车必须严格加载并实际使用
`mso_deconv_342771_candidate_a`，网络为 nominal；不能用 shadow-only 或
`observed_only` 替代。T0 同步放行后禁止人工导航，操作员只允许急停。

三个预注册运行 ID 为：

1. `tf27_n4_mso_repeat_01`；
2. `tf27_n4_mso_repeat_02`；
3. `tf27_n4_mso_repeat_03`。

每次先预录 30 s，T0 后正式窗口固定 600 s，再后录 10 s。若 frontier
提前耗尽，四车安全驻车但仍录满 600 s；覆盖不足不得延长或择优重跑。
“把 27 楼自主探索出来”是任务目标，正式结果以冻结 GT 可达掩膜上的
measured-only coverage、0.8/0.9 覆盖时间和完整轨迹报告，预测格不计覆盖。

### 每次必须录制 5 份全 topic rosbag

- `robot_0`、`robot_1`、`robot_2`、`robot_3` 各在本机录一份 all-topic bag；
- coordinator 再录一份覆盖完整分布式图的 all-topic bag；
- 必须包含 hidden topic 和运行中晚出现的 topic；
- T0 前至少 30 s 启动，T=600 s 后等待静止、end marker、flush 和 10 s
  后录再停止；
- 保存 pre/mid/post topic inventory、topic 类型与 QoS、每 topic 消息数和
  首末时间、bag metadata、实际命令、退出码、文件大小与 SHA-256。

现场执行方必须自行提供并预检这 5 个 record-all 进程，包含 hidden 和晚
出现 topic。本仓库现有 recorder 是 topic 白名单审计工具，单独使用并不
满足“录制所有 topic”的要求。任何一份 bag 缺失、损坏或时间窗不完整，
均判 technical-invalid。

### 每台车必须有独立第三人称视频

每次同时生成四份连续原始视频：`robot_0_third_person` 至
`robot_3_third_person`，一车一份。单一 overhead 全景不能代替四份逐车
视角。每份记录 run ID、robot ID、camera ID/serial、分辨率、帧率、时间
基准、首末帧、掉帧统计和 SHA-256；T0 前至少 30 s 开始，T=600 s 后至少
10 s 结束，全程不得暂停或剪接。相机与操作员不得进入 geofence。

### ROS、视频和 GT 必须统一时间轴

四车计算机、coordinator、GT/reference 系统和四路视频统一到同一
UTC/PTP 或已验证 NTP 基准；`use_sim_time=false`，不得出现 `/clock`。
主机绝对 offset 必须不超过 5 ms，RTT 不超过 20 ms，GT join 不超过
100 ms，并保存 run 前、运行中和 run 后的时钟采样。

在 T=0、300、600 s 向所有可见 bag 写 `/experiment/sync_marker`，同时让
同一个物理 LED/闪光被四路视频拍到。视频对 ROS 的残差不得超过 1 帧
（30 fps 时为 33.3 ms）。marker 缺失、无法解释的漂移、时钟跳变/回拨或
任一阈值超限，均判 technical-invalid。

### 通过百度网盘向实验负责人交付

每次运行封存并生成完整 SHA-256 清单后，必须把 5 份原始 bag 和 4 份逐车
第三人称原始视频交付给实验负责人；可以使用私密、受限访问的百度网盘
（百度云盘）目录作为交付渠道。
三个计划 ID、所有 technical-invalid 运行和所有 replacement 都必须交付；
不得因为运行失败、覆盖率低或结果不理想而漏传。

目录至少按 campaign/run 明确分层，例如：

```text
tf27_n4_mso_three_run_v1/
  tf27_n4_mso_repeat_01/
    bags/{robot_0,robot_1,robot_2,robot_3,coordinator}/
    videos/{robot_0_third_person,robot_1_third_person,
            robot_2_third_person,robot_3_third_person}/
    manifests/{run_manifest,topic_inventory,video_inventory,SHA256SUMS}
```

允许无损压缩或分卷上传，但必须保留原始文件，并同时提供原文件哈希和上传
分卷哈希。百度网盘分享链接、提取/访问方式、有效期、上传人、完成时间、
总字节数和顶层 SHA-256 清单，应通过私密渠道发给实验负责人；不得把任何
百度网盘分享 URL、提取/访问信息或凭据提交到公开仓库。另行批准的公开发布
必须使用脱敏数据包和不同的公开链接。只有负责人确认可以访问并完成清单
核验后，才算交付完成；确认前必须保留第二份本地副本。百度网盘
只是传输渠道，不能代替原始 bag、视频、manifest 和哈希审计。

### 三次运行之间必须冷重置

每次先封存并哈希全部证据，再清空四车局部/合并/持久地图、MSO cache、
registration graph、frontier/planner history 和 DDS 暂态状态；重启运行
进程，清除所有网络故障规则并证明 nominal，恢复冻结的门/家具/排除区，
按预注册起点复位。随后重新核对模型哈希与 strict-load、软件提交、时钟、
存储、all-topic discovery、四路视频和急停。严禁加载上一 run 的任何状态。

错模型、人工导航、未冷重置、起点/环境错误、关键 bag/video/topic/sync/GT
缺失属于 protocol/technical invalid。原 ID 和全部证据必须永久保留；
replacement 使用新 ID（例如 `_r1`）并填写 `replacement_for`。低覆盖、
MSO 效果差、frontier 耗尽或证据完整的安全停止是结果，不得无痕替换。

本 README 和 `EVIDENCE_CHECKLIST.md` 是对现场执行方发布的需求。本次交付
只提供已核验模型和中英文需求，不替现场团队编写录包、视频或时间同步
程序。执行方必须在现场前把这些条款落实为自己的 run manifest、实际
命令、时钟报告和逐次验收审计；现有白名单 recorder 或单路全景视频不能
作为“已经满足需求”的证据。

## 后续可选的 32-run 因子扩展

既有 `config/campaign.yaml` 保留为后续 2 × 2 确认性扩展：8 个起点区组，
MSO/observed-only × nominal/impaired，共 32 次。它不是当前三次实机任务，
两套数据不得混合，三次结果也不能事后塞入 32-run 单元。

## 模型锁

本次实验模型为完整 FFC student：

- model ID：`mso_deconv_342771_candidate_a`
- 交付文件：`repro_reconstructed/checkpoints/recovered_candidate_deconv_a.pt`
- 参数量：342,771
- FFC blocks：4
- artifact SHA-256：
  `da4458514656d41fba0e0ce6d4f4967997ff0a97f2e905a458757609edf3a3a8`
- loader：`distill_map_net_deconv_raw_state_v1`

候选 A 来自部署副本中保留的 `model-epoch-deconv.ckpt`，由实验负责人
前瞻性指定为本次 full-FFC 目标。历史在线入口加载的是另一个 bilinear
checkpoint，因此这里不声称候选 A 曾被历史在线系统使用，也不把它冒充
论文 checkpoint。模型锁、固定输入、五个输出的形状/有限性/指纹和来源
边界见 `config/model_lock.yaml` 与 `MODEL_ARTIFACT_AUDIT.md`。

## 采集前闸门

以下项目必须全部形成可校验的路径、提交号、哈希和报告：

1. 四车完整在线链：predictor、registration、gate、commit、persistent map、planner；
2. GT 占据图、有效 ROI、动态排除掩膜及其元数据；
3. 独立位姿参考、标定、机器人外参和不确定度；
4. 四车 nominal 网络配置、时钟采样和故障规则清理证据；
5. 四台机器人硬件清单、软件提交、运动与安全策略；
6. 不计入三次正式运行的全链集成演练通过且无关键安全故障；
7. 逐次 auditor 能核对 600 s 窗口、5 bags、4 videos、同步 marker、
   GT trace、nominal 网络和模型绑定。

当前仓库没有针对这三次任务的现场自动化或放行程序。在线栈、GT/参考、
五路录包、四路视频、同步与逐次审计没有由执行方独立验证前，状态必须
保持 `collection_ready: false`。

## 执行顺序

1. 填写机器人、场地、GT/参考、网络和在线栈锁文件；
2. 在四台机器人上核对同一软件提交、模型 SHA 和运行环境；
3. 完成静止与低速 dry capture，确认时钟、TF、topic 和安全接口；
4. 完成不计入三次正式运行的全链集成演练并独立审计；
5. 由独立审查人逐项复核 run manifest、5 份全 topic bag 方案、4 路视频
   方案、同步报告模板和逐次验收清单；既有 32-run validator 不适用于本任务；
6. 只有所有现场门禁经书面复核通过后，才按 01、02、03 顺序完成三次实验；
7. 每次结束立即冻结 bag、事件日志、GT/reference trace、网络证据和哈希；
8. 三次全部完成后一次性汇总，不得按中间结果提前停止或择优重跑。

## 公开数据边界

公共分支只提交协议、模板、模型身份记录和不敏感的验证材料。办公室原始
GT 图、现场照片、机器人序列号、主机地址、原始 bag 和外部跟踪记录应放在
受控证据存储中；原始 bag 和视频必须私密交付给实验负责人，可以使用上面
定义的受限百度网盘渠道。仓库只记录可审计的相对标识与 SHA-256，不记录
任何百度网盘分享 URL、提取/访问信息或凭据。当前分支没有 N=4 结果图、
视频或性能结论。

## 权威文件索引

- [英文完整协议](README.md)
- 本页“立即执行的三次 MSO 自主探索”及录包/视频/同步条款
- [后续 32-run 可选扩展](config/campaign.yaml)
- [模型锁](config/model_lock.yaml)
- [在线栈锁](config/online_stack_lock.yaml)
- [后续 32-run 的可选预实验计划](config/pilot_plan.yaml)
- [证据检查表](EVIDENCE_CHECKLIST.md)
- [模型来源与选择边界](MODEL_ARTIFACT_AUDIT.md)
