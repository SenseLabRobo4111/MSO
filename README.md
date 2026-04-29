# MSO — Make Sense at Once

> **Make Sense at Once: Lightweight Map Prediction and Fusion for Efficient Multi-Robot Exploration**
> CoRL 2026 submission.

This repository hosts the `sensemap` ROS 2 package: the lightweight 342K-parameter
local map predictor (FFC + GAN + knowledge distillation), training pipeline, and
the multi-robot exploration deployment glue used in the CoRL 2026 paper.

本仓库存放 `sensemap` ROS 2 包：CoRL 2026 论文中使用的轻量化 342K 参数局部地图预测器
（FFC + GAN + 知识蒸馏）、训练流程，以及多机器人探索部署代码。

## Contents / 仓库结构

```
sensemap/
├── explore_model/
│   ├── SenseMapNet.py           # Student / teacher network definition
│   ├── critic_model.py          # Adversarial discriminator
│   ├── ffc.py                   # Fast Fourier Convolution block
│   ├── dataset.py               # KTH / HouseExpo training dataset
│   ├── lightning_model.py       # PyTorch Lightning wrapper
│   ├── main_unet_gan.py         # Training entry point
│   └── train_gan_new.py         # Knowledge-distillation training
└── predict_map.py               # ROS 2 inference node
launch/                          # ROS 2 launch files
test/                            # ament tests
package.xml, setup.py, setup.cfg # ROS 2 ament_python package descriptors
```

The LaTeX paper source, plotting scripts, real-world rosbag analysis, and the
results CSVs live in a sibling working tree (`corl_2026/`, `Exp/`); they are not
mirrored here yet. The trained checkpoints will be released upon acceptance.

LaTeX 论文源代码、绘图脚本、真机 rosbag 分析、结果 CSV 在另一处工作目录
（`corl_2026/`、`Exp/`）；checkpoint 录用后再发布。

---

# Rebuttal / Revision TODO / 待办清单

> Tracks outstanding items for the CoRL 2026 rebuttal/revision cycle.
> Done items are recorded in the LaTeX source and not duplicated here.
>
> 仅记录尚未完成的 rebuttal / 修订项；已完成的写在论文里。

Legend / 标记说明:
- `🚧 BLOCKED` — depends on this simulator / external work / decision
  依赖此 simulator / 外部工作 / 决策
- `🟡 OPTIONAL` — nice to have / 锦上添花
- `🟢 READY` — can be executed now / 现在就能做

## High priority / 高优先级

### A2. Multi-robot scaling on MRPB (N ∈ {2, 3, 5}) · `🚧 BLOCKED`

**EN — what to do.**
Run **full MSO only** (`ours_multi_ours_orb`) at two new team sizes (N=3
and N=5). Use the **same 10 MRPB scenes** as Sec 4.3 of the paper, **5 random
seeds per (scene, N) cell** = **50 runs per N**.

We do not rerun the frontier baseline at N=3/5: frontier-multi does not share
predicted maps across robots and scales by pure geometric coverage only, so
its N=3/5 curves add little signal beyond the existing N=2 reference. The
appendix figure plots full MSO at N=2/3/5 against the frontier-multi N=2
baseline already in `Exp/4.3/all_metrics/`.

Total runs:
- N=2: existing main-paper data, **no rerun** (50 runs full MSO already in
  `Exp/4.3/all_metrics/ours_multi_ours_orb.csv`)
- N=3: 10 scenes × 5 seeds = **50 runs** (new)
- N=5: 10 scenes × 5 seeds = **50 runs** (new — headline reviewer ask)

**New work total: 100 runs.** If N=5 is not feasible (compute or
communication-bandwidth limit), drop to N ∈ {2, 3} (50 new runs) — but N=5
is strongly preferred because the reviewer asked specifically for
"5+ robot scaling".

**EN — file naming convention.** Each (scene, seed, N) cell produces one
CSV with columns
`step, coverage, predicted_map_quality, topological_understanding`. Place at:

```
Exp/4.3_v2/scaling/ours_multi_ours_orb_N<n>_scene<i>_seed<j>.csv
```

`<n>` ∈ `{3, 5}`, `<i>` ∈ `0..9`, `<j>` ∈ `0..4`. Robots start from a
paired-pose configuration that is **deterministic in (scene, seed)** so
cross-N curves remain directly comparable for a fixed (scene, seed).

**CN — 要做什么.**
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

**新工作量 100 runs**。N=5 算力或通信带宽吃不消可以退到 N ∈ {2, 3}（新工作量
减为 50 runs）；但 N=5 是审稿人明确点名的"5+ robot scaling"，强烈建议保留。

**CN — 文件命名约定.** 每个 (scene, seed, N) cell 输出一个 CSV，列必须是
`step, coverage, predicted_map_quality, topological_understanding`，路径：

```
Exp/4.3_v2/scaling/ours_multi_ours_orb_N<n>_scene<i>_seed<j>.csv
```

同 (scene, seed) 跨不同 N 的 run 必须从同一组 paired 起点出发（起点应由
(scene, seed) **确定性**生成），保证曲线可比。

**Acceptance / 验收**

- [ ] Simulator accepts `--robot_count={3, 5}` (extend if missing)
- [ ] Simulator accepts `--seed=<int>` and respects it for start-pose
      sampling, planner tie-breaking, sensor noise (deterministic across
      `--robot_count` for the same seed)
- [ ] Smoke test: 1 scene × 1 seed × N ∈ {3, 5} = 2 CSVs land at the
      expected paths before kicking off the full 100-run sweep
- [ ] All 100 expected CSVs land at `Exp/4.3_v2/scaling/...` (any missing
      run blocks plotting — `plot_scaling_multirobot.py` will detect and
      bail)
- [ ] Ping me with **"A2 done"** — I will write
      `plot_scaling_multirobot.py`, render
      `figs/appendix/scaling_multirobot.png` (3 MSO curves at N=2/3/5 plus
      a frontier N=2 reference), replace the placeholder in Appendix
      `app:scaling`, and update the caption with the actual seed count and
      observed std

**Reviewer hook.** Reviewer #3 explicitly asked for 3+ robot results and
flagged "scaling beyond two agents" as an open question.

---

### A3. KTH benchmark sweep · `🚧 BLOCKED`

**EN — what to do.**
Run multi-robot exploration on **5 large KTH scenes** (held out — predictor
never saw these specific layouts, but did see KTH-like statistics during
training) for **3 methods**: `ours_multi_ours_orb`, `nearest-multi-our-orb`,
`ours_multi_ours_orb_nomerge`, **5 seeds each**. Two-robot teams to match the
main MRPB benchmark.

Total runs: 5 scenes × 5 seeds × 3 methods = **75 runs**.

**EN — file naming convention.**
```
Exp/4.3_v2/kth/<method>_scene<i>_seed<j>.csv
```
`<method>` ∈ `{ours_multi_ours_orb, nearest-multi-our-orb, ours_multi_ours_orb_nomerge}`,
`<i>` ∈ `0..4`, `<j>` ∈ `0..4`.

KTH is **in-distribution** (predictor saw KTH layout statistics in training
but never these specific test scenes); the appendix already states this so
it cannot be mistaken for held-out generalisation.

**CN — 要做什么.**
在 **5 个大型 KTH 场景**（训练 held-out，predictor 没见过具体布局但见过同
分布的 KTH 数据）上跑 **3 个 method**：`ours_multi_ours_orb`、
`nearest-multi-our-orb`、`ours_multi_ours_orb_nomerge`，每个 method × 5 seeds。
两机器人 team，与主 MRPB 实验对齐。

总数：5 × 5 × 3 = **75 runs**。

**CN — 文件命名约定.**
```
Exp/4.3_v2/kth/<method>_scene<i>_seed<j>.csv
```
KTH 是 **in-distribution**，appendix 已经写明，不要误称 held-out。

**Acceptance / 验收**

- [ ] 5 specific KTH scenes selected and listed in
      `Exp/4.3_v2/kth/SCENES.txt` (one path per line)
- [ ] Smoke test: 1 scene × 1 seed × 3 methods → 3 CSVs land at expected
      paths before full sweep
- [ ] All 75 expected CSVs land at `Exp/4.3_v2/kth/...`
- [ ] Ping me with **"A3 done"** — I will write `plot_kth_curves.py`, render
      `figs/appendix/kth_curves.png`, replace the placeholder in Appendix
      `app:kth`, update the caption.

**Reviewer hook.** Adds a second simulated benchmark; answers
"3 scenes is too few".

---

## Medium priority / 中等优先级

### A4. MSO fusion-only ablation · `🚧 BLOCKED + 🟡 OPTIONAL`

**EN — what to do.**
Run MSO with the predictor disabled (use observed-only maps as fusion input)
on the same N=2 × 10 scenes × 5 seeds = 50 runs grid as A2. Requires
simulator support for a `--predictor=none` flag (or equivalent constructor
argument that bypasses the predictor and feeds the observed map straight to
fusion). If implementing the flag is more than ~1 hour of simulator work,
skip and mark as future work — do not block release.

**CN — 要做什么.**
关掉 predictor 用 observed-only 地图喂 fusion，跑 N=2 × 10 scenes × 5 seeds
= 50 runs（与 A2 同网格）。需要 simulator 支持 `--predictor=none`。改动若超过
1 小时就跳过，标 future work，不能阻塞发版。

**Output path / 输出路径.**
```
Exp/4.3_v2/ablation/ours_multi_fusion_only_scene<i>_seed<j>.csv
```

**Acceptance / 验收**

- [ ] Simulator wires `--predictor=none` (or equivalent) so fusion consumes
      observed-only occupancy as input
- [ ] 50 CSVs land at the path above
- [ ] OR explicit "infeasible" confirmation — I will add a one-line note in
      the Limitations section

---

### P4-2. Topology / wall-connectivity metric · `🟢 READY 🟡 OPTIONAL`

**EN — what to do.**
Write a post-processing script that, for every predicted map in the held-out
KTH/HouseExpo test split (1,356 samples), computes:

1. **Skeletonisation IoU** — Zhang–Suen thinning of the obstacle channel,
   compared to ground-truth skeleton via IoU.
2. **Connected-component delta** — number of obstacle CCs in prediction
   minus number in ground truth (signed; closer to 0 is better).
3. **Doorway preservation rate** — fraction of ground-truth narrow gaps
   (1-pixel doorways) that remain traversable in the prediction.

Add three new columns to `tab:quantitative-results(a)` next to FID/KID.
Per-sample dump to `Exp/4.1_topology/<method>_topology.csv`.

**EN — what I need from you.**
Confirm the predictor checkpoint and the test-split tensors are locally
accessible (or tell me where they live), then ping me — I will write the
script (~1–2 hours).

**CN — 要做什么.**
对 held-out KTH/HouseExpo 测试集（1,356 样本）的每张 predicted map 算三个
指标：(1) 骨架化 IoU；(2) 连通分量差；(3) 门洞保留率。作为
PSNR/SSIM/LPIPS/FID/KID 的补充，加进 `tab:quantitative-results(a)` 三列。

**Acceptance / 验收**

- [ ] Confirm predictor checkpoint path
- [ ] Confirm KTH/HouseExpo test-split tensor location
- [ ] Ping me to write the metric script
- [ ] Three new columns appear in Tab 2(a) without breaking the table's
      `\arraystretch` / page geometry

**Reviewer hook.** Reviewer #6 noted FID/KID don't directly reflect
downstream robot behaviour.

---

## Low priority / 低优先级

### P4-4. Rebuttal letter · `🚧 BLOCKED on phase decision`

**EN — what to do.**
Bullet-by-bullet response to the seven concrete reviewer questions
collected in the original review (fusion success rate, hallucination
sensitivity, predicted-free-space safety, communication payload, robot
scaling, MapEx/IG-Hector real-world omission, real-world stat reporting).
Only needed if currently in the **rebuttal** phase, not in the camera-ready
revision.

**CN — 要做什么.**
对审稿人原始 review 中 7 条具体问题逐条回应（fusion 成功率、hallucination
敏感性、predicted free-space 安全性、通信带宽、robot 规模、MapEx/IG-Hector
真机缺席、真机统计报告）。**仅** rebuttal 阶段需要。

**Acceptance / 验收**

- [ ] Confirm phase: rebuttal vs camera-ready
- [ ] If rebuttal, ping me with the exact reviewer text — I will draft a
      response letter targeting CoRL's word limit (typically 5,000 chars
      per reviewer, 2,000 char meta)

---

## Sanity checks before submission / 提交前自检

- [ ] `pdflatex example.tex && bibtex example && pdflatex × 2` no errors,
      no `Citation undefined` warnings
- [ ] Main text body still ≤ 8 pages (Sec 1 → Sec 6 Conclusion)
- [ ] Both `app:scaling` and `app:kth` placeholder figures replaced with
      real `\includegraphics` once A2/A3 land
- [ ] Caption of `fig:exploration-overview` says "averaged over 50 runs per
      method"
- [ ] `tab:ros-topics-measured` numbers match latest rosbag re-measurement
      (run `py -3 measure_rosbag_bandwidth.py` to verify)
- [ ] No leftover `\textcolor{red}{\textbf{[TODO ...]}}` markers in the
      compiled PDF

---

# Already completed / 已完成

<details>
<summary>Click to expand / 点击展开</summary>

### Phase 0 — Writing-only fixes / 纯写作修正
- `robust → improved` for fusion claims throughout.
- `without global alignment → without pre-known global relative poses`.
- De-duplicate `Finally` in Sec 1 contributions.
- Add MACs/FLOPs definition footnote in Tab 2(a).
- Disambiguate "MSO predictor" vs "MSO system".
- Reorder Sec 4.2 Results: per-pair Dice mean (0.510 → 0.588, +15.3%) as primary metric, threshold-based SR as secondary.
- Rewrite Sec 1 contributions to lead with "local structural prior + uncertainty-aware decentralized fusion" unified principle.

### Phase 1 — Fusion narrative / Fusion 论证
- New `app:fusion-dice` (4-point analysis: high threshold / Dice distribution shift / matcher headroom / end-to-end is operational).
- New `app:fusion-failures` (3 failure modes: hallucinated rooms / low overlap / symmetric layouts).

### Phase 2A — Per-seed annotation / 50 seed 标注
- Per-seed CSV data was lost; honest "averaged over 50 runs per method" labelling in caption + appendix; per-seed std ≈ ±0.02–0.05.

### Phase 4 — Misc / 其他
- Empirical ROS 2 bandwidth measured from 10 real-world rosbags → `tab:ros-topics-measured` in appendix.
- 8-page main-text limit verified (Sec 1 → Sec 6 fits exactly on pages 1–8).
- Curve aesthetics: uniform alpha=0.78, round line caps, increased font sizes.
- Empty appendix scaffolding for `app:scaling` and `app:kth` ready to receive A2/A3 results.
- `plot_exploration_csv_results.py` upgraded to read per-seed CSVs and render mean ± std bands (currently fallback path active).

</details>

---

# Citation / 引用

```bibtex
@inproceedings{mso2026,
  title={Make Sense at Once: Lightweight Map Prediction and Fusion for Efficient Multi-Robot Exploration},
  author={Anonymous},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2026}
}
```
