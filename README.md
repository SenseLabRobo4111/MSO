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

### A2. 3-robot scaling sweep · `🚧 BLOCKED`
**EN.** Re-run the MRPB simulator with `robot_count=3` for `ours_multi_ours_orb`
and `nearest-multi-our-orb` over 10 MRPB scenes × 5 random seeds (= 50 runs/method).
Drop per-(scene, seed) CSVs into `Exp/4.3_v2/all_metrics/<method>_3robot_seed<i>.csv`.
Once data lands, the figure goes into Appendix `app:scaling`.

**CN.** 用 `robot_count=3` 重跑 MRPB simulator，`ours_multi_ours_orb` 和
`nearest-multi-our-orb` 两个 method × 10 scenes × 5 seeds = 50 runs/method。
per-seed CSV 命名 `<method>_3robot_seed<i>.csv`，丢到 `Exp/4.3_v2/all_metrics/`。
数据到位后绘图填进 Appendix `app:scaling`。

- [ ] Configure simulator with `N=3` robots
- [ ] Run 50 × 2 = 100 simulations
- [ ] Drop per-seed CSVs into the canonical location
- **Reviewer hook.** Reviewer #3 explicitly asked for 3+ robot scaling.

### A3. KTH benchmark sweep · `🚧 BLOCKED`
**EN.** Run multi-robot exploration on 5 large KTH scenes × 5 seeds for
`ours_multi_ours_orb`, `nearest-multi-our-orb`, and
`ours_multi_ours_orb_nomerge` (= 75 runs total). Output to
`Exp/4.3_v2/kth/<method>_scene<i>_seed<j>.csv`. KTH is **in-distribution**
(predictor saw KTH layout statistics during training, never these specific
scenes); frame the appendix accordingly.

**CN.** 在 5 个大型 KTH 场景 × 5 seeds 跑 `ours_multi_ours_orb` /
`nearest-multi-our-orb` / `ours_multi_ours_orb_nomerge` 共 75 runs，输出到
`Exp/4.3_v2/kth/<method>_scene<i>_seed<j>.csv`。KTH 是 **in-distribution**，
appendix 措辞要明确，不要混作 held-out。

- [ ] Pick 5 large held-out KTH scenes (predictor never saw these)
- [ ] Run 75 simulations
- [ ] Drop per-seed CSVs
- **Reviewer hook.** Answers "3 scenes is too few" critique.

## Medium priority / 中等优先级

### A4. MSO fusion-only ablation · `🚧 BLOCKED + 🟡 OPTIONAL`
**EN.** Run MSO with the predictor disabled (use observed-only maps as fusion
input) on the same 50 runs/method as A2. Requires simulator support for a
`--predictor=none` flag. If infeasible, mark as future work — do not block
release on this.

**CN.** 关掉 predictor 用 observed-only 地图喂 fusion，跑同样 50 runs/method。
需要 simulator 支持 `--predictor=none` 开关。做不了就跳过，标 future work。

- [ ] Add `--predictor=none` flag to simulator (if not present)
- [ ] Run 50 simulations OR confirm "infeasible"

### P4-2. Topology / wall-connectivity metric · `🟢 READY 🟡 OPTIONAL`
**EN.** Post-processing script for predicted maps: skeletonisation →
connected-component count → doorway-preservation rate. Add a new column to
`tab:quantitative-results(a)` to complement PSNR/SSIM/LPIPS/FID/KID.

**CN.** 给 predicted map 写后处理脚本：骨架化 → 连通分量计数 → 门洞保留率，
补到 `tab:quantitative-results(a)` 一列。

- [ ] Confirm predictor checkpoint is locally accessible
- [ ] Write metric script (~1–2 hours)
- **Reviewer hook.** Reviewer #6 noted FID/KID don't directly reflect
  downstream robot behaviour.

## Low priority / 低优先级

### P4-4. Rebuttal letter · `🚧 BLOCKED on phase decision`
**EN.** Bullet-by-bullet response to the seven concrete reviewer questions.
Only needed if currently in the **rebuttal** phase.

**CN.** 对审稿人 7 条具体问题逐条回应。**仅** rebuttal 阶段需要。

- [ ] Confirm: rebuttal phase or camera-ready revision?

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
