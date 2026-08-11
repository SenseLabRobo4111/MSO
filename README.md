# MSO

**Design and audit of resource-constrained predictive multirobot exploration**

[中文说明](README.zh.md)

MSO (Make Sense at Once) is a research system for two-dimensional indoor
multirobot exploration. It combines a compact local occupancy predictor,
pairwise map registration, and observation-constrained planning. Predicted
structure can guide registration and target ranking, while measured occupancy
remains authoritative for collision checking and persistent-map updates.

This repository accompanies the manuscript **“Design and Audit of
Resource-Constrained Predictive Multirobot Exploration.”** It contains a
usable ROS 2 predictor, a declared reconstruction of the training workflow, an
offline audit of archived two-robot registration, and tooling for future
physical-team data collection. It is not an exact historical reproduction of
every result in the manuscript.

## Four-robot physical experiment branch

The canonical branch for the prospective four-SenseBeetle experiment at
Tongfang 27F is
[`sensebeetle-n4-tongfang27`](https://github.com/SenseLabRobo4111/MSO/tree/sensebeetle-n4-tongfang27).
Use that branch for field-readiness review and every future N=4 run commit.
Start with the [English experiment guide](experiments/physical_team/tongfang27_n4/README.md)
or the [中文实验说明](experiments/physical_team/tongfang27_n4/README.zh.md).

The immediate field task is three independent 600-second repetitions in the
frozen Tongfang 27F ROI. All four robots must autonomously explore with the
locked MSO model under nominal networking. Each repetition requires five
all-topic rosbags (four robot-local plus one coordinator), four synchronized
third-person videos (one per robot), and a common ROS/video/GT time axis. The
existing 32-run factorial is an optional later extension, not the immediate
field task.

This is currently a protocol package, not a physical-result release:
`collection_ready: false`, `results_status: not_collected`, and
`claim_authorized: false`. The 342,771-parameter model identity is verified;
the complete online stack, independent GT/reference system, network
 verification, the integration rehearsal/pilot gates, and per-run auditor still block
collection. `main` remains the manuscript and general-software review branch.

```bash
git fetch origin
git switch --track origin/sensebeetle-n4-tongfang27
```

## Read this first

| Component | Included | Evidence status |
|---|---:|---|
| 342,771-parameter student and teacher definitions | Yes | Architecture is verified |
| ROS 2 prediction node and launch file | Yes | Runtime interface is usable |
| Historical manuscript trainer and exact 5,385/1,356 split | No | Not recovered |
| Manuscript checkpoint and model-selection trace | No | Not recovered |
| Two recovered generator candidates | Yes | Architecture-compatible; candidate A is locked for the prospective N=4 deployment experiment, not identified as the manuscript checkpoint |
| Reconstructed training and evaluation workflow | Yes | Forward reconstruction, not historical reproduction |
| Preserved-archive inventory | Yes | 8,304 sample records with source hashes; not the manuscript split |
| Archived two-robot registration replay | Yes | Offline negative-result audit; not deployed validation |
| Reconstructed integrated offline audit | Yes | Portable, one-scene exploratory audit; not a deployed closed loop |
| Passive N=2/3/4/5 collection protocol | Yes | Instrumentation only; no new N=3, N=4 or N=5 physical results |
| Physical rosbags | External | [Public read-only Google Drive folder](https://drive.google.com/drive/folders/1mbCuIISidEy87mmWPbZTtiRKfW54Fhii?usp=sharing); 18 bags and 18 metadata files verified |
| Portable peer-review data | Yes | [`paper/nature_communications/peer_review_data/`](paper/nature_communications/peer_review_data/) with repository-level hashes |

The reported evidence is limited to structured two-dimensional indoor settings.
It does not establish building-disjoint generalisation, operation through
communication failures, online recovery after an incorrect map commit, or
physical scaling beyond the evaluated two-robot arenas.

## Raw data access

The 10 core two-robot bags and 8 auxiliary controlled single-robot bags are
available in a [public read-only Google Drive folder](https://drive.google.com/drive/folders/1mbCuIISidEy87mmWPbZTtiRKfW54Fhii?usp=sharing).
The folder contains 18 ROS 2 bag files, their 18 `metadata.yaml` companions, a
data README and the same [SHA-256 inventory](RAW_ROSBAG_SHA256SUMS.md) recorded
here. The verified payload is 4.134 GiB. These raw records are outside the
BSD-3-Clause software licence; public access does not by itself grant a
separate data-reuse licence.

The portable processed data used by the reconstructed offline audit are
versioned directly under
[`paper/nature_communications/peer_review_data/`](paper/nature_communications/peer_review_data/).
They are processed audit inputs, not additional physical robot trials. The
paper snapshot's `SHA256SUMS` records every included byte.

## Reviewed version

The canonical manuscript and general-software review branch is `main`. The
immutable snapshot prepared for the current manuscript is tagged
`nc-submission-2026-08-07`.
Retrieve and verify that snapshot with:

```bash
git fetch origin --tags
git switch main
git pull --ff-only origin main
git switch --detach nc-submission-2026-08-07
git rev-parse HEAD
```

The tag and full commit identifier, rather than a moving branch name, should be
recorded in any evaluation report. The older `journal-submission` branch is a
historical alias and is not the canonical entry point.

## Repository layout

```text
MSO/
|-- experiments/
|   `-- physical_team/       # passive N=2/3/4/5 collection protocol
|-- integrated_offline/      # reconstructed one-scene exploratory audit
|-- registration_replay/     # offline audit of archived two-robot bags
|-- repro_reconstructed/     # declared training reconstruction and candidates
|-- launch/
|   |-- sensemap.launch.py
|   |-- physical_team_capture.launch.py
|   |-- physical_team_3.launch.py
|   |-- physical_team_4.launch.py
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

## ROS 2 runtime

The runtime requires Python 3, ROS 2, PyTorch, NumPy, OpenCV, and scikit-learn.
ROS dependencies are declared in `package.xml`. The reported deployment used
ROS 2 Humble on Ubuntu 22.04 and an NVIDIA Jetson AGX Orin. On Jetson hardware,
install the PyTorch build compatible with the installed JetPack version.

```bash
mkdir -p ~/mso_ws/src
cd ~/mso_ws/src
git clone git@github.com:SenseLabRobo4111/MSO.git
cd ~/mso_ws

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

The manuscript checkpoint is not available in this revision. Supply an
explicit compatible checkpoint or state dictionary when launching:

```bash
ros2 launch sensemap sensemap.launch.py \
  robot_id:=0 \
  model_path:=/absolute/path/to/student_weights.ckpt \
  architecture:=deconv \
  crop_size:=256
```

Equivalent direct invocation:

```bash
ros2 run sensemap sensemap_predictor --ros-args \
  -p robot_id:=0 \
  -p model_path:=/absolute/path/to/student_weights.ckpt \
  -p architecture:=deconv \
  -p crop_size:=256
```

The default `deconv` model has 342,771 trainable parameters. Loading fails if
the path is empty or the state dictionary does not match the selected
architecture. The prospective Tongfang 27F protocol locks recovered candidate
A to this architecture as `mso_deconv_342771_candidate_a`. This is a deployment
choice for a new experiment, not evidence that the artifact is the checkpoint
underlying a manuscript table.

### ROS 2 interfaces

For `robot_id:=N`, the predictor uses:

| Direction | Name | Type | Purpose |
|---|---|---|---|
| Subscribe | `/robot_N/map` | `nav_msgs/msg/OccupancyGrid` | Measured occupancy map |
| Publish | `/robot_N/provisional_predicted_map` | `nav_msgs/msg/OccupancyGrid` | Current provisional local prediction |
| Publish | `/robot_N/provisional_predicted_map_global` | `nav_msgs/msg/OccupancyGrid` | Accumulated provisional prediction |
| Publish | `/robot_N/way_point` | `geometry_msgs/msg/PoseStamped` | Optional frontier goal |
| Transform | `global_map` to `robot_N/base_link` | TF2 | Pose used for crop placement |

The inference callback runs once per second. A planner may read the latest
cached prediction more frequently; that read frequency is not the inference
frequency.

`crop_size` is a validated positive integer and defaults to 256 cells, matching
the model input side length. Occupancy channels are resized with
nearest-neighbour sampling. Every known measured cell is copied unchanged into
the local provisional output, unknown sentinels are excluded from accumulation
arithmetic, and current measurements are re-applied after accumulation. The
callback publishes nothing when the required transform or a valid positive map
resolution is unavailable. The
`provisional_` topic prefix is deliberate: these code-level contracts and unit
tests do not establish checkpoint suitability, integrated deployment behavior,
planner safety, or online registration accuracy. These outputs must not enable
map-transform state mutation; the public pairwise manager remains fail-closed.
If an incoming map origin no longer lies on the accumulated cell lattice, the
node resets provisional history instead of rounding the offset. Set
`crop_size:=534` only to reproduce the legacy physical support window; doing so
still resizes categorical input to the 256-cell model interface and is not the
submission default.

## Reconstructed training package

[`repro_reconstructed/`](repro_reconstructed/README.md) provides:

- a canonical reconstructed training and evaluation entry point;
- two recovered generator-only candidate states with source provenance;
- an 8,304-row inventory for the preserved 6,841/1,463 archive labels;
- per-sample source and processed-pair hashes;
- a split builder that requires an explicit semantic group such as building,
  floorplan, or scene and rejects cross-split group leakage and exact processed
  duplicates; and
- locked reconstruction dependencies and artifact verification.

Verify the package before use:

```bash
python3 repro_reconstructed/tools/verify_artifacts.py
python3 -m pytest repro_reconstructed/tests -q
```

The preserved archive is not the manuscript's reported 5,385/1,356 partition.
Worker prefixes in legacy directory names are generation-process identifiers,
not building or floorplan identities, and the tools refuse to treat them as
semantic split groups.

The legacy files `lightning_model.py`, `main_unet_gan.py`, and
`train_gan_new.py` remain research prototypes. They depend on missing historical
modules and do not implement the complete manuscript training programme. Use
the reconstructed package for any new, explicitly labelled experiment.

## Registration replay

[`registration_replay/`](registration_replay/README.md) audits archived
two-robot physical bags offline. It reconstructs the legacy affine behaviour,
compares it with a constrained unit-scale estimator, and records structural and
pose-reference checks. The audit exposes failure modes in the archived
registration evidence; it does not validate deployed online transform accuracy,
false-accept/false-reject rates, rollback, or recovery.

The reusable pairwise manager is fail-closed by default. Any future commit path
requires separate validation on newly collected event-level transforms and an
independent reference. Do not cite the replay as a successful online
registration experiment.

The archived bags retain the historical `/robot_N/predicted_map` topic names;
that provenance is not evidence that the current provisional runtime contract
was deployed during those recordings.

## Reconstructed integrated offline audit

[`integrated_offline/`](integrated_offline/README.md) replays saved probability
maps through a reconstructed registrar, pre-commit gate, atomic measured-map
update and deterministic planning proxies. It also compares prediction and
observed-only inputs on matched perturbations. The portable input ZIP, exact
digest and extraction layout are listed above and in the component README.

The audit is intentionally exploratory and limited to correlated snapshots
from one archived A3 scene. Its cluster-aware results do not establish a safety
advantage, causal exploration benefit, online execution, physical scaling or
communication robustness. It should be cited as a bounded audit rather than a
validated deployment.

## Physical-team collection protocol

[`experiments/physical_team/`](experiments/physical_team/README.md) contains a
passive recorder, schemas, preflight checks, N=2/3/4/5 launch examples, and run
and campaign diagnostics. It contains no simultaneous three-, four- or
five-robot physical dataset. The diagnostics explicitly set
`claim_authorized` to `false`; they are collection aids, not certificates for a
manuscript claim.

The [prospective Tongfang 27F four-SenseBeetle campaign](experiments/physical_team/tongfang27_n4/README.md)
publishes the immediate three-run field requirements and a separate optional
N=4 factorial plan. Its exact 342,771-parameter recovered model candidate is
identified and verified. Collection remains blocked until the full online
stack, independent GT/reference, recording, synchronization, integration
rehearsal and per-run audit evidence exist.

## Dataset loader format

The legacy `MapReconstructionDataset` expects one directory per sample:

```text
dataset_root/
|-- train/
|   `-- sample_000001/
|       |-- obs_0.png
|       `-- local_map_0.png
`-- test/
    `-- ...
```

`obs_0.png` is a three-channel observation and `local_map_0.png` is a binary
target occupancy map. Images are resized to 256 by 256 pixels with
nearest-neighbour sampling. The directory names describe the legacy loader
only; they do not establish a building-disjoint split or an untouched model-
selection partition.

## Verification

The non-ROS evidence packages can be checked with:

```bash
python3 repro_reconstructed/tools/verify_artifacts.py
python3 -m pytest registration_replay/tests repro_reconstructed/tests integrated_offline/tests -q
```

ROS package tests additionally require the normal ROS 2 ament test plugins.

## Citation and contact

No article DOI or archival software DOI exists for this release. Cite the
`nc-submission-2026-08-07` tag together with the exact Git commit returned by
`git rev-parse HEAD`.

Correspondence: Fei Qiao, `qiaofei@tsinghua.edu.cn`.

## Licence and provenance

Source and documentation owned by the MSO authors are released under the
[BSD 3-Clause License](LICENSE). This permits editors, reviewers and other users
to inspect, run, modify and redistribute that author-owned material under the
conditions in `LICENSE`.

Some files include or adapt material distributed under separate upstream
licences. Their origins, affected paths, modification status and retained terms
are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Those upstream
terms remain in force; in particular, the identified FFC implementation and ROS
2 test templates remain under Apache-2.0. The BSD licence does not establish or
expand reuse rights in recovered checkpoint candidates, external datasets or
raw rosbags. [`REVIEW_ACCESS_TERMS.md`](REVIEW_ACCESS_TERMS.md) records the
separate boundary for the publicly shared raw records.

The occupancy-contract tests verify deterministic array transformations only.
No integrated robot run, end-to-end checkpoint evaluation, or deployment
evidence was generated as part of this software correction.
