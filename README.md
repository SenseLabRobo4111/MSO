# MSO

**A predictive system for multirobot indoor exploration under resource constraints**

[中文说明](README.zh.md)

MSO (Make Sense at Once) is a research system for two-dimensional indoor
multirobot exploration. It combines a compact local occupancy predictor,
pairwise map registration, and observation-constrained planning. Predicted
structure can guide registration and target ranking, while measured occupancy
remains authoritative for collision checking and persistent-map updates.

This repository accompanies the manuscript **“A Predictive System for
Multirobot Indoor Exploration under Resource Constraints.”** It contains a
usable ROS 2 predictor, a declared reconstruction of the training workflow, an
offline audit of archived two-robot registration, and tooling for future
physical-team data collection. It is not an exact historical reproduction of
every result in the manuscript.

## Read this first

| Component | Included | Evidence status |
|---|---:|---|
| 342,771-parameter student and teacher definitions | Yes | Architecture is verified |
| ROS 2 prediction node and launch file | Yes | Runtime interface is usable |
| Historical manuscript trainer and exact 5,385/1,356 split | No | Not recovered |
| Manuscript checkpoint and model-selection trace | No | Not recovered |
| Two recovered generator candidates | Yes | Architecture-compatible; not identified as the manuscript checkpoint |
| Reconstructed training and evaluation workflow | Yes | Forward reconstruction, not historical reproduction |
| Preserved-archive inventory | Yes | 8,304 sample records with source hashes; not the manuscript split |
| Archived two-robot registration replay | Yes | Offline negative-result audit; not deployed validation |
| Passive N=2/3/5 collection protocol | Yes | Instrumentation only; no new N=3 or N=5 physical results |
| Physical rosbags | No | Available to reviewers through the manuscript process |

The reported evidence is limited to structured two-dimensional indoor settings.
It does not establish building-disjoint generalisation, operation through
communication failures, online recovery after an incorrect map commit, or
physical scaling beyond the evaluated two-robot arenas.

## Reviewed version

The canonical development and review branch is `main`. The immutable snapshot
prepared for the current manuscript is tagged `nc-submission-2026-08-07`.
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
|   `-- physical_team/       # passive N=2/3/5 collection protocol
|-- registration_replay/     # offline audit of archived two-robot bags
|-- repro_reconstructed/     # declared training reconstruction and candidates
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
  architecture:=deconv
```

Equivalent direct invocation:

```bash
ros2 run sensemap sensemap_predictor --ros-args \
  -p robot_id:=0 \
  -p model_path:=/absolute/path/to/student_weights.ckpt \
  -p architecture:=deconv
```

The default `deconv` model has 342,771 trainable parameters. Loading fails if
the path is empty or the state dictionary does not match the selected
architecture. The recovered candidates under `repro_reconstructed/` are
research artifacts; they must not be presented as the checkpoint underlying a
manuscript table.

### ROS 2 interfaces

For `robot_id:=N`, the predictor uses:

| Direction | Name | Type | Purpose |
|---|---|---|---|
| Subscribe | `/robot_N/map` | `nav_msgs/msg/OccupancyGrid` | Measured occupancy map |
| Publish | `/robot_N/predicted_map` | `nav_msgs/msg/OccupancyGrid` | Current local prediction |
| Publish | `/robot_N/predicted_map_global` | `nav_msgs/msg/OccupancyGrid` | Accumulated prediction |
| Publish | `/robot_N/way_point` | `geometry_msgs/msg/PoseStamped` | Optional frontier goal |
| Transform | `global_map` to `robot_N/base_link` | TF2 | Pose used for crop placement |

The inference callback runs once per second. A planner may read the latest
cached prediction more frequently; that read frequency is not the inference
frequency.

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

## Physical-team collection protocol

[`experiments/physical_team/`](experiments/physical_team/README.md) contains a
passive recorder, schemas, preflight checks, N=2/3/5 launch examples, and run and
campaign diagnostics. It contains no simultaneous three- or five-robot physical
dataset. The diagnostics explicitly set `claim_authorized` to `false`; they are
collection aids, not certificates for a manuscript claim.

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
python3 -m pytest registration_replay/tests repro_reconstructed/tests -q
```

ROS package tests additionally require the normal ROS 2 ament test plugins.

## Citation and contact

No article DOI or archival software DOI exists for this release. Cite the
`nc-submission-2026-08-07` tag together with the exact Git commit returned by
`git rev-parse HEAD`.

Correspondence: Fei Qiao, `qiaofei@tsinghua.edu.cn`.

## Licence and provenance

No general software licence is granted for the repository as a whole. The
public repository can be inspected, but reuse and redistribution of the authors'
contributions require a later rights-cleared release. Proposed terms for
editorial and peer-review use are recorded in
[`REVIEW_ACCESS_TERMS.md`](REVIEW_ACCESS_TERMS.md) and still require explicit
author approval before they can be relied on.

Some files include or adapt material distributed under separate upstream
licences. Their origins, affected paths, modification status and retained terms
are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Those upstream
permissions apply only to the relevant material and do not license the rest of
this repository.
