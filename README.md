# MSO

**A predictive system for multirobot indoor exploration under resource constraints**

[中文说明](README.zh.md)

MSO (Make Sense at Once) is a research system for two-dimensional indoor
multirobot exploration. It combines a compact local occupancy predictor,
pairwise map registration, and observation-constrained planning. Predicted
structure may support registration and target ranking, while measured occupancy
remains authoritative for collision checking and persistent-map updates.

This repository accompanies the manuscript **“A Predictive System for
Multirobot Indoor Exploration under Resource Constraints.”** It currently exposes
the predictor network definitions and ROS 2 inference interface. It is not a
complete archival reproduction package for the reported training or experiments.

## Evidence boundary

- The deployed student has 342,771 trainable parameters.
- Input is a 256 by 256 local grid with obstacle, unknown, and free channels.
- Output is an occupied-cell probability for the unknown region.
- New measurements overwrite predictions in persistent mapping.
- Prediction may rank candidate targets; planning and collision checking use
  measured occupancy.
- Pairwise registration addresses encounters without a known initial interrobot
  transform.

The reported evidence is limited to structured two-dimensional indoor settings.
It does not establish building-disjoint generalisation, operation through
communication failures, online recovery after an incorrect commit, or physical
scaling beyond the evaluated two-robot arenas.

## Release scope

| Component | Status in this revision |
|---|---|
| Student and teacher network definitions | Included |
| ROS 2 predictor node and launch file | Included |
| Dataset loader | Included as a format reference |
| Historical manuscript training programme | Not included |
| Historical split manifest and checkpoint-selection trace | Not included |
| Manuscript checkpoint | Not included |
| Deployed pairwise-registration package | Not included |
| Offline transform evaluator and event records | Distributed with the manuscript review package, not this repository |
| Physical rosbags | Not included; too large for this repository |
| KTH, HouseExpo, and MRPB source data | Obtain from the original providers |

The files `lightning_model.py`, `main_unet_gan.py`, and `train_gan_new.py` are
legacy research prototypes. They use different defaults, depend on modules that
are not tracked here, and do not implement the complete four-term distillation
programme described in the manuscript. They must not be treated as a canonical
entry point for reproducing the manuscript training results.

## Repository layout

```text
MSO/
|-- launch/
|   `-- sensemap.launch.py
|-- resource/
|   `-- sensemap
|-- sensemap/
|   |-- explore_model/
|   |   |-- SenseMapNet.py
|   |   |-- critic_model.py
|   |   |-- dataset.py
|   |   |-- ffc.py
|   |   |-- lightning_model.py
|   |   |-- main_unet_gan.py
|   |   `-- train_gan_new.py
|   `-- predict_map.py
|-- test/
|-- package.xml
|-- setup.cfg
`-- setup.py
```

## Runtime requirements

The ROS node requires Python 3, ROS 2, PyTorch, NumPy, OpenCV, and scikit-learn.
Its ROS dependencies are `rclpy`, `tf2_ros`, `geometry_msgs`, and `nav_msgs`.
This revision does not provide a locked dependency environment.

The reported deployment used ROS 2 Humble on Ubuntu 22.04 and an NVIDIA Jetson
AGX Orin. On Jetson hardware, use the PyTorch build compatible with the installed
JetPack version.

## Build and launch

```bash
mkdir -p ~/mso_ws/src
cd ~/mso_ws/src
git clone git@github.com:SenseLabRobo4111/MSO.git
cd ~/mso_ws

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

The checkpoint is not stored in this revision. Supply an absolute path when
launching the inference node:

```bash
ros2 launch sensemap sensemap.launch.py \
  robot_id:=0 \
  model_path:=/absolute/path/to/mso_distilled.ckpt \
  architecture:=deconv
```

Equivalent direct invocation:

```bash
ros2 run sensemap sensemap_predictor --ros-args \
  -p robot_id:=0 \
  -p model_path:=/absolute/path/to/mso_distilled.ckpt \
  -p architecture:=deconv
```

The default `deconv` network is the 342,771-parameter student described in the
manuscript. A compatible legacy checkpoint may select `architecture:=bilinear`.
Loading stops with an error if `model_path` is empty or the state dictionary does
not match the selected architecture.

The node accepts either a PyTorch Lightning checkpoint whose generator keys in
`state_dict` begin with `gen.`, or a plain `DistillMapNet` state dictionary.

## ROS 2 interfaces

For `robot_id:=N`, the node uses:

| Direction | Name | Type | Purpose |
|---|---|---|---|
| Subscribe | `/robot_N/map` | `nav_msgs/msg/OccupancyGrid` | Measured occupancy map |
| Publish | `/robot_N/predicted_map` | `nav_msgs/msg/OccupancyGrid` | Current local prediction |
| Publish | `/robot_N/predicted_map_global` | `nav_msgs/msg/OccupancyGrid` | Accumulated prediction |
| Publish | `/robot_N/way_point` | `geometry_msgs/msg/PoseStamped` | Optional frontier goal |
| Transform | `global_map` to `robot_N/base_link` | TF2 | Pose used for crop placement |

The inference callback runs once per second and publishes the refreshed local and
global prediction. A planner may read the latest cached prediction more often;
that read frequency is not the inference frequency.

## Dataset loader format

`MapReconstructionDataset` expects one directory per sample:

```text
dataset_root/
|-- train/
|   `-- sample_000001/
|       |-- obs_0.png
|       `-- local_map_0.png
`-- test/
    `-- ...
```

`obs_0.png` is the three-channel observation and `local_map_0.png` is the binary
target occupancy map. Images are resized to 256 by 256 pixels with
nearest-neighbour sampling.

The directory names above describe the legacy loader only. This revision does
not contain the sample-level floorplan partition, an independent validation
partition, or evidence that the held-out evaluation partition was untouched
during historical model selection.

## Reproducibility requirements

A complete archival training release would need the exact split manifest,
teacher and student checkpoints, checkpoint-selection rule, all loss modules,
training entry point, random seeds, evaluation masks, and locked environment.
Those materials are not present in this revision. Aggregate plots likewise do
not permit reconstruction of seed-level uncertainty.

## Citation

Until an article DOI and archival software record are available, cite the exact
Git commit used. The manuscript records its review revision explicitly.

## Contact

Correspondence: Fei Qiao, `qiaofei@tsinghua.edu.cn`.

## Licence

No software licence is granted by this repository revision. A future archival
release must state the approved licence and any third-party restrictions.
