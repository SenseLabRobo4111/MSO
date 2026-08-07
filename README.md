# MSO

**Resource constrained predictive map exchange for multirobot indoor exploration**

[中文说明](README.zh.md)

MSO (Make Sense at Once) is a research system for two-dimensional indoor
multirobot exploration. It couples a compact local occupancy predictor with
pairwise map registration and observation-constrained planning. Predictions can
propose structure for target ranking and registration, while measured occupancy
remains authoritative for collision checking and persistent map updates.

This repository accompanies the manuscript **“Resource Constrained Predictive
Map Exchange for Multirobot Indoor Exploration.”** The current revision exposes
the predictor architecture, training components, ROS 2 inference node, and
launch configuration. It is not yet the complete archival experiment package;
the scope table below distinguishes included code from material supplied
separately for editorial and peer-review assessment.

## System boundary

- The deployed student has 342,771 trainable parameters.
- Input is a 256 by 256 local grid with obstacle, unknown, and free channels.
- Output is an occupied-cell probability for the unknown region.
- Observed cells overwrite predicted cells when new measurements arrive.
- Prediction ranks candidate targets; path planning and collision checking use
  measured occupancy.
- Pairwise registration is intended for encounters without a known initial
  interrobot transform.

The reported evidence is bounded to structured two-dimensional indoor settings.
It does not establish building-disjoint generalisation, communication-failure
robustness, online recovery after a wrong commit, or physical scaling beyond the
evaluated teams.

## Repository contents

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

| Component | Status in this revision |
|---|---|
| Student and teacher network definitions | Included |
| ROS 2 predictor node and launch file | Included |
| Dataset loader and training-oriented modules | Included |
| Manuscript checkpoint | Supplied to reviewers; public archive in preparation |
| Deployed pairwise-registration package | Supplied to reviewers; public archive in preparation |
| Frozen transform evaluator, event manifests, and event logs | Supplied in the peer-review data archive |
| Physical rosbags and processed source data | Available through a controlled review link |
| KTH, HouseExpo, and MRPB source data | Obtain from the original providers |

## Requirements

The ROS node requires Python 3, ROS 2, PyTorch, NumPy, OpenCV, and scikit-learn.
Its ROS dependencies are `rclpy`, `tf2_ros`, `geometry_msgs`, and `nav_msgs`.
Training additionally uses Pillow, Matplotlib, tqdm, and PyTorch Lightning.

The reported deployment used ROS 2 Humble on Ubuntu 22.04 and an NVIDIA Jetson
AGX Orin. On Jetson hardware, install the PyTorch build matched to the installed
JetPack version.

## Build

```bash
mkdir -p ~/mso_ws/src
cd ~/mso_ws/src
git clone git@github.com:SenseLabRobo4111/MSO.git
cd ~/mso_ws

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

The model checkpoint is not stored in the current Git revision. Pass its
absolute path when launching the node:

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

The default `deconv` model is the 342,771-parameter student reported in the
manuscript. A compatible legacy checkpoint can select `architecture:=bilinear`.
Loading stops with an error if `model_path` is empty or the state dictionary does
not match the selected architecture.

## Checkpoint formats

The inference node accepts either:

1. a PyTorch Lightning checkpoint with a `state_dict` whose generator keys
   begin with `gen.`; or
2. a plain `DistillMapNet` state dictionary.

## ROS 2 interfaces

For `robot_id:=N`, the node uses:

| Direction | Name | Type | Purpose |
|---|---|---|---|
| Subscribe | `/robot_N/map` | `nav_msgs/msg/OccupancyGrid` | Measured occupancy map |
| Publish | `/robot_N/predicted_map` | `nav_msgs/msg/OccupancyGrid` | Current local prediction |
| Publish | `/robot_N/predicted_map_global` | `nav_msgs/msg/OccupancyGrid` | Accumulated prediction |
| Publish | `/robot_N/way_point` | `geometry_msgs/msg/PoseStamped` | Optional frontier goal |
| Transform | `global_map` to `robot_N/base_link` | TF2 | Pose used for crop placement |

The default prediction period is one second. Topic and frame names follow the
deployment convention used in the manuscript.

## Dataset layout

`MapReconstructionDataset` expects one directory per sample:

```text
dataset_root/
|-- train/
|   |-- sample_000001/
|   |   |-- obs_0.png
|   |   `-- local_map_0.png
|   `-- ...
`-- test/
    `-- ...
```

- `obs_0.png` is the three-channel observed map.
- `local_map_0.png` is the binary target occupancy map.
- Images are resized to 256 by 256 pixels with nearest-neighbour sampling.

Keep the exact split manifest used for evaluation. The retained split is
floorplan-disjoint; it should not be interpreted as proof of building-disjoint
generalisation.

## Reproducible reporting

Retain the exact checkpoint, split manifest, random seed, planner configuration,
map resolution, coordinate convention, and evaluation mask for every reported
result. Registration records should additionally store estimated and reference
transforms, candidate validity, gate decision, rejection reason, and event ID.
Aggregate plots alone cannot audit pose error, false acceptance, or false
rejection.

## Citation

Until an article DOI and archival software record are available, cite this
repository with a fixed commit hash. Citation metadata will be added to the
versioned public release.

## Data and code availability

Custom code and processed data needed for editorial and peer-review assessment
are available from the corresponding author. A versioned public archive with a
persistent identifier is planned for publication. Licences for third-party KTH,
HouseExpo, and MRPB data prevent redistributing some original assets here.

## Contact

Correspondence: Fei Qiao, `qiaofei@tsinghua.edu.cn`.

## Licence

No software licence is granted by this repository revision. The archival release
will state the approved licence and any restrictions for third-party components.
