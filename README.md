# MSO

**Lightweight local map prediction and fusion for multirobot exploration**

[中文说明](README.zh.md)

MSO (Make Sense at Once) is a research system for two-dimensional indoor
multirobot exploration. It couples a compact local occupancy-map predictor with
pairwise map registration so that robots can use predicted structure without a
pre-known global relative pose.

This repository is the code home for the manuscript **“Lightweight Local Map
Prediction and Fusion for Multirobot Exploration.”** The current revision
contains the predictor architecture, ROS 2 inference node, launch file, and
training-oriented model components. The complete archival release will also
include the frozen evaluation package, configuration files, and versioned model
weights used for the reported results.

## Method at a glance

- A 342K-parameter distilled predictor completes a local occupancy grid onboard.
- Observed, unknown, and occupied cells are encoded as a three-channel local map.
- The prediction node publishes both local and accumulated global predictions.
- Pairwise registration is applied when peer maps are exchanged. Predicted cells
  are treated as provisional and are superseded by later sensor observations.
- Planning uses prediction for frontier ranking while collision checking remains
  constrained by observed free space.

The manuscript deliberately bounds the evidence to structured indoor settings.
It does not claim building-disjoint generalisation, communication-failure
robustness, or physical scaling beyond the evaluated robot teams.

## Repository layout

```text
MSO/
├── launch/
│   └── sensemap.launch.py       # ROS 2 predictor launch file
├── resource/
│   └── sensemap                 # ament package marker
├── sensemap/
│   ├── explore_model/
│   │   ├── SenseMapNet.py       # student and teacher architectures
│   │   ├── critic_model.py      # adversarial discriminator
│   │   ├── dataset.py           # occupancy-map dataset loader
│   │   ├── ffc.py               # Fast Fourier Convolution blocks
│   │   ├── lightning_model.py   # training wrapper
│   │   ├── main_unet_gan.py     # research training entry point
│   │   └── train_gan_new.py     # distillation training loop
│   └── predict_map.py           # ROS 2 inference node
├── package.xml
├── setup.cfg
└── setup.py
```

## Release scope

The repository is being prepared as a versioned research artifact. The current
contents and remaining archival items are listed explicitly below so that users
do not mistake a partial checkout for the complete experiment package.

| Component | Current status |
|---|---|
| Predictor network definitions | Included |
| ROS 2 predictor node and launch file | Included |
| Dataset loader and training-oriented modules | Included |
| Trained checkpoint used in the manuscript | Archival release in preparation |
| Full pairwise-registration and deployment package | Available to editors and reviewers; archival release in preparation |
| Frozen evaluation manifests, event logs, and plotting scripts | Available to editors and reviewers; archival release in preparation |
| Third-party KTH, HouseExpo, and MRPB data | Obtain from the original dataset providers |
| Physical-experiment rosbags and processed source data | Available for editorial and peer-review audit; public archive in preparation |

## Requirements

The deployed node was developed for ROS 2 and Python 3. A typical installation
requires:

- ROS 2 with `rclpy`, `tf2_ros`, `geometry_msgs`, and `nav_msgs`
- PyTorch built for the target CPU or CUDA platform
- NumPy
- OpenCV
- scikit-learn

Training additionally uses Pillow, Matplotlib, tqdm, and PyTorch Lightning. The
exact software and hardware environment used for the manuscript will be frozen
with the archival release. On NVIDIA Jetson hardware, install the PyTorch build
recommended for the installed JetPack version rather than a generic wheel.

## Build the ROS 2 package

```bash
mkdir -p ~/mso_ws/src
cd ~/mso_ws/src
git clone git@github.com:SenseLabRobo4111/MSO.git
cd ~/mso_ws

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

The checkpoint is not committed to Git. Supply its absolute path at launch:

```bash
ros2 launch sensemap sensemap.launch.py \
  robot_id:=0 \
  model_path:=/absolute/path/to/mso_distilled.ckpt \
  architecture:=deconv
```

The same node can be started directly:

```bash
ros2 run sensemap sensemap_predictor --ros-args \
  -p robot_id:=0 \
  -p model_path:=/absolute/path/to/mso_distilled.ckpt \
  -p architecture:=deconv
```

## Checkpoint format

The inference node accepts either:

1. a PyTorch Lightning checkpoint containing a `state_dict` whose generator keys
   begin with `gen.`, or
2. a plain state dictionary for `DistillMapNet`.

The model path is mandatory. Loading fails early if it is omitted or if the
checkpoint does not match the selected distilled network architecture. The
default `deconv` variant has 342,771 parameters and corresponds to the 342K
model reported in the manuscript. The legacy `bilinear` variant can be selected
explicitly for compatible checkpoints.

## ROS 2 interfaces

For `robot_id:=N`, the node uses the following interfaces.

| Direction | Name | Type | Purpose |
|---|---|---|---|
| Subscribe | `/robot_N/map` | `nav_msgs/msg/OccupancyGrid` | Observed occupancy map |
| Publish | `/robot_N/predicted_map` | `nav_msgs/msg/OccupancyGrid` | Current local prediction |
| Publish | `/robot_N/predicted_map_global` | `nav_msgs/msg/OccupancyGrid` | Accumulated prediction |
| Publish | `/robot_N/way_point` | `geometry_msgs/msg/PoseStamped` | Optional frontier goal interface |
| Transform | `global_map` to `robot_N/base_link` | TF2 | Robot pose used for crop placement |

The default prediction period is one second. Topic names and frame identifiers
currently follow the deployment naming convention used in the manuscript.

## Dataset layout

`MapReconstructionDataset` expects one directory per sample:

```text
dataset_root/
├── train/
│   ├── sample_000001/
│   │   ├── obs_0.png
│   │   └── local_map_0.png
│   └── ...
└── test/
    └── ...
```

- `obs_0.png` is the three-channel observed map.
- `local_map_0.png` is the binary target occupancy map.
- Images are resized to 256 by 256 pixels with nearest-neighbour sampling.

Dataset preparation must preserve the split manifest used for evaluation. The
manuscript does not interpret its retained floorplan-disjoint split as proof of
building-disjoint generalisation.

## Reproducibility and reporting

For a result to be attributable to the submitted manuscript, retain the exact
checkpoint, split manifest, run seed, planner configuration, map resolution,
frame convention, and evaluation mask. Registration results should also retain
the estimated and reference transforms, gate decision, rejection reason, and
event-level identifiers. Aggregate plots alone are not sufficient for auditing
false accepts, false rejects, or pose error.

## Citation

The manuscript is currently under review. Please cite the archival software
record and article DOI once they are available. Until then, cite the repository
and a fixed commit hash so that the referenced code can be recovered.

## Data and code availability

Custom code and processed data needed for editorial and peer-review assessment
are available from the corresponding author. A versioned archive with a
persistent identifier will be created for publication. Dataset licences prevent
redistribution of some third-party source material; the original KTH, HouseExpo,
and MRPB sources should be used for those assets.

## Contact

Correspondence about the manuscript and research artifact should be addressed to
Fei Qiao at `qiaofei@tsinghua.edu.cn`.

## License

No software licence is granted by the current repository revision. The archival
release will state the approved licence and any restrictions applying to bundled
third-party components.
