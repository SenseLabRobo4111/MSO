# Evidence boundary for physical team scaling

Audit date: 2026-08-07.

This audit separates physical runs from simulation and map replay. It does not
upgrade a file to physical evidence merely because its directory contains
multiple robot namespaces.

## Search scope

The inspected project tree occupies 92.443 GiB (approximately 99.2 GB in
decimal units) and contains exactly 18 ROS 2 SQLite bags. The search also
covered the currently reachable lab workstation, its ROS workspaces, and the
available raw-data mirror. Two representative mirrored bags (one two-robot
sense run and one MSO arena run) have identical local/remote SHA-256 digests.
No additional bag was present in the searched remote workspaces.

## Evidence found and classification

- `Exp/experiment_video/rosbag/sense1-1` through `sense5-2` contains ten ROS 2
  bags. Every bag contains only `robot_0` and `robot_1`; the common dynamic
  transforms are `global_map -> robot_0/base_link` and
  `global_map -> robot_1/base_link`. Bag durations range from about 56 to
  113 seconds. These are two-robot physical recordings across five named scenes
  and two recordings per scene, not five-robot runs. They contain local,
  predicted, global-predicted, and merged maps plus TF/rosout, but no raw LiDAR,
  velocity command, registration decision/commit, reference transform, or
  communication-state stream. No locked manifest proves that the apparent
  pairs are matched repeated trials. They therefore remain suitable for the
  bounded message-size audit, not for repeated-team or impairment claims.
- `Exp/arena1` contains eight physical single-robot bags: two trials each for
  MSO, MapEx, UPEN, and IG-Hector. They include LiDAR, velocity, map, and
  transform messages, but no concurrent multirobot evidence.
- `Exp/4.3` contains simulation outputs. `Exp/A2/A2/tianda/N3` and `N5`, and
  `Exp/A3`, contain robot-indexed grid snapshots,
  trajectory CSVs, and software-node logs. They do not contain raw LiDAR,
  odometry, controller, hardware identity, or external tracking streams. They
  are simulation or common-canvas/public-floorplan replay evidence only.
- The inspected deployment revision of `map_merge_node.py` explicitly
  subscribes to `robot_0` and `robot_1`, so it is not evidence of an N>2 online
  fusion service. No synchronised three-, four- or five-robot MSO physical bag
  was found in the available project records.
- The public repository has no hidden remote branch or large-file object with
  three-, four- or five-robot physical data.
- A separate local project contains five matched synchronous/asynchronous
  four-robot communication-scheduling groups (40 bags in total). Its protocol,
  software, and README identify it as a different value-of-information
  scheduling study built on another exploration policy. It is quarantined from
  MSO and cannot be reused as MSO physical evidence.

## Conclusion

No matched repeated MSO N=2 dataset with locked conditions, no executed
communication-loss/disconnection record, and no synchronised N>=3 MSO physical
bag is currently available in the inspected sources. The files in this
directory are collection and audit protocol only. They contain no new
experimental measurements and must not be cited as results. All analysis
scripts keep `claim_authorized` false; a future campaign still requires
independent verification of raw bags, physical references, protocol lock,
network logs, safety outcomes, and statistical analysis.
