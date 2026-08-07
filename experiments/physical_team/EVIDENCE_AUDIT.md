# Evidence boundary for physical team scaling

Audit date: 2026-08-07.

This audit separates physical runs from simulation and map replay. It does not
upgrade a file to physical evidence merely because its directory contains
multiple robot namespaces.

## Evidence found

- `Exp/experiment_video/rosbag/sense1-1` through `sense5-2` contains ten ROS 2
  bags. Every bag contains only `robot_0` and `robot_1`; the common dynamic
  transforms are `global_map -> robot_0/base_link` and
  `global_map -> robot_1/base_link`. Bag durations range from about 56 to
  113 seconds. These are two-robot physical runs across five named scenes and
  two trials, not five-robot runs.
- `Exp/arena1` contains eight physical single-robot bags: two trials each for
  MSO, MapEx, UPEN, and IG-Hector. They include LiDAR, velocity, map, and
  transform messages, but no concurrent multirobot evidence.
- `Exp/A2/A2/tianda/N3` and `N5` contain robot-indexed grid snapshots,
  trajectory CSVs, and software-node logs. They do not contain raw LiDAR,
  odometry, controller, hardware identity, or external tracking streams. They
  are simulation or public-floorplan replay evidence only.
- The inspected deployment revision of `map_merge_node.py` explicitly
  subscribes to `robot_0` and `robot_1`, so it is not evidence of an N>2 online
  fusion service. No synchronised three- or five-robot physical bag was found
  in the available project records.
- The public repository has no hidden remote branch or large-file object with
  three- or five-robot physical data.

## Conclusion

No publishable three- or five-robot physical-team dataset is currently
available in the inspected sources. The files in this directory are a capture
and audit protocol only. They contain no new experimental measurements and
must not be cited as results. The analysis scripts report protocol diagnostics
only and always set `claim_authorized` to `false`; a future matched N=2/3/5
campaign would still require independent verification of the raw bags,
references, protocol lock and statistical analysis before supporting a
manuscript claim.
