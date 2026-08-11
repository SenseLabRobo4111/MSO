# Tongfang 27F three-run four-robot evidence checklist

This checklist is part of the protocol lock. A checked box records that an
item was inspected; it does not by itself authorize a result claim.

## Before the campaign

- [ ] Keep raw GT maps/masks, IP addresses, hardware serials, video and site
  photographs out of public Git; use approved controlled evidence storage.
- [ ] Preserve the exact campaign, team, arena, start-pose, model and reference
  lock files beside the data and record their SHA-256 digests.
- [ ] Replace every `SET_...` value only after the corresponding evidence
  exists. Filling a template or self-authored status field never authorizes the
  immediate task; the receiving team must independently review its actual
  recording, synchronization, online-stack and per-run audit implementation.
- [ ] Verify `mso_deconv_342771_candidate_a` against artifact SHA-256
  `da4458514656d41fba0e0ce6d4f4967997ff0a97f2e905a458757609edf3a3a8`.
- [ ] Strictly load the candidate into
  `DistillMapNetDeconv(image_size=256, dim=4)`, confirm exactly 342,771
  trainable parameters, and reproduce the fixed-fixture output fingerprint.
- [ ] Confirm that all four robots use those exact portable-state bytes and
  that no hardware conversion has become the sole model copy.
- [ ] Describe it as a recovered candidate extracted from a checkpoint retained
  in a preserved deployment copy; do not claim historical online use or identify
  it as the manuscript checkpoint or the historical `w/o FFC` 304K row.
- [ ] Verify one frozen software commit on all four robots and a clean deployed
  worktree.
- [ ] Verify the locked architecture and FFC source hashes and either reproduce
  the reference runtime exactly or retain a reviewed numeric-equivalence report
  for the deployed Torch/device combination.
- [ ] Survey and freeze the three immediate-task four-pose sets in `tf27_gt`;
  verify geofence membership and minimum separation. Keep any eight-block
  follow-up pose set in a separate lock.
- [ ] Freeze the GT occupancy image, accessible-free ROI, dynamic exclusion
  mask, door state and furniture state.
- [ ] Calibrate overhead tracking to `tf27_gt`, measure uncertainty, and freeze
  four robot/tag extrinsics.
- [ ] Confirm that external poses are not subscribed to by MSO or its planner.
- [ ] Confirm a separate out-of-band emergency-stop and tracking/control link.
- [ ] Remove every traffic-control/fault rule and prove that all three immediate
  runs use the frozen nominal network condition.
- [ ] Complete a full-chain integration rehearsal and keep its identifier
  permanently outside the three formal repetitions.

## Immediate task identity

- [ ] Use exactly `tf27_n4_mso_repeat_01`, `tf27_n4_mso_repeat_02`, and
  `tf27_n4_mso_repeat_03`, in that order.
- [ ] Confirm all four robots strictly loaded the same locked MSO artifact and
  that live MSO outputs causally feed the permitted registration/frontier
  interfaces; shadow-only execution is not accepted.
- [ ] Confirm the 600-second formal window, 30-second pre-roll and 10-second
  post-roll before every repetition.
- [ ] Confirm the cold reset attestation: no map, MSO cache, registration,
  planner/frontier, DDS transient or network-fault state survives from the
  previous repetition.

## All-topic rosbag requirement

- [ ] Start one all-topic local bag on each of `robot_0` through `robot_3` and
  one all-topic coordinator bag: exactly five required bags per repetition.
- [ ] Use record-all discovery, include hidden and late-appearing topics, and
  retain the actual command, ROS distribution/version and exit status.
- [ ] Save topic inventories before T0, near T=300 s and after T=600 s, plus
  topic types, offered/requested QoS, message counts, first/last timestamps and
  sequence/drop diagnostics.
- [ ] Start all five bags at least 30 s before T0 and stop only after the end
  marker, stationary state, flush, and 10-second post-roll.

The following streams are an audit minimum, not a recording whitelist.

For each robot `i=0..3`:

- [ ] `/robot_i/lidar/points`
- [ ] `/robot_i/odom`
- [ ] `/robot_i/map` (measured occupancy)
- [ ] `/robot_i/provisional_predicted_map`
- [ ] raw prediction probability or logits before thresholding
- [ ] For an optional 32-run extension only, prediction-shadow output in
  `observed_only` runs
- [ ] `/robot_i/cmd_vel`
- [ ] planner goal, candidate set, selected frontier and planned path
- [ ] controller state, battery, temperature and diagnostics
- [ ] `/ground_truth/robot_i/pose`

Global streams:

- [ ] `/tf` and `/tf_static`, with no `/clock`
- [ ] measured merged map and persistent map revision/hash
- [ ] `/mso/registration_event` for candidate, decision and commit transitions
- [ ] `/mso/coverage_event`, computed from measured cells and frozen GT ROI
- [ ] `/mso/safety_event`
- [ ] For an optional 32-run extension only, network fault events, probes and
  rule packet/byte counters
- [ ] ROS logs and CPU/GPU/power telemetry

## Four robot-specific third-person videos

- [ ] Record four continuous raw videos per repetition:
  `robot_0_third_person` through `robot_3_third_person`, one per robot.
- [ ] Do not substitute a single overview/overhead recording for any of the
  four robot-specific views.
- [ ] For every video retain run/robot/camera ID, camera serial, resolution,
  frame rate, time base, first/last frame, dropped-frame report, size and
  SHA-256.
- [ ] Start each video at least 30 s before T0, never pause or edit it, and stop
  only after the 10-second post-roll. Keep cameras/operators outside geofence.

## Common ROS, video and GT time axis

- [ ] Synchronize four robot hosts, coordinator, GT/reference and four videos
  to one UTC/PTP or verified NTP basis; `use_sim_time=false` and no `/clock`.
- [ ] Save clock samples before, during (at least 1 Hz) and after the run;
  absolute host offset must be <=5 ms and RTT <=20 ms.
- [ ] At T=0, 300 and 600 s write `/experiment/sync_marker` to all visible bags
  and trigger the same physical LED/flash event in all four videos.
- [ ] Verify video-to-ROS residual <=1 frame and GT/reference join <=100 ms.
- [ ] Reject a run with a missing marker, unexplained drift, clock step/backward
  jump or synchronization threshold violation.

## Per-run files

- [ ] Immutable run manifest with software and model digests
- [ ] Robot roster with base, LiDAR and compute serial numbers
- [ ] Four robot-local all-topic bags and one coordinator all-topic bag
- [ ] Append-only registration, planner, coverage, safety and network JSONL
- [ ] `reference_transforms.jsonl` joined to every registration decision
- [ ] Four continuous robot-specific third-person videos and start/end site
  photographs; an optional overview may be retained as additional evidence
- [ ] Snapshots at 0, 120, 240, 360, 480 and 600 seconds
- [ ] Snapshots before and after every persistent-map commit
- [ ] Preflight report, run audit, network audit and final SHA-256 inventory
- [ ] Pre/mid/post topic inventories, clock samples, sync-marker joins, video
  metadata and cold-reset attestation

## Run acceptance

- [ ] The run lasted for the locked 600-second recording horizon, including any
  stationary time after the planner exhausted frontiers.
- [ ] For the immediate task, hardware-to-pose assignment, start poses and run
  identity match the pre-registered three-run manifest. Only an optional 32-run
  extension is checked against `config/campaign.yaml`.
- [ ] The four robot clocks meet the frozen offset and round-trip limits.
- [ ] All four videos meet the one-frame alignment limit and cover the full
  pre-roll, formal window and post-roll.
- [ ] Exactly five valid all-topic bags cover the full required window.
- [ ] No prior-run state was loaded; the signed cold-reset record is complete.
- [ ] Every registration decision has an independent time-matched reference.
- [ ] Coverage counts measured observations only; predicted cells do not count.
- [ ] No per-run GT alignment, result-dependent ROI edit or outlier deletion was
  performed.
- [ ] Any collision, stop, intervention or failure remains in the archive and
  in the safety population.
- [ ] An invalid technical run retains its original ID and evidence; a repeated
  run receives a new ID and is never silently substituted.

## Immediate three-run outputs

- [ ] One row and one complete evidence inventory for each of the three exact
  planned IDs, plus explicit rows for every technical-invalid replacement
- [ ] All three measured-only GT coverage curves over 0--600 s
- [ ] Final coverage and time to 0.8/0.9 coverage, with censoring reported
- [ ] Every safety stop, intervention, collision/near-collision and technical
  invalidity; no outcome-based deletion or replacement
- [ ] All five bag inventories, four video inventories and synchronization
  residuals for each repetition
- [ ] Descriptive n=3 reporting that does not treat one site/hardware pool as
  independent building-level replication

## Optional 32-run follow-up outputs

- [ ] One row for each of the exact 32 planned run IDs
- [ ] Eight complete four-treatment blocks
- [ ] Block-paired primary contrast under nominal network
- [ ] Block-paired contrast under impairment and interaction contrast
- [ ] All individual coverage traces and paired differences
- [ ] Registration errors, gate decisions, wrong commits and safety events
- [ ] Communication loss and post-reconnection recovery timeline
- [ ] `claim_authorized` remains `false` pending independent review
