# Tongfang 27F four-robot evidence checklist

This checklist is part of the protocol lock. A checked box records that an
item was inspected; it does not by itself authorize a result claim.

## Before the campaign

- [ ] Keep raw GT maps/masks, IP addresses, hardware serials, video and site
  photographs out of public Git; use approved controlled evidence storage.
- [ ] Preserve the exact campaign, team, arena, start-pose, model and reference
  lock files beside the data and record their SHA-256 digests.
- [ ] Replace every `SET_...` value only after the corresponding evidence
  exists. This protocol-only branch remains source-gated even if every YAML
  field is changed to `collection_ready: true`; a reviewed executable verifier
  change is additionally required.
- [ ] Verify `mso_deconv_342771_candidate_a` against artifact SHA-256
  `da4458514656d41fba0e0ce6d4f4967997ff0a97f2e905a458757609edf3a3a8`.
- [ ] Strictly load the candidate into
  `DistillMapNetDeconv(image_size=256, dim=4)`, confirm exactly 342,771
  trainable parameters, and reproduce the fixed-fixture output fingerprint.
- [ ] Confirm that all four robots use those exact portable-state bytes and
  that no hardware conversion has become the sole model copy.
- [ ] Preserve the label "recovered deployment candidate"; do not identify the
  artifact as the manuscript checkpoint or the historical `w/o FFC` 304K row.
- [ ] Verify one frozen software commit on all four robots and a clean deployed
  worktree.
- [ ] Verify the locked architecture and FFC source hashes and either reproduce
  the reference runtime exactly or retain a reviewed numeric-equivalence report
  for the deployed Torch/device combination.
- [ ] Survey the eight four-pose sets in `tf27_gt`; verify geofence membership
  and the minimum separation.
- [ ] Freeze the GT occupancy image, accessible-free ROI, dynamic exclusion
  mask, door state and furniture state.
- [ ] Calibrate overhead tracking to `tf27_gt`, measure uncertainty, and freeze
  four robot/tag extrinsics.
- [ ] Confirm that external poses are not subscribed to by MSO or its planner.
- [ ] Confirm a separate out-of-band emergency-stop and tracking/control link.
- [ ] Verify that the impaired DDS data traffic is peer-addressable unicast;
  packet loss measured only on a probe is not sufficient.
- [ ] Complete four integration pilots, one per treatment, and keep their run
  identifiers permanently outside the 32-run analysis set.

## Required topics or equivalent typed streams

For each robot `i=0..3`:

- [ ] `/robot_i/lidar/points`
- [ ] `/robot_i/odom`
- [ ] `/robot_i/map` (measured occupancy)
- [ ] `/robot_i/provisional_predicted_map`
- [ ] raw prediction probability or logits before thresholding
- [ ] prediction-shadow output in `observed_only` runs
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
- [ ] network fault events, probes and rule packet/byte counters
- [ ] ROS logs and CPU/GPU/power telemetry

## Per-run files

- [ ] Immutable run manifest with software and model digests
- [ ] Robot roster with base, LiDAR and compute serial numbers
- [ ] Coordinator bag containing every required global and per-robot stream;
  retain any additional local robot bags required by the deployed full stack
- [ ] Append-only registration, planner, coverage, safety and network JSONL
- [ ] `reference_transforms.jsonl` joined to every registration decision
- [ ] Continuous overhead video and start/end site photographs
- [ ] Snapshots at 0, 120, 240, 360, 480 and 600 seconds
- [ ] Snapshots before and after every persistent-map commit
- [ ] Preflight report, run audit, network audit and final SHA-256 inventory

## Run acceptance

- [ ] The run lasted for the locked 600-second recording horizon, including any
  stationary time after the planner exhausted frontiers.
- [ ] Hardware-to-pose assignment and treatment match `campaign.yaml`.
- [ ] The four robot clocks meet the frozen offset and round-trip limits.
- [ ] Every registration decision has an independent time-matched reference.
- [ ] Coverage counts measured observations only; predicted cells do not count.
- [ ] No per-run GT alignment, result-dependent ROI edit or outlier deletion was
  performed.
- [ ] Any collision, stop, intervention or failure remains in the archive and
  in the safety population.
- [ ] An invalid technical run retains its original ID and evidence; a repeated
  run receives a new ID and is never silently substituted.

## Analysis outputs

- [ ] One row for each of the exact 32 planned run IDs
- [ ] Eight complete four-treatment blocks
- [ ] Block-paired primary contrast under nominal network
- [ ] Block-paired contrast under impairment and interaction contrast
- [ ] All individual coverage traces and paired differences
- [ ] Registration errors, gate decisions, wrong commits and safety events
- [ ] Communication loss and post-reconnection recovery timeline
- [ ] `claim_authorized` remains `false` pending independent review
