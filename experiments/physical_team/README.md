# Physical N=2/3/4/5 team evidence protocol

## Status and evidence boundary

This directory contains collection, integrity, network-impairment, and analysis
tooling. It does not contain matched repeated MSO team results, a live
communication-impairment result, or a three-, four- or five-robot result. The
inspected evidence is summarised in [EVIDENCE_AUDIT.md](EVIDENCE_AUDIT.md).

The protocol is designed to collect evidence relevant to two questions after
data collection:

1. whether the online registrar estimates correct relative transforms, rejects
   wrong candidates before map mutation, and records any recovery; and
2. whether exploration behaviour is measured on simultaneous physical teams of
   two, three, four, and five robots under matched conditions.

The capture node is passive. It subscribes to audit topics and records ROS 2
messages. It never publishes motion commands, calls services, or opens action
clients.

The [prospective Tongfang 27F four-SenseBeetle campaign](tongfang27_n4/README.md)
provides a separate frozen N=4 factorial plan; a concise
[中文说明](tongfang27_n4/README.zh.md) is also available. Its canonical physical
experiment branch is `sensebeetle-n4-tongfang27`. The exact 342,771-parameter
recovered candidate is identified and verified. Its immediate field task is
three cold-reset 600-second MSO-only repetitions, each with five all-topic
bags, four synchronized robot-specific third-person videos, and one common
ROS/video/GT time axis. Collection remains explicitly blocked until the
online-stack, independent GT/reference, nominal-network, synchronization,
integration-rehearsal, and per-run audit gates pass. After each run is sealed,
the five raw bags and four raw videos must be delivered to the experiment owner
with their SHA-256 inventory; a private, access-controlled Baidu Netdisk folder
is an accepted transfer channel. That private delivery is not a public-data
release.

## Minimum campaign for the present evidence gaps

The executable plan in `config/minimum_campaign.yaml` contains 15 runs in one
locked arena:

- five N=2 reference runs;
- five N=2 runs paired by trial, start-pose set, and hardware pool to the
  reference runs, using the logged UDP loss/disconnection/recovery profile; and
- five repeated N=3 reference runs.

This is a minimum data-collection campaign, not a power calculation and not a
result. The committed plan freezes an explicit 1--15 run order and named run
identifiers. Preserve its exact bytes beside the raw data. Use one checkpoint,
software revision, arena, stopping rule, motion-limit identifier, sensing-policy
identifier, and physical robot pool throughout. N=2 paired trials must retain
the same `pair_id` and `start_pose_set_id`. Do not relabel common-canvas replay
or several namespaces on one computer as physical evidence.

`audit_minimum_campaign.py` fails closed unless every planned run passes the
physical-run audit, every impaired run contains executed (not dry-run) fault
logs inside the common sensor interval, the paired N=2 hardware/start conditions
match, and the pre-locked order is a unique 1--15 sequence. It also checks that
physical bag start times follow that sequence and always sets
`claim_authorized` to `false` pending independent review.

## Optional full N=2/3/5 scaling campaign

The generic capture, preparation, preflight and run-level audit paths also
accept N=4. The existing scaling aggregator remains intentionally locked to its
original matched N=2/3/5 contract; an N=4 study requires its own frozen campaign
plan and analysis rather than being added implicitly to that comparison.

For a separate physical scaling claim, use the same arena geometry, robot pool,
checkpoint, motion limits, sensing range, communication policy, and stopping
rule for every team size. `analyse_campaign.py` accepts no fewer than five
matched trials for each of N=2, N=3, and N=5 in one arena. Two distinct arenas
are strongly preferred. Randomise team sizes and start-pose assignments before
the campaign; keep the locked randomisation sheet with the run archive.

Each run must include:

- independent pose reference from motion capture, overhead tracking, or
  surveyed fiducials;
- raw LiDAR, odometry, local map, predicted map, velocity command, merged map,
  and transform streams for every robot;
- every registration candidate and accept/reject/commit/recovery transition;
- an independent cumulative coverage trace, not the planner's own completion
  flag;
- a continuous overhead video, a site photograph, and a robot roster with
  distinct base and LiDAR serial numbers;
- a passed preflight report and SHA-256 inventory.

For every team size, correct committed registration edges must connect the whole
physical fleet. A large number of namespaces on one computer is not accepted as
physical-team evidence.

## Instrumentation contract

The fusion stack publishes JSON on `/mso/registration_event` following
`schema/registration_event.schema.json`. Publish an event at each state change,
using the same `event_id` for the candidate, decision, and commit. A recovery
event points to the affected commit with `parent_event_id`.

An arena evaluator independent of the planner publishes JSON on
`/mso/coverage_event` following `schema/coverage_event.schema.json`. A safety
observer publishes only actual events on `/mso/safety_event`. The passive logger
writes all three streams to append-only JSONL. Reference relative transforms
are exported afterward to `evidence/reference_transforms.jsonl`, one object per
line following `schema/reference_transform.schema.json` and keyed by
`event_id`. Every row carries the frozen calibration/extrinsics identifiers,
their SHA-256 digests, and non-null translation/yaw uncertainty. Empty or
duplicate reference identifiers fail the audit. Each reference timestamp must
be a positive integer and lie within the `reference_timestamp_tolerance_ms`
window frozen in the pre-capture protocol lock around its decision and commit.

Accepted and rejected decisions are pre-commit states: both must record equal
pre/post map revisions and hashes. A rejection that changes the map hash fails.
A commit must create a newer revision and different map hash. A recovery must
identify the wrong committed event, show that exact invalid revision/hash as
its pre-state, and record a newer active revision with a different hash. Its
verification names the recovery method and independent verifier and points to a
SHA-256-addressed JSON file below the run's `evidence/` directory following
`schema/recovery_evidence.schema.json`. That file must observe the same robot
pair and active revision/hash after the recovery event. Only independently
correct committed edges, not accepted candidates, may connect the fleet in the
audit graph.

Do not deliberately commit a wrong transform into a moving fleet. Controlled
bad-transform challenges belong in a shadow map or with stationary, tethered
robots. In normal runs, a zero wrong-commit count is valid; recovery is reported
only if a wrong commit actually occurred and was independently verified.

## Configure a run

From the repository root, copy the relevant template and replace every
`SET_...` value, including the stopping-rule, motion-limit, and sensing-policy
identifiers. The committed files are intentionally non-runnable templates;
`prepare_run.py` rejects any unresolved value.

```bash
cp experiments/physical_team/config/team_3.yaml /data/protocols/arena_a_team3.yaml
```

Use `team_2_reference.yaml` for the matched N=2 reference and `team_4.yaml` for
a four-robot run. Topic names in the templates are the required canonical
interface; remap the existing robot stack to those names before collection.

Create a pre-capture run skeleton with hashed inputs. It is not described as
immutable: the completed capture becomes tamper-evident only after the final
SHA-256 inventory is generated. The checkpoint file is hashed but not copied:

```bash
python3 experiments/physical_team/scripts/prepare_run.py \
  --config /data/protocols/arena_a_team3.yaml \
  --checkpoint /models/mso.ckpt \
  --run-id mso_n3_reference_t01 \
  --site-id sigs_lab \
  --arena-id arena_a \
  --trial-id 1 \
  --operator operator_name \
  --ground-truth-source overhead_tracking \
  --reference-calibration /calibration/overhead_calibration.json \
  --calibration-id overhead_cal_v1 \
  --reference-extrinsics /calibration/robot_extrinsics.json \
  --extrinsics-id robot_extrinsics_v1 \
  --uncertainty-translation-m 0.01 \
  --uncertainty-yaw-deg 0.2 \
  --repo-root /path/to/deployed/system_repository \
  --output-root /data/mso_physical \
  --campaign-id mso_physical_minimum_v1 \
  --campaign-plan experiments/physical_team/config/minimum_campaign.yaml \
  --condition-id reference \
  --pair-id n3_trial_01 \
  --randomization-order 2 \
  --start-pose-set-id n3_poses_01 \
  --network-profile none \
  --network-profile-file experiments/physical_team/config/network_impairment.yaml
```

The deployed system repository must be clean. Its full commit identifier is
stored in the manifest; uncommitted fusion or planner changes stop preparation.
The manifest initially labels the item as an unverified physical candidate.
It also stores the exact campaign/profile, calibration, and extrinsics hashes.
Only a complete run audit can assign `evidence_class: physical`.

Run the read-only preflight after all robot and tracking nodes are live. It
checks unique hardware identities, required topics, disk capacity, absence of
simulation time, SSH reachability, NTP state, round-trip time, and clock offset:

```bash
python3 experiments/physical_team/scripts/preflight.py \
  --config /data/mso_physical/arena_a_t3_trial01/run_config.yaml \
  --output-root /data/mso_physical \
  --report /data/mso_physical/arena_a_t3_trial01/preflight.json
```

A skipped clock check intentionally fails the report and is suitable only for
diagnosing configuration.

## Communication impairment and disconnection

`scripts/network_fault.py` targets UDP traffic between two declared peer IPv4
addresses. Its default is a non-mutating dry run. It never treats a dry run as
physical evidence. The supplied profile records four phases: 30 s baseline,
45 s at 20% random UDP loss per direction, 20 s at 100% loss, and 45 s
recovery. Each endpoint installs only its own OUTPUT rule. The peer endpoint
owns the reverse direction; adding reciprocal INPUT rules would compound 20%
loss into 36% and is rejected by the audit.
The hashed profile also freezes measured per-direction acceptance intervals:
0--5% loss in baseline and recovery, 15--25% in the degraded phase, and
95--100% during disconnection. Executed rules with full packet delivery, or any
phase with no attempted traffic, therefore fail the physical-evidence check.

Live use is allowed only on a dedicated robot-data interface. Emergency stop,
operator control, localisation reference, and tracking must remain on an
independent link. First run the two reciprocal dry runs with an identical
`--start-at-ns`, `--no-wait`, and `--time-scale 0`, then validate them with
`audit_network.py --allow-dry-run`. This checks the scheduler and JSONL contract
but deliberately reports `physical_impairment_evidence: false`.

For a supervised live run, start one process on each of the two robot hosts with
the same future epoch (at least five seconds ahead), reciprocal peer IDs/IPs,
and distinct central log files. Start the bidirectional best-effort topic probes
with the same epoch and 140 s duration before the rule processes. The explicit
safety gates are mandatory:

```bash
python3 experiments/physical_team/scripts/network_topic_probe.py \
  --run-id mso_n2_impaired_t01 \
  --robot-id 0 --peer-robot-id 1 \
  --publish-topic /mso/network_probe/0_to_1 \
  --subscribe-topic /mso/network_probe/1_to_0 \
  --log-path /data/mso_physical/mso_n2_impaired_t01/events/probe_robot_0.jsonl \
  --start-at-ns 2000000000000000000 \
  --duration-s 140 --rate-hz 20 --payload-bytes 256
```

Run the reciprocal probe on robot 1 with swapped IDs/topics and its own log.
The probe uses sequenced best-effort messages and synchronized system clocks so
the audit can report per-direction sent/received counts, throughput, loss, and
one-way delay for every phase.

```bash
sudo python3 experiments/physical_team/scripts/network_fault.py \
  --profile experiments/physical_team/config/network_impairment.yaml \
  --run-id mso_n2_impaired_t01 \
  --robot-id 0 --peer-robot-id 1 --peer-ip 192.0.2.11 \
  --interface robot_data0 \
  --log-path /data/mso_physical/mso_n2_impaired_t01/events/network_events_robot_0.jsonl \
  --start-at-ns 2000000000000000000 \
  --execute --confirm-out-of-band-safety \
  --confirmation APPLY_IMPAIRMENT
```

Run the reciprocal command on robot 1, changing IDs, peer IP, interface, and log
path. Replace the documentation-only IPs and timestamp with actual locked
values. The script installs one exact OUTPUT UDP-drop rule for the peer, records
the matching `iptables-save -c` packet/byte counters before removing it, logs
every phase and command result, and removes its exact rule on normal exit or
interruption. Abort the run if cleanup does not pass; never continue a
moving-robot run to repair missing evidence.

After copying both append-only logs into the run directory, validate them before
the full run audit:

```bash
python3 experiments/physical_team/analysis/audit_network.py \
  --profile experiments/physical_team/config/network_impairment.yaml \
  --run-id mso_n2_impaired_t01 \
  --expected-robot-ids 0,1 \
  --logs /data/mso_physical/mso_n2_impaired_t01/events/network_events_robot_0.jsonl \
         /data/mso_physical/mso_n2_impaired_t01/events/network_events_robot_1.jsonl \
  --probe-logs /data/mso_physical/mso_n2_impaired_t01/events/probe_robot_0.jsonl \
               /data/mso_physical/mso_n2_impaired_t01/events/probe_robot_1.jsonl \
  --output /data/mso_physical/mso_n2_impaired_t01/analysis/network_audit.json
```

The later `audit_run.py` check additionally requires these live phase events to
fall inside the common physical sensor interval, followed by at least three
merged-map messages and three independent coverage samples in the recovery
phase. It reports post-recovery registration and safety events without imposing
a favourable-outcome threshold. An executed fault log alone does not establish
safe recovery or exploration performance.
Successful rule commands and counters are reported separately from measured
topic traffic. Missing probe logs, zero attempted traffic, duplicated endpoint
rules, wrong phase order/duration, or absent drop counters can never become
`physical_impairment_evidence`.

## Three-, four- and five-robot launch examples

Build and source the ROS 2 workspace, then launch the passive recorder. Both
arguments are required so a large bag cannot silently land on the system disk.

```bash
ros2 launch sensemap physical_team_3.launch.py \
  run_id:=arena_a_t3_trial01 \
  output_root:=/data/mso_physical
```

```bash
ros2 launch sensemap physical_team_4.launch.py \
  run_id:=arena_a_t4_trial01 \
  output_root:=/data/mso_physical
```

```bash
ros2 launch sensemap physical_team_5.launch.py \
  run_id:=arena_a_t5_trial01 \
  output_root:=/data/mso_physical
```

Start the robots only through the existing, separately supervised controller.
End collection with Ctrl-C after all robots have stopped. Preserve the normal
human emergency-stop procedure and bounded test arena; this package does not
replace either.

## Complete and audit a run

Place the continuous overhead video and site photograph at the exact paths
declared in the run config. Export independent reference transforms, then run:

```bash
python3 experiments/physical_team/analysis/audit_run.py \
  --run-dir /data/mso_physical/arena_a_t3_trial01
```

The diagnostic audit fails closed if any robot lacks physical sensor messages, if the
common sensor interval is too short, if `/clock` is recorded, if an external
pose reference is missing, if the correct committed edges do not connect the
fleet, if a wrong commit lacks recovery, or if evidence files and hashes are
incomplete. It validates the manifest and every registration, reference,
recovery-evidence, safety, coverage, network, and traffic-probe JSON object
against the checked-in schemas. Successful outputs are written below the run's
`analysis/` directory.

After collecting matched trials for all three team sizes:

```bash
python3 experiments/physical_team/analysis/analyse_campaign.py \
  --campaign-root /data/mso_physical \
  --output-dir /data/mso_physical/campaign_analysis \
  --min-trials-per-team 5
```

`run_audit.json` reports `protocol_diagnostic_pass`, and
`campaign_audit.json` reports `campaign_protocol_complete`. These fields only
describe whether the currently implemented checks completed; both reports set
`claim_authorized` to `false`. They do not certify a manuscript claim because
the current auditor has not yet undergone the independent validation required
for claim authorization. The generated tables remain useful for inspection of
run counts, final coverage, time to 80% coverage, coverage AUC,
translation/yaw errors, accepted/rejected encounters, false accepts, false
rejects, and wrong commits.

For the 15-run minimum campaign, use the locked copy of the plan instead:

```bash
python3 experiments/physical_team/analysis/audit_minimum_campaign.py \
  --campaign-root /data/mso_physical \
  --plan /data/mso_physical/mso_physical_minimum_v1/minimum_campaign.yaml \
  --network-profile /data/mso_physical/mso_physical_minimum_v1/network_impairment.yaml \
  --output-dir /data/mso_physical/mso_physical_minimum_v1/analysis
```

Protocol smoke tests use only transient synthetic dry-run logs:

```bash
python3 -m unittest discover \
  -s experiments/physical_team/tests -p 'test_*.py' -v
```

Passing these tests verifies fail-closed tooling behaviour; it is not a robot
experiment.
