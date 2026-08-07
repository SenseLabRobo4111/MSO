# Physical N=2/3/5 team evidence protocol

## Status and evidence boundary

This directory contains collection, integrity, and analysis tooling. It does
not contain three- or five-robot results. The inspected evidence is summarised
in [EVIDENCE_AUDIT.md](EVIDENCE_AUDIT.md).

The protocol is designed to collect evidence relevant to two questions after
data collection:

1. whether the online registrar estimates correct relative transforms, rejects
   wrong candidates before map mutation, and records any recovery; and
2. whether exploration behaviour is measured on simultaneous physical teams of two,
   three, and five robots under matched conditions.

The capture node is passive. It subscribes to audit topics and records ROS 2
messages. It never publishes motion commands, calls services, or opens action
clients.

## Minimum defensible campaign

Use the same arena geometry, robot pool, checkpoint, motion limits, sensing
range, communication policy, and stopping rule for every team size. The minimum
accepted by `analyse_campaign.py` is five matched trials for each of N=2, N=3,
and N=5 in one arena. Two distinct arenas are strongly preferred for a journal
claim. Randomise the order of team sizes and start-pose assignments before the
campaign; keep the randomisation sheet with the run archive.

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

For N=5, correct accepted registration edges must connect all five robots. A
large number of namespaces on one computer is not accepted as physical-team
evidence.

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
`event_id`.

Do not deliberately commit a wrong transform into a moving fleet. Controlled
bad-transform challenges belong in a shadow map or with stationary, tethered
robots. In normal runs, a zero wrong-commit count is valid; recovery is reported
only if a wrong commit actually occurred and was independently verified.

## Configure a run

From the repository root, copy the relevant template and replace every
`SET_...` value. The committed files are intentionally non-runnable templates;
`prepare_run.py` rejects any unresolved value.

```bash
cp experiments/physical_team/config/team_3.yaml /data/protocols/arena_a_team3.yaml
```

Use `team_2_reference.yaml` for the matched N=2 reference. Topic names in the
templates are the required canonical interface; remap the existing robot stack
to those names before collection.

Create an immutable run directory. The checkpoint file is hashed but not
copied:

```bash
python3 experiments/physical_team/scripts/prepare_run.py \
  --config /data/protocols/arena_a_team3.yaml \
  --checkpoint /models/mso.ckpt \
  --run-id arena_a_t3_trial01 \
  --site-id sigs_lab \
  --arena-id arena_a \
  --trial-id 1 \
  --operator operator_name \
  --ground-truth-source overhead_tracking \
  --repo-root /path/to/deployed/system_repository \
  --output-root /data/mso_physical
```

The deployed system repository must be clean. Its full commit identifier is
stored in the manifest; uncommitted fusion or planner changes stop preparation.

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

## Three- and five-robot launch examples

Build and source the ROS 2 workspace, then launch the passive recorder. Both
arguments are required so a large bag cannot silently land on the system disk.

```bash
ros2 launch sensemap physical_team_3.launch.py \
  run_id:=arena_a_t3_trial01 \
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
pose reference is missing, if the correct accepted edges do not connect the
fleet, if a wrong commit lacks recovery, or if evidence files and hashes are
incomplete. Successful outputs are written below the run's `analysis/`
directory.

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
