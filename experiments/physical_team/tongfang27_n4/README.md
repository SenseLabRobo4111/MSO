# Tongfang 27F four-SenseBeetle MSO campaign

## Status

This subtree is a prospective, fail-closed experiment package. It contains no
physical result, no 304K model binary and no claim-authorized output.

- `collection_ready: false`
- `results_status: not_collected`
- `claim_authorized: false`

The requested 304K model does not currently exist in the inspected local,
remote or archived weight collections. The manuscript's `304K` entry is an
aggregate row for the historical `w/o FFC` ablation, not a recovered artifact.
The available 342,771-parameter candidates are explicitly rejected as
substitutes. Do not rename, round or upload either candidate as this model.
The byte-level search boundary and retained parameter-count evidence are in
`MODEL_ARTIFACT_AUDIT.md`.

The public repository also lacks the complete deployed online chain. Its
predictor and passive recorder do not establish an executable

```text
predictor -> registration -> gate -> commit -> persistent map -> planner
```

system. A clean, commit-addressed external full-stack repository and a hashed
machine-readable integration report are mandatory collection gates.

The shared generic capture, N=4 wrapper and preflight are usable for passive
four-robot topic and host checks. The shared minimum-campaign preparer/auditor,
however, uses a different `condition_id` argument and manifest contract and is
not the lock authority for this factorial. This subtree uses `predictor_mode`
and `network_condition` and therefore provides its own validator and run-lock
entry point. Do not translate the factors back into the old field or declare a
five-robot run with one missing robot.

## Research questions

The campaign asks three bounded questions in one office ROI:

1. Does allowing the verified MSO predictor to influence registration proposal
   and temporary frontier ranking improve four-robot measured exploration over
   an otherwise identical observation-only system?
2. Does that effect change when one robot temporarily loses its three data-plane
   peer links?
3. Are deployed registration decisions and persistent-map commits correct with
   respect to an independent external pose reference?

Prediction may never make an unmeasured cell traversable and may never bypass
the registration gate or persistent-map transaction. In `observed_only` runs,
the model still executes and its outputs are logged in a shadow namespace to
hold compute load constant, but no registration or planner subscriber may use
them.

## Experimental design

The confirmatory campaign is a within-start-pose-block 2 x 2 factorial:

| Label | `predictor_mode` | `network_condition` |
|---|---|---|
| A | `mso_304k` | `nominal` |
| B | `observed_only` | `nominal` |
| C | `mso_304k` | `isolated_robot_impairment` |
| D | `observed_only` | `isolated_robot_impairment` |

Eight blocks each contain all four cells. Each cell therefore has eight runs
and the plan has exactly 32 confirmatory runs. The four Williams sequences are
used twice, four physical platforms rotate through four pose slots twice, and
each logical robot is the impaired robot in two blocks. The exact order is
fully enumerated in `config/campaign.yaml`; it must not be regenerated after
collection begins.

Every run records for exactly 600 seconds after the synchronized start barrier.
If the planner exhausts frontiers, the robots remain stationary and recording
continues to 600 seconds. A safety stop remains an outcome and is not silently
replaced.

### Integration pilots

`config/pilot_plan.yaml` fixes four 300-second pilots:

- `tf27_n4_pilot_A`: full pipeline and evidence logging;
- `tf27_n4_pilot_B`: prediction-shadow write barrier;
- `tf27_n4_pilot_C`: six-direction network impairment and recovery;
- `tf27_n4_pilot_D`: combined shadow and impairment barriers.

These identifiers are permanently excluded from the 32-run confirmatory
analysis. Pilot results may be used to repair instrumentation or safety faults,
but not to tune endpoints, select a favourable checkpoint or enter the main
table.

## Primary and secondary endpoints

An evaluator independent of the planner computes coverage from the frozen GT
accessible-free mask and measured observations only:

```text
C(t) = GT accessible-free cells observed by a physical sensor by time t
       ---------------------------------------------------------------
                   all GT accessible-free cells in the ROI
```

Predicted cells do not contribute to the numerator. The sole formal primary
endpoint is

```text
normalized_coverage_auc = integral from 0 to 600 s of C(t), divided by 600 s.
```

The sole formal primary contrast is the block-paired difference
`mso_304k - observed_only` under nominal networking.

Secondary, effect-size-first endpoints are:

- the same prediction contrast under impairment;
- the impairment effect within each predictor mode and the difference-in-
  differences interaction;
- final measured coverage at 600 seconds;
- time to 80% measured coverage, with non-reaching runs reported as censored;
- GT occupied precision, recall, F1 and IoU on the frozen evaluation mask;
- translation/yaw median and P95 errors for registration decisions;
- false accepts, false rejects and wrong commits;
- time to the first independently correct connected four-robot registration
  graph;
- post-reconnection time to fresh peer traffic, a new map, a correct edge and a
  connected graph;
- per-robot path length, stationary fraction and measured coverage contribution;
- collision, near-collision, emergency stop, operator intervention and planner
  abort counts.

Registration events are nested within runs. They must not be treated as
independent experimental replicates.

## GT map and independent pose reference

The supplied office GT image is sufficient for coverage and map comparison,
but not for event-level registration truth. Before any pilot:

1. freeze `tf27_gt`, the occupancy image, resolution, accessible-free mask,
   dynamic exclusion mask, door state, furniture state and geofence;
2. survey all eight four-pose sets in that frame;
3. calibrate an overhead AprilTag or equivalent tracker to `tf27_gt`;
4. measure and freeze all four tag-to-base extrinsics;
5. record external poses at a positive finite rate, preferably 10--20 Hz;
6. export `T_target^-1 T_source` at every registration decision and commit,
   using no more than the locked 100 ms join tolerance.

The GT-to-global transform is fixed before collection. Per-run ICP alignment,
result-dependent cropping and ROI edits are forbidden. External poses are not
inputs to MSO or its planner.

All GT images, metadata, masks, calibration and extrinsics must exist and match
their declared SHA-256 values. Positive finite resolution, pose rate and
reference uncertainty are independently checked; hand-filled YAML values do
not unlock collection.

## Model identity gate

`config/model_lock.yaml` deliberately describes a missing required artifact.
Readiness requires all of the following:

- an explicitly supported exact architecture loader;
- strict state loading with empty missing/unexpected key lists;
- an actual trainable-parameter count of exactly 304,000;
- a finite forward pass for a fixed `[1, 3, 256, 256]` fixture;
- artifact, fixture, training-provenance and selection-record files whose bytes
  match their lock-file SHA-256 values;
- a deterministic output fingerprint;
- a hashed machine-readable verifier report with all of the above fields.

`verify_model_artifact.py` has an intentionally empty supported-loader registry
because no matching architecture has been recovered. It currently produces
only a failed report. Adding a loader requires a reviewed source change; an
arbitrary module name from YAML is never imported. The verifier contains no
fallback to either known 342,771-parameter candidate, and the campaign
validator rejects their known artifact hashes.

Example diagnostic invocation, expected to fail until the architecture exists:

```bash
python3 experiments/physical_team/tongfang27_n4/verify_model_artifact.py \
  --artifact /models/actual_student.pt \
  --loader-id reviewed_mso_304k_v1 \
  --fixture /protocol/fixed_256_fixture.npy \
  --output /protocol/model_verification.json
```

Only the portable inference state should eventually be versioned. Do not add a
large optimizer checkpoint or hardware-specific engine as the sole model copy.

## Full online-stack gate

`config/online_stack_lock.yaml` is also unresolved. Readiness requires a clean
external repository at one exact 40-character commit, a real N=4 launch entry,
and a hashed JSON verification report establishing:

- four-robot support;
- successful full-pipeline integration testing;
- all six stages from predictor through planner;
- registration-event lifecycle output;
- persistent map revision/hash transitions;
- measured-only collision authority;
- the `observed_only` prediction-shadow write barrier.

Merely setting six YAML booleans to true does not suffice: the report bytes,
external repository HEAD, clean worktree and launch-file existence are checked.

## Network treatment gate

The impaired condition isolates one logical robot from its three peers on the
DDS data interface. At `t=180 s` the locked profile begins:

| Interval | Phase | Per-direction UDP loss |
|---|---|---:|
| 180--210 s | baseline | 0% |
| 210--255 s | degraded | 20% |
| 255--275 s | disconnected | 100% |
| 275--320 s | recovery monitor | 0% |
| 320--600 s | extended post-recovery observation | 0% |

An isolated robot has three peers and six directed paths. C/D collection is
blocked until `config/network_lock.yaml` and its hashed report establish:

- peer-addressed DDS unicast on the impaired interface;
- no multicast path bypassing peer rules;
- three peers and six executed directed rules;
- attempted and received probe traffic in all six directions;
- packet/byte counters and phase timing for all rules;
- output-only semantics with no compounded input rule;
- exact cleanup after normal exit and interruption;
- a physically independent safety/tracking/control interface.

The existing pairwise network audit cannot by itself authorize this N=4
treatment. Packet loss seen only by a synthetic probe is insufficient unless
the actual MSO DDS traffic uses the audited data plane.

## Safety and stopping rules

The template begins with conservative platform limits of 0.25 m/s linear and
0.60 rad/s angular speed and 0.35 m minimum robot separation. A platform safety
owner must approve and freeze the identifiers before pilots. Predictions never
alter the measured collision grid.

Use one observer per two robots and a third campaign controller. Every robot
must have an independent emergency stop, plus a team stop. Immediately stop a
run for a person entering the geofence, physical contact, geofence escape,
odometry/controller heartbeat loss, logging failure, unsafe map mutation,
out-of-band control loss, or a fault rule that cannot be cleaned up.

Once the start barrier is crossed, the run remains in the safety population.
For a safety or system stop, carry the last valid measured coverage forward to
600 seconds in the conservative complete-run analysis and show a separate
per-protocol sensitivity analysis. A technical-invalid run remains archived
under its original ID; a replacement receives a new ID and cannot silently
overwrite the planned record. A replacement does not automatically enter the
confirmatory analysis: doing so requires a prospective replacement rule or a
dated, independently reviewed protocol amendment made without inspecting
treatment outcomes. Otherwise the paired analysis remains incomplete.

In the endpoint CSV, `protocol_pass` means that the pre-registered evidence and
measurement contract is complete. It does not mean that the outcome was
favourable or that no safety event occurred. A safety-stopped run with intact
logs, valid GT coverage and conservative carry-forward remains
`protocol_pass=true` and retains its non-zero `safety_stops` count. A run whose
endpoint cannot be audited because evidence or calibration failed is
`protocol_pass=false` and blocks the confirmatory paired analysis.

## Evidence contract

`EVIDENCE_CHECKLIST.md` is the operator-facing checklist. At minimum record:

- each robot's raw LiDAR, odometry, measured map, provisional prediction,
  probability/logit output, prediction-shadow output, command, goal, path,
  controller state, battery and diagnostics;
- TF/TF static and independent external pose for all four robots;
- measured merged map and persistent-map revision/hash;
- every registration candidate, decision and commit with one event identity;
- independent coverage and all safety events;
- all network rules, counters, probes, DDS configuration and cleanup results;
- local bags on all robots, coordinator bag, continuous overhead video, site
  photographs and system telemetry;
- map snapshots at 0, 120, 240, 360, 480 and 600 seconds and before/after every
  commit;
- preflight report, run audit, network audit and complete SHA-256 inventory.

### Public-data boundary

The public branch contains templates, protocols, software and digests only. Do
not commit the Tongfang 27F raw floor plan, accessible-free or exclusion masks,
network addresses, hardware serial numbers, site video, site photographs or
unredacted calibration files. Keep raw evidence in controlled storage and, if
needed, a reviewer-only read-only package. Before any public release, obtain
site/data authorization, remove identifiers and network details, and publish
only an approved redacted derivative with its own licence and digest inventory.

## Validation and preparation workflow

Validate the static 32-run design. This succeeds while clearly reporting that
collection is blocked:

```bash
python3 experiments/physical_team/tongfang27_n4/validate_campaign.py
```

The mandatory field gate must fail until every required artifact, full stack,
GT/reference item, network report and pilot lock is resolved:

```bash
python3 experiments/physical_team/tongfang27_n4/validate_campaign.py \
  --require-collection-ready
```

Exit status 2 means the plan is structurally valid but collection is not
authorized. Exit status 1 means the plan itself is invalid.

This branch also contains an unconditional source-level collection gate. It
cannot be unlocked by editing a self-authored JSON report. A later reviewed
change must implement trusted executable checks for the model, full online
stack, N=4 network impairment and per-run evidence audit before any field run
can be prepared.

After all gates genuinely pass, prepare one exact confirmatory run through the
dedicated entry point:

```bash
python3 experiments/physical_team/tongfang27_n4/prepare_n4_run.py \
  --run-id tf27_n4_b01_B \
  --operator OPERATOR_ID \
  --output-root /data/tf27_n4
```

The preparer refuses unplanned IDs and non-ready locks. It builds in a temporary
directory beside the final run and renames only after every lock copy and
pre-capture digest succeeds, so a failed staging attempt cannot leave a partial
final run directory.

The preparer deliberately does not create the `bag/` directory: `ros2 bag`
must create that output target itself. Once a future reviewed change implements
collection enablement, run the shared read-only preflight against the copied
team lock before capture:

```bash
python3 experiments/physical_team/scripts/preflight.py \
  --config /data/tf27_n4/tf27_n4_b01_B/locks/team.yaml \
  --output-root /data/tf27_n4/tf27_n4_b01_B \
  --report /data/tf27_n4/tf27_n4_b01_B/evidence/preflight.json
```

Only after that report passes and an operator supervises all four robots may
the passive recorder be launched. The current protocol-only branch cannot
reach this step.

## Paired analysis

`analyse_paired_campaign.py` accepts exactly one pre-audited endpoint row for
each frozen confirmatory run. It does not inspect bags, reconstruct GT coverage
traces, apply the 600-second carry-forward rule or authorize a claim. Those
checks require a future N=4 run auditor and remain collection blockers. The
analyser rejects missing, duplicate, unplanned, metadata-mismatched or
protocol-failing rows. Required columns are:

```text
run_id,block_id,predictor_mode,network_condition,protocol_pass,
normalized_coverage_auc,final_coverage,registration_decisions,
false_accepts,false_rejects,wrong_commits,safety_stops
```

Run it only after all 32 audited rows exist:

```bash
python3 experiments/physical_team/tongfang27_n4/analyse_paired_campaign.py \
  --runs-csv /data/tf27_n4/campaign_runs.csv \
  --output-dir /data/tf27_n4/paired_analysis
```

The output directory must not already exist. Each analysis directory is
immutable and the audit records SHA-256 digests for the campaign, endpoint
table and analysis implementation.

It creates:

- `paired_differences.csv`: all five block-level contrasts for nAUC and final
  coverage;
- `paired_summary.csv`: mean/median differences, sample SD, deterministic
  block-bootstrap intervals and exact two-sided sign-flip p-values;
- `cell_summary.csv`: descriptive cell summaries and event totals;
- `analysis_audit.json`: completeness report.

The sole formal test is the predeclared nAUC prediction contrast under nominal
networking. Other outputs are secondary unless a separate multiplicity plan is
frozen before collection. No event-level pseudoreplication is used. The
analysis report always keeps `claim_authorized: false`; independent review of
the raw evidence and analysis implementation is still required.
The sign-flip calculation assumes exchangeability and symmetry of the eight
paired block differences; it is not described as a generated randomization
test. Endpoints listed elsewhere but not present in the required CSV are
planned only and remain unimplemented collection gates.

## Planned visual outputs

No result graphic is included now. After collection, preserve the source table
for every panel and generate at least:

1. four trajectories over the frozen GT office map;
2. all 32 measured coverage traces plus block-aware summaries;
3. nominal A/B paired slope plot;
4. predictor-by-network interaction plot;
5. translation/yaw error scatter and gate confusion matrix;
6. impairment timeline with loss, graph connectivity, commits and coverage;
7. final measured map, GT and categorical error maps;
8. per-robot distance/coverage contribution and safety-event table.

All figures must label pilots separately and exclude them from the confirmatory
sample size.

## Tests

```bash
python3 -m unittest discover \
  -s experiments/physical_team/tongfang27_n4/tests \
  -p 'test_*.py' -v
```

The tests use only temporary synthetic fixtures. They verify protocol
balancing, the current NO-MODEL/NO-STACK/NO-NETWORK fail-closed state, rejection
of 342,771-parameter substitutes, atomic run preparation and complete-block
paired analysis. They are not physical evidence.
