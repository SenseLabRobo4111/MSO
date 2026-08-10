# Reconstructed matched closed-loop study

This directory contains a new, headless, three-robot simulation experiment. It is not the historical ROS experiment and it is not evidence from a physical robot deployment. Its purpose is to test one narrow causal question under controlled conditions: when every random stream and every measured observation is paired, does provisional map prediction improve closed-loop exploration relative to an observed-only control?

## Prospective draft contract

`protocol.json` is a prospective draft for the two arms, team size, five seeds, 1,500-tick horizon, building-disjoint split, endpoints, and building-level analysis. It is not preregistered or frozen and remains validation-only until replacement-training geometry is resolved. The sealed held-out set contains 23 floors from 22 buildings. Train/validation preparation uses only held-out directory identifiers and resolution metadata to verify membership; it does not open held-out bitmap pixels.

The causal contrast changes only the predictor:

- `prediction_on`: provisional predictions may enter registration inputs and frontier ranking.
- `observed_only`: unknown cells remain neutral in the same places.

Team starts and private frames are randomized without the arm name and their digest must match across the pair. Sensor ray casting, motion/collision transitions, and encounter scheduling are deterministic shared components and therefore declare no unused random streams. Measured cells always override prediction. Prediction can never create traversable cells, enter the persistent map, or change a commit directly.

The executed chain is:

`world/scan -> predictor or observed control -> encounter registration -> observed-support gate -> temporal/cycle consistency -> atomic measured commit -> persistent team map -> measured-safe frontier planning -> motion`

The building assignment and GT loader are reused from `training_recovery/generate_floorplan_dataset.py`. Registration and atomic measured commit are reused from `integrated_offline/reference_registrar.py` and `integrated_offline/run_paired_ablation.py`.

## Predictor geometry

Simulator resolution, physical prediction-window side, training raw side, and model input side are separate protocol fields. The current 40 m, 640-pixel, and 256-pixel values are configurable smoke defaults only; they are neither a historical match nor an evidence-backed test contract. The checkpoint adapter resizes categorical measured occupancy to the configured model side with nearest-neighbour interpolation, runs the frozen model, maps probabilities explicitly back to the world-crop shape, and re-imposes measured free/occupied cells. Held-out execution and freezing remain disabled while the replacement-training geometry and polarity are being finalized.

The deterministic predictor and identity registrar are smoke-test fixtures only. They are rejected by the held-out runner. A future validation-selected `best.pt` can be connected through the existing checkpoint adapter after its geometry and polarity contract is frozen.

## Audit trail

Each run writes:

- `run_manifest.json`: protocol digest, source digest, predictor digest, exact parameters, random-stream seeds, initial map/start/private-frame digest, and truth-access boundary;
- `events.jsonl`: canonical, append-only stage records with previous-event hashes;
- `summary.json`: persistent-map quality, root-pose error, distance, collision, mobility, registration, recovery, and runtime endpoints.

Each complete campaign also writes `building_cluster_analysis.json`. It contains complete arm-pair checks, building-level effects, the two-sided randomization tests, building-cluster bootstrap intervals, Holm correction over the two primary endpoints, censoring counts, and joint descriptive safety metrics.

Stage events include input/output digests, candidate transforms, observed-support gate decisions, temporal/cycle decisions, map revisions and state hashes, planner routes, and motion results. Any rejected commit is asserted to preserve both map revision and map hash. GT is used only for initialization, sensor/motion simulation, stopping, and post-decision evaluation of root-pose and persistent-map quality. It is unavailable to the predictor, registration, gates, recovery, commits, and planning.

## Safe workflow

Run the contract tests from the repository root:

```bash
PYTHONPATH=MSO_public python -m pytest -q -p no:cacheprovider MSO_public/closed_loop_reconstructed/tests
```

Run a short validation smoke without opening held-out bitmaps:

```bash
PYTHONPATH=MSO_public python -m closed_loop_reconstructed.run_campaign \
  --source-root /path/to/record_dataset \
  --output-root /tmp/mso_closed_loop_val_smoke \
  --split val --limit-floorplans 1 --seeds 11 \
  --arms prediction_on observed_only \
  --predictor fixture --registrar reference --maximum-ticks 20
```

After the replacement-training geometry audit is complete, explicitly enable freezing in the protocol and then freeze all relevant software:

```bash
PYTHONPATH=MSO_public python -m closed_loop_reconstructed.freeze_protocol \
  --source-root /path/to/record_dataset \
  --checkpoint /path/to/best.pt \
  --training-manifest /path/to/training_manifest.json \
  --selection-trace /path/to/selection_trace.json \
  --environment-lock /path/to/environment.lock
```

This writes `FROZEN_PROTOCOL.sha256`, `SOFTWARE_SHA256SUMS.json`, and `FROZEN_MANIFEST.json`. Test execution additionally requires a deliberate unlock JSON containing:

```json
{
  "allow_test": true,
  "protocol_sha256": "<frozen protocol digest>",
  "software_manifest_sha256": "<frozen software-manifest digest>",
  "checkpoint_sha256": "<checkpoint digest>",
  "training_manifest_sha256": "<training-manifest digest>",
  "selection_trace_sha256": "<selection-trace digest>",
  "environment_lock_sha256": "<environment-lock digest>"
}
```

Freezing is a deliberate future action: it content-hashes held-out source files, the selected checkpoint, training manifest, validation-selection trace, independent environment lock, protocol, and every listed software file. At execution, the runner rehashes the files themselves, not just the software manifest. It refuses fixtures, altered seeds or arms, shortened horizons, selected floors, nonempty output directories, changed bytes, or a missing/mismatched unlock. Do not freeze or run the held-out split until replacement training, checkpoint selection, and all validation decisions are complete.

## Prospective analysis plan

The primary endpoints use the persistent team map, not the union of simulator-seen cells: correctly registered known-cell coverage AUC versus cumulative team distance and distance to 80% correctly registered known-cell coverage. Raw known coverage, semantic accuracy, class precision/recall, off-domain pollution, and contaminated area remain separately visible. The value at the fixed 180 m budget is linearly interpolated when observations bracket the budget and otherwise carried forward from the last in-budget observation; a post-budget value is never clamped onto 180 m. Distance to 80% is interpolated within the budget or right-censored and restricted to 180 m. Arms are paired by building, floorplan, and seed. Seeds/floors are repeated measurements; inference uses building-averaged arm differences, two-sided paired sign randomization, building-cluster bootstrap intervals, and Holm family-wise correction across the two primary endpoints.

Safety is reported jointly and descriptively; it does not make a standalone inferential claim from collision rate. The record includes episode collision count, attempts and attempt rate, no-route, replans, stuck episodes, distance, persistent-map coverage, and explicit null values for zero denominators. Gate-level wrong accepts are separate from final-decision false accepts/rejects. Registration correctness is evaluated after the decision against the actual committed root transform at 0.25 m translation and 5 degrees absolute yaw. Cascaded errors from an incorrect target root are unsafe; an erroneous committed transform must later be invalidated with a measured-map rebuild or the run is marked a safety failure.

Validation smoke or fixture output supports implementation integrity only. It does not establish a predictor benefit, a held-out causal result, a historical reproduction, a large-team physical result, or robustness to impaired communication. Those claims require the frozen checkpoint and complete paired held-out campaign, with no protocol changes after unsealing.
