# Training provenance audit and replacement-study workspace

Historical provenance remains unresolved.  The original trainer, exact split,
paper checkpoint, model-selection log, and seed-level run records have not
been authenticated.  The negative historical result is frozen in
[`HISTORICAL_FORENSIC_FINDINGS.md`](HISTORICAL_FORENSIC_FINDINGS.md).

The first replacement design in this directory is now a
**geometry-mismatched pilot**.  Its direct 256-cell, 0.04 m/cell generator does
not match the preserved train-only archive's raw 640-cell geometry or class
marginals.  The sequential campaign was paused before student training and
before test evaluation.  Its artifacts are retained only as negative evidence;
they are not paper results and must not be resumed as the preregistered study.

A HouseExpo study with explicit source house IDs is restricted to
train/validation-only source, renderer, and geometry diagnostics.  Its target
marginals do not match the preserved archive, so it is not the forward
replacement.  The sole prospective v2 candidate returns to the 156 KTH source
floorplans; its 23-floorplan test partition remains sealed.  No long v2
training has been authorized.  Marginal agreement, if obtained, will not
authenticate the historical run.

## Superseded pilot design (do not use as a final protocol)

- Source data: 156 real `GT.bmp` floorplans with their original directory IDs
  and SHA-256 values.
- Grouping: the leading numeric source ID before `_` is the building ID;
  `tianda` is a singleton named building.  This yields 146 building groups.
- Split seed: 20260807.
- Split: 102/22/22 buildings and 110/23/23 floorplans for train/validation/test.
- Samples: 48 deterministic limited-observation samples per floorplan, giving
  5,280/1,104/1,104 samples (7,488 total).
- Leakage gates: building, floorplan, exact source-GT hash, and exact processed
  observation-target pair must have zero cross-split overlap.
- Pilot model window: 256 by 256 cells at 0.04 m/cell; source 0.10 m/cell maps
  were nearest-resampled before limited-observation simulation.  Geometry and
  class-marginal audit later invalidated this as a final design.
- Selection: minimum validation unknown-region BCE.  The test split is not
  evaluated during training.
- Seeds: one teacher at 101; five students at 11, 23, 37, 53, and 71.
- Precision: FP32.  The model's FFT layer does not accept BF16 in the locked
  PyTorch environment, so no implicit mixed-precision fallback is used.
- Schedule: at most 500 epochs, no stopping before epoch 50, then patience 30.

The pilot loss is declared in `mso_recovery/objective.py`: unknown-region
BCE, unknown-region soft Dice, full-image BCE, and (for the student) four-tap
feature distillation.  It is not claimed to be the missing historical loss.

## Reproduce the superseded pilot only

```bash
python training_recovery/generate_floorplan_dataset.py \
  --source-root /path/to/record_dataset \
  --output-root /new/path/to/dataset \
  --samples-per-floorplan 48

python training_recovery/validate_generated_dataset.py \
  --dataset-root /new/path/to/dataset \
  --manifest /new/path/to/dataset/manifest.csv \
  --expected-buildings 146 \
  --expected-floorplans 156 \
  --expected-samples 7488 \
  --verify-images
```

The generator refuses a non-empty output path.  It writes a sample manifest,
the building/floorplan assignment, source and generated hashes, coverage
metadata, and a summary.  A generation-interrupted marker remains if the run
does not finish.

## Pilot trainer (not authorized for a new campaign)

```bash
python training_recovery/train_reconstructed.py \
  --stage teacher \
  --dataset-root /path/to/dataset \
  --manifest /path/to/dataset/manifest.csv \
  --output-dir /new/path/to/teacher_seed_101 \
  --seed 101

python training_recovery/train_reconstructed.py \
  --stage student \
  --teacher-checkpoint /path/to/teacher_seed_101/best.pt \
  --dataset-root /path/to/dataset \
  --manifest /path/to/dataset/manifest.csv \
  --output-dir /new/path/to/student_seed_11 \
  --seed 11
```

Each run writes `run_manifest.json`, append-only `metrics.csv`, validation-
selected `best.pt`, resumable `last.pt`, and `completion.json` with artifact
hashes.  A non-empty run directory is rejected unless an explicit compatible
resume checkpoint is supplied.

## Pilot evaluator (test remains sealed)

```bash
python training_recovery/evaluate_reconstructed.py \
  --dataset-root /path/to/dataset \
  --manifest /path/to/dataset/manifest.csv \
  --checkpoint /path/to/student_seed_11/best.pt \
  --output-dir /new/path/to/student_seed_11_test
```

The evaluator exists for a future frozen replacement protocol, but the pilot
test split was not evaluated.  Do not run it on pilot artifacts.  When a new
protocol is frozen, evaluation must check checkpoint/manifest binding and may
open each held-out test exactly once after validation-only selection.

`evaluate_recovered_candidates_validation.py` is a separate validation-only
path for the two explicitly labelled recovered candidates.  Its frozen metric
and provenance boundary are documented in
`RECOVERED_CANDIDATE_VALIDATION_PROTOCOL.md`; it never treats those weights as
authenticated manuscript checkpoints and contains no test-split option.
The completed validation-only comparison is frozen in
`evidence/recovered_candidate_validation_v1.json`.  Both candidates transferred
poorly to the new dataset, so neither was advanced to test evaluation.

`FORENSIC_ASSIGNMENT_PROTOCOL.md` and
`evidence/preserved_to_floorplan_assignment_failure_v2.json` document the
attempt to recover source-building labels for the preserved archive.  None of
100 fixed train-only samples passed the conservative assignment gates, so the
preserved archive must not be relabelled as a building-disjoint dataset from
the current 156 source floorplans.

`inventory_houseexpo_sources.py`, `audit_houseexpo_visible_filter.py`, and
`validate_houseexpo_renderer.py` establish the proposed new source chain.  The
35,126 explicit HouseExpo IDs can be split atomically by house.  The accessible
old generator's size filter retains only 45 houses, so that filter is useful
for historical diagnostics but is not adequate evidence of broad building
generalization.  `generate_houseexpo_geometry_smoke.py` and
`run_houseexpo_official_simulator_smoke.py` are train/validation-only smoke
tools; their outputs must be labelled as new-experiment diagnostics, never as
historical recovery.

`PROSPECTIVE_V2_PROTOCOL.md` freezes the sole forward candidate and its
train/validation-only go/no-go gates.  `v2_lock/REMOTE_LOCK_MANIFEST.json`
records the read-only remote lock and its inventory, split, environment,
source, and positive-control hashes.  The candidate generator and closed
runner are syntax- and unit-tested but have not been launched; passing the
documented atomic five-gate decision is required before full generation.
There is no test decode or evaluation path in this protocol.

## Persistent campaign

`run_remote_campaign.sh` is retained to make the superseded pilot auditable.
Its parent process was paused before students and test.  Do not resume it.  A
future campaign needs a new run root and a newly frozen KTH-source protocol.
