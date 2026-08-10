# Transform validation data and reproduction

This directory contains the portable event assets, reconstructed registrar,
event-level records, and evaluators used for the main transform-audit figure and
Supplementary Table 6. The registrar is an
offline reference implementation reconstructed from the manuscript
specification. It is not the deployed robot executable.

All paths in `manifest.json` and the event CSV files are relative to this
directory. Run the commands below from this directory.

## Recreate registrar events

```powershell
py run_synthetic_se2_benchmark.py manifest.json `
  --registrar manuscript_reference_registrar:register `
  --method MANUSCRIPT_RECONSTRUCTED_REFERENCE `
  --output events_manuscript_reference.csv
```

The manifest fixes 300 events from one archived scene. Each of the observed,
thresholded-prediction, and probability-preserving representations contains 50
known-transform positive events and 50 predeclared low-support temporal
mismatches. The latter are abstention challenges and do not have a unique
ground-truth pose-error label.

## Panel A and Panel B

```powershell
py evaluate_transform_events.py events_manuscript_reference.csv `
  --output-dir results_manuscript_reference `
  --tau-translation-m 0.25 `
  --tau-yaw-deg 5 `
  --cluster-field cluster_id `
  --bootstrap-reps 10000 `
  --bootstrap-seed 20260807

py summarise_manuscript_decisions.py
```

The 0.25 m and 5 degree values are analysis correctness limits. The bootstrap
resamples whole `cluster_id` groups, uses 10,000 replicates, and fixes the
pseudorandom seed to 20260807.

`transform_accuracy.csv` supplies Panel A. `table3_decision_summary.csv` supplies
Panel B and separates positive-transform correctness from low-support
acceptance. The generic evaluator's `FA` and `TR` fields classify every event
whose `gt_positive` value is false as a negative decision case. The manuscript
does not interpret those generic labels as pose errors because the low-support
events lack a unique reference transform.

For the probability-preserving representation, the expected counts are 46
correct positive accepts, one out-of-limit positive accept, three positive
rejects, zero accepted low-support cases, and 50 rejected low-support cases. Its
median translation and yaw errors are 0.047 m and 0.066 degrees.

## Panel C

Regenerate the fixed wrong-transform proposals and evaluate them with:

```powershell
py run_wrong_transform_injection.py manifest.json `
  --output wrong_transform_injection_events.csv `
  --translation-m 1.0 `
  --yaw-deg 15 `
  --gate-threshold 0.20 `
  --tau-translation-m 0.25 `
  --tau-yaw-deg 5

py evaluate_transform_events.py wrong_transform_injection_events.csv `
  --output-dir wrong_transform_injection_evaluator `
  --tau-translation-m 0.25 `
  --tau-yaw-deg 5 `
  --cluster-field cluster_id `
  --bootstrap-reps 10000 `
  --bootstrap-seed 20260807
```

The 150 proposals add 1.0 m translation and 15 degrees yaw to positive events.
Feature matching is bypassed. The harness rejects all 150 before its simulated
atomic OR-merge; it does not test the deployed commit path, rollback, or
temporal recovery.

## File guide

- `assets/`: portable target, source, and companion observed maps.
- `manifest.json`: fixed perturbations, event selection, and asset hashes.
- `run_synthetic_se2_benchmark.py`: registrar adapter runner.
- `manuscript_reference_registrar.py`: reconstructed offline registrar.
- `events_manuscript_reference.csv`: 300 registrar events used in Panels A/B.
- `evaluate_transform_events.py`: metric and cluster-bootstrap evaluator.
- `summarise_manuscript_decisions.py`: Panel B semantic summary.
- `table3_decision_summary.csv`: Panel B counts used in the manuscript.
- `run_wrong_transform_injection.py`: Panel C injection harness.
- `wrong_transform_injection_events.csv`: 150 fixed injected proposals.
- `results_manuscript_reference/`: regenerated Panel A and generic event results.
- `wrong_transform_injection_evaluator/`: regenerated Panel C results.
