# Full-objective MSO reconstruction V4 protocol

Status: prospectively declared after the V3 checkpoint-selection audit. V4 is
not an authenticated recovery of the unavailable historical trainer or its
historical train/validation split.

## Why V4 is a new run

V3 optimises the reported adversarial, feature-reconstruction, masked-pixel
and distillation objectives but selects checkpoints by minimum validation
unknown-region BCE. The running teacher trajectory showed that this rule can
retain a conservative early model while thresholded occupied-map F1 and IoU
continue to improve. Changing the selector inside the running V3 campaign
would invalidate its frozen protocol. V4 therefore starts a new trajectory
under a separately hash-bound protocol.

## Frozen training invariants

Relative to V3, V4 keeps all of the following unchanged:

- the 6,142/699/1,463 model-data split and model-manifest SHA-256;
- the measured-authoritative known-cell target policy;
- 256 x 256 inputs, batch size 16, worker count 8 and no dropped last batch;
- teacher and student networks, widths and deterministic initialisation seeds;
- teacher seed 101 and student seeds 11, 23, 37, 53 and 71;
- the reconstructed FFC critic and frozen ADE20K ResNet feature extractor;
- adversarial, feature-reconstruction, masked-pixel and distillation equations;
- loss weights 10, 30, 1 and 5;
- Adam at 0.001 with betas (0.5, 0.999), zero weight decay and float32;
- horizontal/vertical flips and quarter-turn augmentation;
- 500 epochs with no early stopping.

The V4 trainer imports the locked V3 batch-training implementation and refuses
to run unless its SHA-256 equals the value in `full_mso_config_v4.json`.

## Validation table

All entries are micro-aggregated over every unknown pixel in the 699-sample
validation split. Positive means occupied. The decision threshold is fixed at
0.5 before training starts and is never tuned on test data.

| Stored column | Definition |
|---|---|
| `validation_unknown_tp` | Integer occupied true positives |
| `validation_unknown_fp` | Integer occupied false positives |
| `validation_unknown_fn` | Integer occupied false negatives |
| `validation_unknown_tn` | Integer occupied true negatives |
| `validation_unknown_f1` | `2 TP / (2 TP + FP + FN)` |
| `validation_unknown_iou` | `TP / (TP + FP + FN)` |
| `validation_unknown_bce` | Mean clipped binary cross-entropy |
| `validation_unknown_brier` | Mean squared occupied-probability error |
| `validation_unknown_ece15` | Occupancy-probability ECE in 15 equal-width bins |

The metrics file also retains precision, recall, unknown-pixel count, every
training-loss component, epoch duration and both selection indicators.

## Predeclared checkpoint ordering

`best_primary.pt` is selected by:

1. maximum validation unknown occupied micro-F1 at threshold 0.5;
2. if F1 is exactly tied as an integer fraction, lower unknown Brier score;
3. if both remain tied, the earlier epoch.

`best_calibration.pt` is an audit checkpoint selected by minimum unknown Brier
score, then lower unknown BCE, then the earlier epoch. It is never the parent
of a student in this protocol.

`last.pt` is the atomic epoch-boundary resume state. Generator-only immutable
snapshots are saved after every 10 completed epochs: 9, 19, ..., 499. Thus a
completed 500-epoch run contains exactly 50 periodic snapshots. The primary
checkpoint is used for latency measurements and exported predictions.

Before completion is declared, the trainer rereads all 500 rows of
`metrics_by_epoch.tsv`, independently recomputes both ordered argbest records,
and checks them against the `selection_record` embedded in each selected
checkpoint. It then writes `selection_manifest.json` with both rules, epochs,
all selected-epoch validation metrics and checkpoint SHA-256 values. The
manifest hash is bound into `completion.json` and the run `SHA256SUMS` file.
Both selected checkpoints are also checked against the complete frozen run
metadata, including stage, seed, protocol, configuration, data manifest,
source hashes, topology, container, bundle and parent-teacher provenance.
The completion schema names BCE explicitly as
`best_primary_validation_unknown_bce`; it is the BCE observed at the
F1-selected primary checkpoint and must not be interpreted as minimum BCE.

## Teacher-to-student chain

Every student must receive both the path and explicit SHA-256 of the completed
teacher `best_primary.pt`. The trainer verifies the file hash, checkpoint kind,
teacher stage, seed 101, V4 protocol identifier and all parent provenance
fields before loading it. The verified parent hash is written into every
student manifest, checkpoint and completion record. A calibration-best, last,
periodic or V3 teacher checkpoint is rejected.

The remote completion verifier receives the current campaign teacher SHA as an
explicit argument for every student. The aggregate independently requires the
same teacher SHA for all five students and cross-checks it across completion,
run-manifest and selection-manifest provenance records.

## Interpretation boundary

V4 answers whether the fully declared reconstruction performs better when its
checkpoint rule matches the occupied-map discrimination claim. It does not
recover the historical split, historical trainer, historical model-selection
trajectory or manuscript checkpoint. V3 remains reportable as the frozen
minimum-BCE branch; comparisons between V3 and V4 must be labelled as a
prospective checkpoint-protocol comparison.

## Frozen-archive ranking contract

The cross-method question is narrower than the checkpoint-selection question.
The ranking metric is the evaluator's test-set `unknown_occupied_f1` at the
fixed 0.5 decision threshold, which is one of the primary metrics locked in the
uniform archive protocol. Ranking is by descending arithmetic mean over the
complete seed set 11, 23, 37, 53 and 71. No seed may be removed, substituted or
selected after observing results, and no best-seed result is reported.

The frozen archive registers 12 baseline methods. Eleven currently have all
five seeds and must all appear in the combined table and ranking. RePaint has
zero of five completed seeds; it remains visible in the status and ranking
tables as incomplete and unranked, rather than being silently removed or
assigned a numeric result. V4 adds one complete five-seed method, producing 12
rank-eligible methods from 13 registered entries.

For every complete baseline, the aggregate records all five same-seed
differences `V4 - baseline` and reports their arithmetic mean, sample standard
deviation and two-sided 95% t interval with four degrees of freedom. These are
descriptive seed-level protocol contrasts. They do not replace the archive's
planned sample-level bootstrap analyses, do not support cross-benchmark or
universal state-of-the-art language, and do not correct for V4's longer
training budget. A rank of one may therefore be described only as first on
this frozen archive benchmark under the declared metric.

The V4 aggregate emits the selection-audit tables plus a unioned per-seed
table, explicit method status, method summary, primary-metric ranking, all 55
matched-seed differences and 11 baseline-level difference intervals. Its
verifier reads the independent selection manifest, checks the campaign teacher
SHA through every student, and never labels the primary checkpoint as a BCE
minimum.
