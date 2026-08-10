# Prospective v2 locked protocol

## Status and claim boundary

This package is infrastructure for a new experiment.  It does not reconstruct
or authenticate the manuscript trainer, 5,385/1,356 split, checkpoint, seeds,
or model-selection history.  No v2 sample generation, training, validation, or
test evaluation has been run.

The numeric prefix before `_` is an **operational grouping rule only**.  It has
not been verified as semantic building identity.  The only permitted claim is
“operational-group-disjoint”; “building-disjoint” is forbidden unless external
identity evidence is later supplied without changing this experiment.

## Immutable lock

`v2_lock.json` binds the following through SHA-256:

- all 156 source `GT.bmp` byte streams and available `GT.json` metadata;
- the 146 operational groups and fixed 102/22/22-group,
  110/23/23-floorplan train/validation/test assignment;
- the exact Python environment, installed distributions, GPU identity, driver,
  CUDA and library versions;
- every generator, validator, gate, trainer, statistics, model, and runner
  source file;
- recovered positive-control A under the single logical filename
  `recovered_candidate_deconv_a.pt`, SHA-256
  `da4458514656d41fba0e0ce6d4f4967997ff0a97f2e905a458757609edf3a3a8`.

The positive control is validation-only and is not a manuscript checkpoint.
The lock and sidecar digest are read-only.  Every downstream artifact stores
the lock SHA, source-split SHA, dataset-manifest SHA, configuration SHA, and
parent-checkpoint SHA where applicable.

## Fully specified geometry

The machine-readable lock is authoritative.  It fixes image decode mode,
threshold and polarity, minority-mask content-box logic, padding, nearest
resampling and rounding, 16 px/m working grid, 3-iteration obstacle dilation,
component connectivity/scoring/tie-break, distance-transform convention,
10–80% start box, 2-cell clearance and fallback, start/heading distributions,
30 m/480 raw field, oriented-patch size/centre/crop and rotation equation,
border values, 360 ray angles/rounding/de-duplication, 15 m/240-cell
first-hit-inclusive scan, exact action and scan schedule, collision behavior,
known-mask dilation, RGB channel semantics, save inequalities and attempt
ordering, 480→256 nearest resize, PNG encoder, and SHA-derived sample seeds.
None is a runtime option.

Smoke requests four samples per train/validation floorplan with 160 attempts.
Full generation requests 48 with 1,920 attempts.  Test requests zero in both.

## Atomic five-gate decision

The campaign advances only if one immutable audit atomically records PASS for
all five gates:

1. exact hashes, one-hot and known-target agreement, and zero cross-split
   operational-group/source/pair leakage;
2. at least 90% complete floorplan coverage in both train and validation;
3. locked observation/target marginal tolerances against the preserved
   train-only reference;
4. positive-control A validation F1 ≥ 0.45 and IoU ≥ 0.30;
5. seed-101 teacher smoke within 30 epochs, finite metrics, ≥5% validation BCE
   reduction from epoch 1, and validation F1 ≥ 0.05 at threshold 0.5.

The audit also verifies that the validation summary, positive control, smoke
teacher run manifest, metrics, completion record, and both selected/last
checkpoints bind to one lock and one dataset manifest.  Missing, malformed,
non-finite, hash-mismatched, or test-tainted input is an error.  Any failed
gate writes `NO_GO` and stops before full generation.
Thresholds cannot be changed after the smoke exists.

## Full training and checkpoint selection

After `GO`, the runner requires 100% full-dataset train/validation coverage and
integrity.  It then trains:

- one teacher, seed 101;
- five students, seeds 11, 23, 37, 53, and 71, all distilled from the same
  SHA-bound selected teacher checkpoint.

All use FP32, batch 16, deterministic kernels, and Adam (lr 0.001, betas
0.5/0.999, epsilon 1e-8, zero weight decay, AMSGrad/foreach/fused/capturable/
differentiable all false), at most 500 epochs, no stopping before epoch 50,
then patience 30.  Each run
selects the minimum validation unknown-region BCE; exact ties select the
earliest epoch.  Test never participates.

Every checkpoint binds lock SHA, dataset manifest SHA, source split SHA,
training-configuration SHA, stage, seed, epoch, selected validation metric,
and parent-teacher SHA.  A checkpoint lacking any binding is invalid.

The experimental unit for primary uncertainty is the student seed.  All five
seeds are included; exclusion and best-seed reporting are forbidden.  For each
validation metric report mean, sample standard deviation (ddof 1), two-sided
95% t interval with df=4, median, minimum, and maximum.  Spatial clustering, if
reported, uses the operational prefix group and must not be called a verified
building cluster.

## Single execution path and test prohibition

The only supported entrypoint is `run_v2_locked_campaign.py`.  Its CLI accepts
paths only; it exposes no seed, sample, attempt, geometry, epoch, optimizer,
hash-skip, resume, or test switches.  It verifies the environment, source
bytes, source files, split, and positive control before doing work.  It then
executes the locked sequence and stops on the first failure.

Test rows may be assigned and raw bytes hashed for the lock, but no program in
the package may decode a test image, generate a test sample, compute a test
metric, or expose a test entrypoint.  Test remains disabled even after full
five-seed validation completes.  Enabling it requires a future, separately
authorized protocol rather than a flag in this runner.
