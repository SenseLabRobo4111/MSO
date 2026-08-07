# Reconstructed MSO training and evaluation package

This directory is a forward-looking, auditable reconstruction. It is not the
historical training release and it does not identify either recovered weight
file as the checkpoint used for any manuscript table.

The package provides four things that can be verified independently:

1. two generator-only state dictionaries extracted from recovered full
   checkpoints that strictly load into `DistillMapNetDeconv`;
2. a sample-level inventory of the preserved 6,841/1,463 archive without
   copying its images;
3. tools for constructing a new group-disjoint split from explicit semantic
   metadata; and
4. a runnable PyTorch training/evaluation path whose reconstruction choices
   are declared below and in `PROVENANCE.md`.

## Status boundaries

- `checkpoints/recovered_candidate_*.pt` are **recovered candidates**.
- `manifests/preserved_archive_6841_1463.csv` describes the preserved archive,
  not the manuscript's reported 5,385/1,356 split.
- Files produced by `tools/build_split_manifest.py` are **reconstructed
  canonical splits**. They become suitable for new experiments only after an
  author supplies semantic building/floorplan/scene metadata.
- `train.py` implements a transparent reconstruction. It does not recreate the
  missing historical trainer or certify the manuscript's reported metrics.

## Verify the recovered candidates

Run from the repository root:

```bash
python repro_reconstructed/tools/verify_artifacts.py
python repro_reconstructed/tools/checkpoint_inspect.py \
  repro_reconstructed/checkpoints/recovered_candidate_deconv_a.pt
```

Verification checks the published SHA-256 values, loads each state dictionary
strictly, and confirms 342,771 trainable parameters.

## Preserved archive inventory

The committed CSV contains one row per preserved sample with:

- relative sample ID and on-disk archive partition;
- source worker prefix (recorded only as provenance, never as a building ID);
- raw observation and target SHA-256 values;
- a processed-pair SHA-256 after the documented model preprocessing; and
- original and processed shapes.

Regenerate it on a machine holding the archive:

```bash
python repro_reconstructed/tools/inventory_dataset.py \
  --dataset-root /path/to/SenseMapDatasets \
  --output repro_reconstructed/manifests/preserved_archive_6841_1463.csv \
  --summary repro_reconstructed/manifests/preserved_archive_6841_1463.summary.json
```

The inventory command records existing directory labels in both `split` and
`archive_partition`; every row carries
`split_status=preserved_archive_label_not_manuscript_split`. It does not turn
those labels into a new scientific split.

## Build a reconstructed group-disjoint split

Create an author-verified metadata CSV with these columns:

```text
sample_id,split,group_key,group_kind
train/0_0,train,KTH_building_01,building
test/16_0,test,KTH_building_21,building
```

Then run:

```bash
python repro_reconstructed/tools/build_split_manifest.py \
  --inventory repro_reconstructed/manifests/preserved_archive_6841_1463.csv \
  --metadata /path/to/author_verified_groups.csv \
  --output /path/to/canonical_split.csv \
  --summary /path/to/canonical_split.summary.json
```

The tool requires a semantic group key, rejects worker/thread/process group
kinds, rejects groups or exact processed duplicates spanning multiple splits,
and rejects metadata in which the proposed group key merely reproduces the
worker prefix. There is no option to infer a building from a sample name.

## Reconstructed training entry

Install the tested dependency set from `requirements-lock.txt`. Train a
reconstructed teacher first, then the student:

```bash
python repro_reconstructed/train.py \
  --stage teacher \
  --dataset-root /path/to/SenseMapDatasets \
  --manifest /path/to/canonical_split.csv \
  --output-dir /path/to/run_teacher

python repro_reconstructed/train.py \
  --stage student \
  --teacher-weights /path/to/run_teacher/teacher_last.pt \
  --dataset-root /path/to/SenseMapDatasets \
  --manifest /path/to/canonical_split.csv \
  --output-dir /path/to/run_student
```

Both training commands verify every selected raw image against the manifest
SHA-256 values before starting. `--skip-raw-hash-verification` is a
diagnostic-only bypass; any resulting model is marked
`dataset_integrity=not_verified_diagnostic_bypass`.

The default reconstructed seed is 0. Use `--seed N` for each declared
multi-seed run. The historical identities of the manuscript's five seeds were
not recovered; if seeds 0--4 are used for a new five-run study, report them as a
reconstructed choice rather than as the historical seeds.

The reconstructed objective uses masked adversarial BCE, a declared
multi-scale reconstruction term, masked pixel L2, and four feature-distillation
terms. These are functional replacements for unavailable historical loss
modules; they are not asserted to be numerically identical to the original
implementation.

The defaults intentionally follow optimizer settings embedded in the recovered
candidate files: Adam learning rate 0.001 and betas `(0.0, 0.99)`. The batch
size and all objective weights remain explicit command-line/config values.

Use `--dry-run` to validate the data/model path without updating weights.

## Evaluation entry

```bash
python repro_reconstructed/evaluate.py \
  --dataset-root /path/to/SenseMapDatasets \
  --manifest /path/to/canonical_split.csv \
  --split test \
  --weights repro_reconstructed/checkpoints/recovered_candidate_deconv_a.pt \
  --output /path/to/evaluation.json
```

Evaluation also verifies raw image hashes by default. The same diagnostic
bypass is available, and changes the result status to
`declared_artifact_evaluation_unverified_input`.

This entry reports PSNR and SSIM under one pinned preprocessing protocol. It
does not silently select the better candidate and does not claim to recreate
LPIPS/FID/KID or any manuscript table.

## Tests

```bash
python -m pytest repro_reconstructed/tests -q
```

The fixture tests create tiny synthetic images in a temporary directory, test
manifest construction and rejection rules, and run strict candidate loading.
