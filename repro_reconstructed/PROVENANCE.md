# Provenance and unresolved historical gaps

## What was recovered

Two full PyTorch Lightning checkpoints were found in a preserved deployment
copy. Their generator subtrees strictly load into the repository's
`DistillMapNetDeconv(image_size=256, dim=4)` implementation.

| Candidate artifact | Source filename | Source SHA-256 | Epoch | Global step |
|---|---|---|---:|---:|
| `recovered_candidate_deconv_a.pt` | `model-epoch-deconv.ckpt` | `0e7f949925b9f6310f93e36b10152e9729a8cacef15c9061d9e361f1b7c649ad` | 499 | 72,000 |
| `recovered_candidate_deconv_label_b.pt` | `model-epoch-deconv-label.ckpt` | `e16cdf189debda41ffa116173f4e8ac8f441afff6a01c02cc15effe361cb6ea5` | 499 | 72,000 |

Both source files are 570,448,323 bytes and report PyTorch Lightning 2.5.1.
The generator has 342,771 trainable parameters and 347,690 state values
including buffers. The candidates are not duplicates: 499 of 594 generator
tensors differ.

The source architecture files had these SHA-256 values when inspected:

- `SenseMapNet.py`: `669f5e7d851ddb2e830a9db95860a5ab959d15da24395cb94987fe7f1ac6a023`
- `ffc.py`: `d3dbbf714017d5e66ec467b2ed06ccd09053f69fc844e3c8e9b0fc56abb89f86`
- legacy `dataset.py`: `80ec07b2dabdbea78e4ba2940073c54c36447743a1c7f121a80cf27d78bc214f`

The checkpoint callback metadata embeds a historical absolute training path.
The account name is redacted from the public provenance record; the original
training directory and exact trainer were not present in the accessible
project records.

## Checkpoint evidence

Each candidate records:

- 500 attempted epochs and 72 batches in the final epoch;
- manual optimization with 72,000 optimizer steps across two optimizers;
- Adam learning rate 0.001, betas `(0.0, 0.99)`, zero weight decay;
- no learning-rate scheduler; and
- state prefixes `gen.`, `teacher.`, `critic.`, and
  `ResNetPL_criterion.`.

The embedded optimizer evidence conflicts with the current manuscript setting
of betas `(0.5, 0.999)`. The 72 batches per epoch also cannot establish the
historical sample count or batch size. This package preserves the evidence and
does not resolve the conflict by assumption.

## What was not recovered

An exhaustive search of the current repository histories, reachable branches,
tags, release assets, large-file pointers, and the accessible experiment
workstation did not recover:

- the exact historical distillation training entry;
- the missing adversarial, ResNet perceptual, and feature-matching modules used
  by the old Lightning prototype;
- a sample-level 5,385/1,356 split manifest;
- the identities and order of the five random seeds reported in the manuscript;
- proof identifying either recovered candidate as the source of a manuscript
  table; or
- a locked historical software environment.

The old public Lightning file is not the missing trainer. It uses a different
model path, Adam learning rate 0.0002, weight decay 0.001, a step scheduler, and
300 epochs.

## Preserved dataset archive

The closest preserved archive has 6,841 directories under `train` and 1,463
under `test`. It does not match 5,385/1,356. The committed inventory records all
8,304 samples without copying images.

The numeric prefix in IDs such as `2_17` is produced by the dataset generator's
`thread_id` and is therefore a source-worker field, not a building or floorplan
identifier. Four worker counts happen to sum to 1,456, leaving 5,385 samples
when removed from the preserved train directory. This arithmetic coincidence
is not evidence of the manuscript split and is deliberately rejected by the
canonical split tool.

No exact duplicate processed pair was found within or across the preserved
archive partitions during the audit. This does not prove semantic scene
disjointness.

## Reconstruction decisions

The new training path is labelled reconstructed because the exact historical
teacher wrapper and loss implementations are unavailable. It uses repository
models and explicit, inspectable replacements:

- `TeacherMapNet2` for a newly trained reconstruction teacher;
- `DistillMapNetDeconv` for the 342,771-parameter student;
- conditional BCE adversarial loss with the repository critic;
- multi-scale L1 reconstruction as the declared feature term;
- unknown-region L2; and
- mean-squared feature distillation at the four outputs exposed by both models.

Every newly trained artifact stores its configuration and the input manifest
SHA-256. Raw image hashes are verified against the manifest before training by
default; an explicit diagnostic bypass marks the resulting artifact as
unverified. The default seed is 0, and `--seed` permits declared reconstructed
multi-seed runs. A reconstructed choice such as seeds 0--4 must not be presented
as the unrecovered historical seed identities. Results produced by this route
must be labelled reconstructed and reported separately from historical
manuscript numbers.
