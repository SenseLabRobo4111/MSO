# Frozen historical training forensic findings

Audit frozen: 2026-08-07

This record separates evidence recovered from the historical project from the
new replacement experiment in this directory.  A replacement run must never
be described as the historical training run.

## Findings that are directly supported

The accessible deployment archive contains four 500-epoch distillation
checkpoints and one older non-distillation checkpoint.  Metadata was inspected
without modifying the source artifacts.

| Archived filename | SHA-256 | Generator family | Epoch | Global step | Saved optimizer |
|---|---|---|---:|---:|---|
| `model-epoch=499.ckpt` | `851d1060721e145334350f6ebd926006ac047a837a581f678a4e2549b9933895` | bilinear student | 499 | 72,000 | Adam, lr 0.001, betas (0.0, 0.99) |
| `model-epoch-label.ckpt` | `062529bc64c4d26df8146eb407e2decfc289e5421ecc11af7d61bc3e0acb43ec` | bilinear student | 499 | 72,000 | Adam, lr 0.001, betas (0.0, 0.99) |
| `model-epoch-deconv.ckpt` | `0e7f949925b9f6310f93e36b10152e9729a8cacef15c9061d9e361f1b7c649ad` | deconvolution student | 499 | 72,000 | Adam, lr 0.001, betas (0.0, 0.99) |
| `model-epoch-deconv-label.ckpt` | `e16cdf189debda41ffa116173f4e8ac8f441afff6a01c02cc15effe361cb6ea5` | deconvolution student | 499 | 72,000 | Adam, lr 0.001, betas (0.0, 0.99) |

The two deconvolution hashes also identify the released recovered candidates
in `repro_reconstructed/`.

All four 500-epoch files contain generator, teacher, critic, and perceptual-loss
state.  Their progress metadata records 72 batches in an epoch, 36,000 batches
over 500 epochs, and 72,000 manual optimizer steps for two optimizers.  The
checkpoint callbacks have no monitor, no best score, no best-k entries, and no
recorded validation batches.  Therefore these files do not contain a
validation-based model-selection trace.

The archived deployment entry instantiates the bilinear `DistillMapNet` and
loads `model-epoch-label.ckpt`.  It does not instantiate the 342,771-parameter
`DistillMapNetDeconv`.  This proves which file that preserved deployment path
loads; it does not prove which checkpoint produced a manuscript table.

The inspected model source has SHA-256
`669f5e7d851ddb2e830a9db95860a5ab959d15da24395cb94987fe7f1ac6a023`.
The older reachable training sources have the following SHA-256 values:

- `main_unet_gan.py`: `b34f5a9a1a17b7572d3da9a759d07323010ed8aa8d2119d0990e006794db9407`
- `train_gan_new.py`: `b85776be20f0869da9e05151d2c088e98f07b86776b09dd4b1aa8b3bd5774cef`
- `lightning_model.py`: `c1cd9f4ed3ee6d1ec33c920cb73a3bd303aa8c179facdd1d6754f625e5949d6b`
- `dataset.py`: `80ec07b2dabdbea78e4ba2940073c54c36447743a1c7f121a80cf27d78bc214f`
- older model implementation: `f41cf8e8f780a381de8c0bfb18a8a710d44be569ec3400cc400d74464be569d9`

Those sources train different model paths and do not implement the recovered
teacher/student distillation state.  The missing historical loss modules are
also not present.

## Dataset evidence

The closest preserved sample archive contains 6,841 `train` directories and
1,463 `test` directories (8,304 total).  Its archive SHA-256 is
`7e16b8cf29b19a47e6e03589cecf7949f96be3cf10a1c7eba342e06f492463a3`.
It contains no sample-to-building manifest and does not match the reported
5,385/1,356 split.

A second archive contains only 132 training samples and no test partition.  Its
SHA-256 is
`75667300b078077cdb4b109e1b4c516297ea7d9e25f450867d7efa0941efcfda`.
Neither archive contains the missing split declaration.

The numeric prefix in preserved sample IDs is a worker identifier produced by
the old generator.  It is not evidence of a building identity and was not used
to manufacture a semantic split.

## Incompatibilities with the manuscript protocol

For 5,385 training examples on one GPU with batch size 16, an epoch has
`ceil(5385 / 16) = 337` batches unless an explicitly documented sampler changes
the definition.  The recovered 500-epoch files record 72 batches per epoch.
Their saved Adam betas are `(0.0, 0.99)`, whereas the manuscript protocol states
`(0.5, 0.999)`.  No recovered metadata reconciles either discrepancy.

Consequently, the recovered files cannot be certified as products of the
reported single-GPU, batch-16, 5,385-sample run.  They remain useful recovered
candidates, not authenticated manuscript checkpoints.

## Exhaustive negative result

The audit searched the reachable repository histories, branches, tags, working
trees, accessible experiment-storage volumes, dataset and model archives,
shell history, bytecode strings, and editor-history locations.  It did not
recover:

- the exact teacher/student training entry used for the manuscript;
- a 5,385/1,356 sample manifest with semantic building or floorplan identity;
- the paper's five historical seed values and execution order;
- a validation metric, selection log, or best-checkpoint trace;
- a locked historical environment; or
- evidence linking a recovered deconvolution candidate to a manuscript table.

This is a negative finding about currently accessible evidence, not proof that
the assets never existed.  It rules out presenting a newly written trainer or
newly generated manifest as historical.

## Approved forward path

The defensible replacement is a new, explicitly labelled experiment with
source building IDs, a building-disjoint frozen split, declared seeds, a
validation-only checkpoint-selection rule, per-epoch metrics, hashes, and an
untouched test partition.  The generator and trainer in this directory
implement that path.  Results must be reported as new reconstructed results and
kept separate from prior manuscript values.
