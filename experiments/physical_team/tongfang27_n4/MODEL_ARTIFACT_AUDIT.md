# 342,771-parameter deployment artifact audit

Audit date: 2026-08-11.

## Conclusion

The prospective Tongfang 27F experiment uses the complete FFC student with
exactly 342,771 trainable parameters. The locked identity is:

| Field | Value |
|---|---|
| Model ID | `mso_deconv_342771_candidate_a` |
| Campaign factor | `mso_342771` |
| Architecture | `DistillMapNetDeconv(image_size=256, dim=4)` |
| Portable artifact | `repro_reconstructed/checkpoints/recovered_candidate_deconv_a.pt` |
| Artifact SHA-256 | `da4458514656d41fba0e0ce6d4f4967997ff0a97f2e905a458757609edf3a3a8` |
| Trainable parameters | 342,771 |
| Provenance status | recovered deployment candidate; not the manuscript checkpoint |

Candidate A is present in the repository and strictly matches the architecture.
Its model-specific identity can therefore be verified. The campaign itself
remains fail-closed:

```text
collection_ready: false
results_status: not_collected
claim_authorized: false
```

No physical result follows from selecting or verifying this artifact.

## Source relationship

The portable artifact contains only the `gen.` state subtree, with that prefix
removed, from `model-epoch-deconv.ckpt`, a checkpoint retained in a preserved
deployment copy. The source record is:

| Field | Value |
|---|---|
| Source SHA-256 | `0e7f949925b9f6310f93e36b10152e9729a8cacef15c9061d9e361f1b7c649ad` |
| Source size | 570,448,323 bytes |
| Recorded epoch | 499 |
| Recorded global step | 72,000 |

The machine-readable source record is
`repro_reconstructed/checkpoints/provenance.json`. The preserved online entry
instantiated a bilinear `DistillMapNet` and loaded `model-epoch-label.ckpt`, not
this deconvolution checkpoint. Candidate A is therefore an owner-selected
prospective experiment target recovered from the deployment copy; it is not
evidence of historical runtime use and does not prove that it generated any
manuscript table or figure.

The current package implementation is locked by source hashes:

| Source | SHA-256 |
|---|---|
| `sensemap/explore_model/SenseMapNet.py` | `c84ff05fa168c2e856002f12a903e53896da70cd8854a152b85f407d176727b3` |
| `sensemap/explore_model/ffc.py` | `835fc0a0ccc67bf73c9f6317d3ecc0679a87730e26dcc622aa84e6f5c1c15454` |

In one Torch 2.12 CPU process, the retained historical implementation and the
current package implementation produced bit-identical prediction and four
feature tensors for candidate A and the frozen fixture. Their definition bodies
are AST-equivalent; byte differences are the package-relative import and added
licensing comments. This establishes implementation equivalence for the locked
reference environment. A different Torch/backend/device must reproduce that
environment or pass a separately reviewed numerical-equivalence check.

## Correction of the 304K request

The earlier request for a 304K deployment model conflated two different
objects. In the manuscript source, `304K` is a rounded resource entry for the
historical `w/o FFC` ablation. It is not the intended complete-FFC deployment
model, an architecture definition, an inference state, or a selection record.

The retained complete-FFC implementation has four FFC blocks and exactly
342,771 trainable parameters. Candidate A matches that implementation. It is
therefore used directly; it is not renamed, rounded, or presented as a 304K
artifact.

For completeness, the earlier archive audit observed 342,771-parameter states
but no exact 304,000-parameter state. That negative search remains true, but it
is no longer a deployment blocker because 304,000 was the wrong target.

## Selection boundary

Two recovered generator candidates exist:

| Candidate | Artifact SHA-256 | Source filename | Selection |
|---|---|---|---|
| A | `da4458514656d41fba0e0ce6d4f4967997ff0a97f2e905a458757609edf3a3a8` | `model-epoch-deconv.ckpt` | locked for the new experiment |
| B | `022689a97336b47fe0c3a39d85e120a4d07e99ba3cc34e4f2be5422c1e515e2c` | `model-epoch-deconv-label.ckpt` | not selected |

Candidate A was selected after the experiment owner explicitly corrected the
prospective target to the 342,771-parameter full-FFC student. Its source is the
deconvolution checkpoint retained in the preserved deployment copy, while the
preserved online entry loaded a different bilinear checkpoint. It was not
selected after inspecting validation, test, pilot or Tongfang 27F treatment
outcomes. The committed selection record freezes this choice before pilots or
confirmatory runs.

## Verification boundary

The model lock requires all of the following to agree exactly:

1. the candidate-A artifact digest;
2. the reviewed `DistillMapNetDeconv` loader and strict state loading with no
   missing or unexpected keys;
3. exactly 342,771 trainable parameters;
4. the frozen occupied/unknown/free 256 x 256 fixture digest;
5. four FFC blocks, 594 state tensors and 347,690 state values;
6. a finite five-output forward contract and reference-environment output
   fingerprint;
7. the architecture and FFC implementation source digests;
8. the provenance and pre-outcome selection-record digests; and
9. the machine-readable verifier-report digest and fields.

Hand-editing `model_lock.yaml` cannot establish these facts. The local index in
`deployment_models/README.md` identifies the fixture, selection record and
verification report used by this protocol.

Model verification is only one readiness gate. Collection remains prohibited
until the complete online chain, independent GT/pose reference, four-robot
network treatment, four integration pilots and trusted per-run evidence audit
also pass.

## Distribution boundary

The field artifact is the portable generator state, not the 570 MB optimizer
checkpoint or a hardware-specific engine. Repository presence does not expand
training-data or third-party rights. Any external release must continue to
follow `THIRD_PARTY_NOTICES.md`, `REVIEW_ACCESS_TERMS.md` and the recorded
artifact licence boundary.
