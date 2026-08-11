# Locked deployment-model evidence

This directory holds the small, reproducible evidence files that bind the
Tongfang 27F protocol to one exact portable model state. The weight artifact
itself is stored once at
`repro_reconstructed/checkpoints/recovered_candidate_deconv_a.pt`.

## Identity

| Field | Locked value |
|---|---|
| Model ID | `mso_deconv_342771_candidate_a` |
| Campaign factor | `mso_342771` |
| Architecture | `DistillMapNetDeconv(image_size=256, dim=4)` |
| Trainable parameters | 342,771 |
| Artifact SHA-256 | `da4458514656d41fba0e0ce6d4f4967997ff0a97f2e905a458757609edf3a3a8` |
| Fixture SHA-256 | `a653af98e3f3ab8e23e6a7d46c013668954fa22286ef64a8800d236a215071ef` |
| Output fingerprint | `31e3cd4ec395b615a9d5dafd1024286fd9860f58acaffc1df09d23bc624778e0` |
| Architecture source SHA-256 | `c84ff05fa168c2e856002f12a903e53896da70cd8854a152b85f407d176727b3` |
| FFC source SHA-256 | `835fc0a0ccc67bf73c9f6317d3ecc0679a87730e26dcc622aa84e6f5c1c15454` |

Files in this directory:

- `fixed_fixture_occ_unknown_free_v1.npy` is the deterministic one-hot
  occupied/unknown/free input fixture;
- `selection_record.json` freezes the pre-outcome choice of candidate A; and
- `mso_deconv_342771_candidate_a.verification.json` records the version-2
  strict-load contract, exact parameter/state/FFC counts, implementation
  hashes, five-output finite forward execution and output fingerprint.

Regenerate the fixture only through the reviewed
`../build_model_fixture.py` implementation, then rerun
`../verify_model_artifact.py` and update every dependent digest atomically.

Candidate A was recovered from `model-epoch-deconv.ckpt`, which was retained in
a preserved deployment copy. The preserved online entry loaded a different
bilinear checkpoint, so this package does not claim historical runtime use.
The owner selected candidate A for this new prospective experiment; it is not
an authenticated manuscript checkpoint. These files verify model identity,
but do not make the campaign collection-ready or constitute a physical result.
