# Recovered-candidate validation-only protocol

Frozen before evaluation: 2026-08-07

This protocol compares two recovered 342,771-parameter deconvolution-generator
candidates on the complete validation split of the new building-disjoint
dataset.  It does not authenticate either candidate as a manuscript checkpoint
and it does not access the test split.

The candidates are:

- `recovered_candidate_deconv_a.pt`, SHA-256
  `da4458514656d41fba0e0ce6d4f4967997ff0a97f2e905a458757609edf3a3a8`;
- `recovered_candidate_deconv_label_b.pt`, SHA-256
  `022689a97336b47fe0c3a39d85e120a4d07e99ba3cc34e4f2be5422c1e515e2c`.

The primary comparison metric is validation unknown-region binary
cross-entropy, minimized.  Fixed-threshold unknown-region F1 and IoU use 0.5.
Unknown-region PSNR and full-image PSNR are computed from mean squared error;
full-image SSIM uses `skimage.metrics.structural_similarity` with
`data_range=1`.  All 1,104 validation samples and their manifest-declared image
hashes must be verified.  Candidate and manifest SHA-256 values are written
with the results.

The lower-BCE candidate may be suggested by this validation protocol, but the
comparison does not authorize test evaluation.  Any eventual test evaluation
must occur only once, after the selected artifact and its SHA-256 are frozen.
