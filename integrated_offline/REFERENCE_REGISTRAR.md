# Reconstructed reference registrar

`reference_registrar.py` is a new implementation reconstructed from the
manuscript description for controlled offline diagnostics. It is not the
historical MSO matcher, does not reproduce the original 1,356/1,000-pair
experiment, and must not be used to relabel the paper's original results. Its
outputs are limited to exploratory synthetic-SE(2) saved-map audits.

## Input and output contract

The local runner calls `register(target_path=..., source_path=..., context=...)`.
The returned transform maps source pixels into target pixels
(`H_i_from_j_pixel`). Raw affine diagnostics are preserved separately.
Registration itself does not demonstrate a persistent-map commit, rollback,
temporal consensus, or post-commit recovery.

For predicted inputs, grayscale value divided by 255 is treated as an archived
occupancy probability. The companion observed grid uses 100 as unknown and
0/255 as measured free/occupied; ambiguity is forced to zero in measured cells.
For thresholded images this interpretation is only a deterministic proxy, so
`predicted_raw` is the closest available input to the manuscript probability
contract.

## Reconstruction choices

The following items follow the manuscript description:

- ambiguity `U = 0.7 p(1-p) + 0.3 S`, with normalized binary Canny response
  `S`, and `U=0` in measured cells;
- scales 1, 0.5, and 0.25;
- ORB, Canny-edge, and morphological-skeleton feature families;
- family-wise feature retention at `U(k) < 0.8 mean(U)`;
- Hamming KNN, Lowe ratio 0.9, best-80% direct fallback when fewer than eight
  ratio-test matches remain;
- manuscript exponential match weight, retaining `w_m > 0.5`;
- displacement, keypoint-scale, and rotation-consistency filtering;
- `estimateAffinePartial2D` stages at 3 px/30%, 5 px/20%, and 10 px/3%, with
  at most 5,000 RANSAC iterations.

The manuscript leaves detector counts, Canny thresholds, skeleton details,
geometric-consistency bandwidths, ORB settings, RANSAC confidence/refinement,
and a commit gate unspecified. This reconstruction fixes them as follows:
Canny 50/150; iterative 3x3 cross-element skeletonization with at most 32
iterations; ORB descriptors for all three families; family-specific matching
with pooled scales; robust median/MAD filters with 45-degree, log(1.8), and
160-pixel minimum bandwidths; RANSAC confidence 0.99 and 10 refinement
iterations. It does not perform a separate overlap-restricted full-resolution
refinement. These are documented exploratory choices, not a claim of historical
implementation equivalence.

The ground-truth-free pre-commit gate requires all of:

1. a proper near-unit SE(2) estimate, with affine singular values within 0.02 of
   one and no reflection;
2. RANSAC inlier ratio at least 0.20;
3. observed occupied-support overlap coefficient at least 0.20 after warping.

This gate is an exploratory evaluation safeguard. Ground truth is not read by
the registrar and is used only downstream for diagnostics. The manifest's
selection-lock self-declaration is not treated as preregistration.

## Local checks

From the repository root:

```powershell
py -3 -m unittest integrated_offline.tests.test_pipeline.MatcherRegressionTests -v
py -3 -m unittest integrated_offline.tests.test_pipeline.PortableBundleReplayTests -v
```

The matcher regression includes three descriptors, which forces the direct
fallback and verifies compatibility when OpenCV returns an immutable tuple.
