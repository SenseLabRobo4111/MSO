# Third-party notices and provenance boundary

This inventory records third-party material that is visibly identifiable in
the repository revision prepared for manuscript review. It is not a legal
opinion and does not establish ownership of files whose history is incomplete.

## Fast Fourier Convolution implementation

- Affected path: `sensemap/explore_model/ffc.py`
- Upstream project: Fast Fourier Convolution, official PyTorch implementation
- Upstream source: <https://github.com/pkumivision/FFC>
- Compared source: `model_zoo/ffc.py`
- Upstream licence: Apache License 2.0
- Upstream licence source: <https://github.com/pkumivision/FFC/blob/main/LICENSE>
- Local modifications visible in this revision: migration from the removed
  `torch.rfft`/`torch.irfft` interface to `torch.fft`, normalisation handling,
  and project-specific residual-network wrappers.

The upstream repository identifies its implementation as Apache-2.0. A copy of
that licence is retained at `third_party/licenses/Apache-2.0.txt`. This notice
does not imply that other MSO files are derived from the upstream project.

## ROS 2 Python package test templates

- Affected paths: `test/test_copyright.py`, `test/test_flake8.py`, and
  `test/test_pep257.py`
- Visible attribution: Open Source Robotics Foundation, Inc.
- Licence: Apache License 2.0, as stated in each file header
- Local status: standard ROS 2 Python package test scaffolding; the original
  notices are retained.

The same Apache-2.0 text is available at
`third_party/licenses/Apache-2.0.txt`.

## External runtime dependencies

PyTorch, ROS 2, NumPy, OpenCV, scikit-learn, Pillow, SciPy, PyYAML and the other
packages named in the dependency files are referenced as external dependencies;
their source distributions are not vendored here. Each remains governed by its
own upstream terms.

## Files with incomplete historical rights records

The legacy research prototypes under `sensemap/explore_model/` and the recovered
candidate weights under `repro_reconstructed/checkpoints/` entered this public
history without a complete per-file authorship, training-data-rights or
third-party-asset ledger. The repository history establishes when they entered
this Git repository, but not every earlier source or right. For that reason:

- no repository-wide open-source licence is asserted;
- the recovered candidates are research artifacts, not the manuscript
  checkpoint and not a generally licensed model release; and
- an author-approved rights review is required before a reusable archival
  software or model release.

## Audit method and date

This notice was prepared on 7 August 2026 from the tracked Git tree, per-file
headers, repository history, and the official upstream pages linked above.
Authors must add any unrecorded source, contributor, employer right, dataset
condition, patent restriction or third-party asset before approving a broader
licence.
