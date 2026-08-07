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
does not imply that other MSO files are derived from the upstream project. The
repository's BSD-3-Clause licence does not replace these Apache-2.0 terms.

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

## Author-owned material, model artifacts and data

Source and documentation owned by the MSO authors are released under the BSD
3-Clause License in `LICENSE`. A licence from the MSO authors cannot grant
rights held by an unidentified third party, so any unrecorded upstream material
must be added to this notice when discovered.

The recovered candidate weights under `repro_reconstructed/checkpoints/`
entered the public history without a complete training-data-rights or model-
selection ledger. They are research inspection artifacts, are not identified
as the manuscript checkpoint, and are not offered as a generally reusable model
release under the source licence. External datasets and raw ROS 2 bags are also
outside the BSD-3-Clause software grant. Their access and reuse require their
own rights and data-sharing review.

## Audit method and date

This notice was prepared on 7 August 2026 from the tracked Git tree, per-file
headers, repository history, and the official upstream pages linked above.
Authors must add any unrecorded source, contributor, employer right, dataset
condition, patent restriction or third-party asset if it is later discovered.
