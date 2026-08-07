# Software licensing and raw-data access

## Public software

Source and documentation owned by the MSO authors are available under the BSD
3-Clause License in `LICENSE`. Editors and invited reviewers may therefore
retrieve, run, modify and make copies of that material under the licence's
terms; no separate review-only software permission is required.

The repository is mixed-source. The identified Fast Fourier Convolution
implementation and ROS 2 test templates retain their Apache-2.0 upstream terms,
as recorded in `THIRD_PARTY_NOTICES.md`. External dependencies remain governed
by their own licences.

The software licence does not establish or expand rights in the recovered
checkpoint candidates under `repro_reconstructed/checkpoints/`, external
datasets, or raw ROS 2 bags. The candidates may be inspected as research
artifacts but are not identified as the manuscript checkpoint or offered as a
generally reusable model release.

## Public raw-data access

The 18 raw ROS 2 bags and their 18 metadata files are available through a
[public read-only Google Drive folder](https://drive.google.com/drive/folders/1mbCuIISidEy87mmWPbZTtiRKfW54Fhii?usp=sharing). The folder also
contains a data README and the SHA-256 inventory reproduced in
`RAW_ROSBAG_SHA256SUMS.md`. Upload verification found 38 matching objects and
no transfer differences.

The raw records are outside the BSD-3-Clause software grant. Public view and
download access supports inspection of the reported work but does not, by
itself, establish a separate licence for redistribution, derivative datasets or
commercial reuse. Requests for uses beyond manuscript verification should be
directed to the corresponding author.

## No warranty or expanded evidence claim

The research material is supplied as is, without a claim that every historical
experiment can be reproduced. The README and component-specific provenance
documents define the known evidence boundaries. Access to a file does not
upgrade a reconstructed workflow, recovered candidate or negative offline audit
into evidence for a stronger manuscript claim.

## Recorded status on 7 August 2026

- Author-owned source and documentation: BSD-3-Clause approved.
- Identified third-party source: upstream Apache-2.0 terms retained.
- Recovered checkpoint candidates: research inspection artifacts; no general
  model-release licence asserted.
- Raw data: public read-only link active; 18 bags, 18 metadata files and two
  root documents verified after upload.
