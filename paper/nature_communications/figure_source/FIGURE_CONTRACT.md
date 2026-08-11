# Figure contract

Target: Nature Communications Article, double-column figures at 183 mm width.

## Resource and prediction profile

- Core conclusion: the evaluated MSO predictor occupies a low-cost operating point and the integrated stack fits the reported onboard timing envelope.
- Archetype: asymmetric six-panel quantitative grid. Panel a places recorded
  profiler operation entries against FID, panel b places parameter count against
  LPIPS, panel c reports exact comparator-to-MSO parameter and profiler-entry
  ratios, panel d is a 4-model by 5-metric raw-value matrix, panel e is a
  5-configuration by 7-field direction-adjusted percentage-change matrix, and
  panel f places all recurring and encounter stages on one unified logarithmic
  runtime axis.
- Evidence: exact retained values in `source_data/model_profile.csv` and
  `source_data/runtime_profile.csv`. Panel d prints the untransformed PSNR,
  SSIM, LPIPS, FID and KID point estimates; its colour is only a within-column,
  direction-aware reading aid. Panel e uses parameter count, profiler entries
  and the same five quality metrics; positive signed values consistently denote
  a favourable direction relative to Full. Ratios are deterministic divisions
  of the displayed profiler entries, not MAC, FLOP, energy or speed-up
  measurements.
- Boundary: one retained training run per model; no training-seed uncertainty or unknown-region-only metrics.

## MRPB exploration traces

- Core conclusion: the retained aggregate records support a descriptive
  comparison of exploration coverage and predictive occupied-cell precision on
  one MRPB layout without reducing the original source-backed method set.
- Archetype: two source-backed time-series panels below the supplied qualitative
  progression plate: nine coverage traces and six predictive occupied-cell-
  precision traces.
- Evidence: all nine CSV records under `peer_review_data/mrpb/all_metrics/` are
  read through step 1800 in file order. Every method with a retained predictive
  completion output contributes to occupied-cell precision. No interpolation,
  smoothing, resampling or synthetic endpoint is used.
- Boundary: only aggregate traces were retained, so no seed-level interval or
  statistical test is reconstructable. The `MSO multi, ORB` and `MSO no merge`
  records are numerically identical for both plotted fields through step 1800
  and therefore coincide; sparse markers identify the records without moving
  either curve. The legacy `MSO multi, improved ORB (OBS)` trace has no retained
  CSV and is not restored from its approximate hard-coded arrays.

## Offline registration audit

- Core conclusion: the reconstructed one-scene reference implementation returns low median pose errors on selected positive encounters, while explicit decision and injection records expose its abstention boundary.
- Archetype: six-panel quantitative grid comprising translation and yaw error
  distributions, pre-commit outcomes, fixed-error injection scores,
  positive/low-support gate-score distributions and staged-RANSAC usage counts.
- Evidence: 300 event-level benchmark rows, the associated decision summary
  and 150 fixed wrong-transform injections.
- Boundary: offline reconstructed implementation, one scene, no deployed rollback or recovery claim.

## Scaling and scene-level system behaviour

- Core conclusion: retained aggregate traces show earlier recorded coverage milestones for the five-robot configuration and high final coverage on one retained KTH floorplan, while the unequal logged horizons bound any scaling interpretation.
- Archetype: six-panel quantitative grid with team-size traces, exact 0.5/0.8
  crossings, a trace-horizon diagnostic, a KTH trace, exact KTH 0.5/0.8/0.9
  crossings and paired physical endpoints.
- Evidence: retained aggregate CSV traces and six descriptive physical endpoint records.
- Boundary: run-level seed files were not retained, endpoints differ, and no
  uncertainty bands or matched-horizon speed-up claim is shown. The physical
  panel contains one run per method and scene and is descriptive.

All generated quantitative plots use Python with editable SVG and PDF text. The
author-supplied Fig. 1 and Fig. 2 pages are exported directly from the matching
pages of the packaged `source_assets/figs.drawio` file (identical to the
project-level source) onto a true-white canvas; no
diagram element, text, colour or layout coordinate is edited. Other retained
image plates and the method-figure stacking operation are documented in
`FIGURE_QA.md`.
