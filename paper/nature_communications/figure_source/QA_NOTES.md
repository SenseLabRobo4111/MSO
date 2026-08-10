# Figure quality assurance

Target format: Nature Communications Article, 183 mm double-column width.

## Author-supplied schematic exports

- Main Fig. 1 and the architecture row of Main Fig. 2 are direct exports of the
  `fig1` and `fig2` pages in the project-level `figs.drawio` file.
- `export_drawio_figures.py` validates page order and dimensions, composites any
  exporter alpha over true white, and writes 600-dpi PNG and LZW TIFF files.
  Diagram elements, text, lines, colours and layout coordinates are unchanged.
- `figure2_composite.png` scales and stacks the white-background architecture
  export and the supplied registration schematic with external row headers; it
  does not crop or edit either schematic.

## Resource and prediction profile

- Source: retained model-comparison, ablation and Jetson timing values.
- Six-panel evidence map: (a) profiler operation entries versus FID; (b)
  parameter count versus LPIPS; (c) exact comparator-to-MSO parameter and
  profiler-entry ratios; (d) a 4-model by 5-metric raw-value matrix; (e) a
  5-configuration by 7-field direction-adjusted percentage-change matrix; and
  (f) recurring and encounter stages on one unified logarithmic runtime axis.
- Panel d prints the untransformed PSNR, SSIM, LPIPS, FID and KID point
  estimates. Its direction-aware within-column colour is only a reading aid.
  Panel e includes parameters, profiler entries and the same five quality
  metrics; positive values always indicate change in the favourable direction
  relative to Full.
- The ratios are deterministic divisions of the archived values in
  `source_data/model_profile.csv` (U-Net: 90.6× parameters and 162.8× profiler
  entries; LaMa-Fourier: 78.8× and 196.3×; MI-GAN: 17.4× and 32.8×). They are
  not interpreted as MAC, FLOP, energy or end-to-end speed-up ratios.
- Claim boundary: point estimates from one retained training run per model;
  latency ranges are descriptive and no training-seed uncertainty is shown.
- Recurring and encounter timing records share one logarithmic axis, with a
  separator and distinct pathway colours. Encounter-stage ranges are not
  stacked or treated as additive measurements; the archived 40–60 ms
  fusion-event total remains a separate record.
- Panel f uses compact stage labels so that the latency axis, rather than
  hardware prose, occupies the available width. Circle, square and diamond
  markers retain the CPU, GPU and mixed-execution distinction; schedule details
  are stated in the manuscript legend.
- Visual checks: readable at final width, no clipped labels, editable SVG/PDF
  text, colour-blind-safe palette, and no synthetic uncertainty marks. Raw
  model metrics remain printed in every cell; the within-column background
  colour is a direction-aware reading aid and is not a cross-metric score.

## MRPB exploration traces

- Source: all nine aggregate CSV records under
  `peer_review_data/mrpb/all_metrics/`; no hard-coded approximation is used.
- Evidence map: panel c retains nine coverage traces. Panel d retains six
  predictive occupied-cell-precision traces for MSO multi, MSO no merge, MSO
  multi with classical ORB, MSO single, MapEx and UPEN.
- The frontier-multi, frontier-single and IG-Hector quality fields are all-zero
  unavailable-value sentinels and are not presented as measured predictive
  precision. Their coverage traces remain visible.
- `MSO multi, ORB` and `MSO no merge` are numerically identical in both fields
  at every step from 0 through 1800. Their curves therefore overlap; sparse
  markers distinguish the records without applying a visual offset.
- The legacy `MSO multi, improved ORB (OBS)` curve exists only as an
  approximate hard-coded conference-draft array. No retained CSV was found, so
  that curve is not restored.
- All displayed points are read in file order through step 1800. No
  interpolation, smoothing, resampling, synthetic endpoint, uncertainty band
  or significance mark is introduced.

## Controlled transform audit

- Source: 300 event-level registration records and 150 fixed erroneous-transform
  injections in the accompanying audit package.
- Six-panel evidence map: translation and absolute-yaw error distributions for
  returned positive candidates; pre-commit outcome counts; fixed-error injection
  scores; event-level positive/low-support gate-score distributions; and
  staged-RANSAC usage counts.
- Claim boundary: one archived scene and a reconstructed offline reference
  implementation; the injection test bypasses feature matching and evaluates
  rejection before simulated mutation, not deployed rollback or recovery.
- Visual checks: empirical distributions show all returned rigid-valid positive
  candidates; raw decision counts remain visible; the yaw analysis limit is
  stated inside the panel because it lies outside the displayed range. The
  decision legend is placed inside the panel in a reserved empty band, and the
  threshold annotations are offset from their dashed reference lines.

## Team-size and scene-level behaviour

- Source: retained aggregate team-size and KTH coverage CSV files and the six
  physical endpoint records.
- Six-panel evidence map: team-size traces; exact 0.5/0.8 team crossings;
  endpoint coverage versus retained trace horizon; KTH traces; exact KTH
  0.5/0.8/0.9 crossings; and the physical endpoint trade-off.
- Claim boundary: run-level seed records are unavailable, so no confidence band,
  significance mark or matched-horizon speed-up claim is drawn. The physical
  panel has one run per method and scene and shows a trade-off, not a treatment
  effect.
- Visual checks: exact 0.5/0.8 team-size crossings, exact 0.5/0.8/0.9 KTH
  crossings, endpoints and last retained steps are derived directly from the
  CSV rows without interpolation. The separate horizon panel makes the unequal
  stopping records visible, and line styles remain distinguishable without
  colour.

## Shared quantitative palette

- Main Figs. 3, 4 and 8 use the brighter blue--violet family sampled from the
  unchanged Fig. 1 artwork: deep indigo `#5659BD`, medium blue `#5A7DBE`, light
  blue `#7EB6E0`, pale blue `#B1DDF0`, violet `#7E57C2` and pale violet
  `#B9A6D9`.
- The palette revision did not change data, axis limits, line styles, markers
  or statistical content. A later spacing refinement shortened the panel-f
  stage labels and aligned all six panels to the same compact left boundary;
  the hardware and schedule meanings remain in the manuscript legend.
- Bar annotations choose black or white from the fill luminance so that the
  lighter palette does not reduce legibility.

## Prospective replacement boundary

- The present Main Figs. 3 and 8 are interim archived-evidence figures. Their
  sparsity reflects four retained model operating points, five ablation points,
  six timing records, three aggregate team-size traces and single physical
  endpoints; no points or uncertainty marks are duplicated to increase visual
  density.
- Main Fig. 3 is replaced only after the registered multi-method, five-seed
  prediction benchmark closes with per-method accuracy, calibration, topology
  and compute records. Main Fig. 8 is replaced only after the registered
  N=2/3/5/8/10 team-scaling campaign closes with episode-level uncertainty and
  system-load records.

Strict source validation and PDF glyph audits passed for all three figures. The
standalone minimum text sizes are 6.02 pt, 6.20 pt and 6.20 pt, respectively;
after placement in the 160-mm review manuscript they remain 5.052 pt, 5.433 pt
and 5.497 pt, with no text span below 5 pt.
