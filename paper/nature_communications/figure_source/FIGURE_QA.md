# Submission figure QA and source map

Export target: Nature Communications Article at 183 mm. The present
single-column review manuscript places these figures at 160 mm; font audits
below use the actual placed size.

Quantitative backend: Python. The author-supplied schematic pages are exported
by the draw.io command-line renderer through a Python wrapper. The workflow
preserves the supplied visual identity and does not introduce synthetic
observations, smoothing, interpolation, uncertainty bands or inferred run-level
variation.

## Figure contract

| Display | Evidence role and archetype | Quantitative source | Replicate and statistical status | Processing performed |
|---|---|---|---|---|
| Main Fig. 1 | System-overview composite; author-supplied explanatory schematic | Schematic only. The embedded profiler wording is retained as artwork and is not used to establish a MAC/FLOP convention | No experimental sample size applies | Page `fig1` of packaged `source_assets/figs.drawio` (identical to the project source) is exported directly at 4409 by 2616 pixels and composited over true white. No diagram element, text, line, colour or layout coordinate is edited; 600-dpi PNG and LZW TIFF outputs are written. |
| Main Fig. 2 (`figure2_composite.png`) | Two-row method figure: predictor/distillation architecture followed by registration/fusion workflow | Explanatory schematics only | No experimental sample size applies | Page `fig2` of `figs.drawio` is exported directly at 7695 by 2354 pixels on true white. The former Supplementary Fig. 1 workflow is then scaled and stacked below it with external row headers; neither source schematic is altered. |
| Main Fig. 3 | Six-panel resource, prediction and runtime profile | `source_data/model_profile.csv` and `source_data/runtime_profile.csv`, read directly by `make_system_figures.py` | One retained point estimate per model or ablation; operational latency values/ranges are descriptive, with no training-seed variation or statistical interval | Panel a shows profiler operation entries versus FID; b, parameters versus LPIPS; c, exact comparator-to-MSO parameter and profiler-entry ratios; d, the 4-model by 5-metric raw-value matrix; e, the 5-configuration by 7-field direction-adjusted percentage-change matrix; and f, all recurring and encounter stages on one unified logarithmic runtime axis. Ratios are not interpreted as MAC, FLOP, energy or speed-up comparisons. |
| Main Fig. 4 | Six-panel controlled offline transform audit | `transform_validation/results_manuscript_reference/evaluated_events.csv`, `table3_decision_summary.csv` and `wrong_transform_injection_evaluator/evaluated_events.csv`, read by `make_system_figures.py` | 300 reconstructed event rows and 150 fixed erroneous-transform proposals from one archived scene; no deployed trial or rollback experiment | Panels a--f show positive-event translation/yaw error distributions, decision counts, injection scores, positive/low-support gate-score distributions and staged-RANSAC usage. All event points and counts remain explicit. |
| Main Fig. 8 | Six-panel aggregate system-trace and physical-endpoint grid | Retained `scaling_kth/A2_*.csv`, `scaling_kth/A3_*.csv` and `tables_and_figures/physical_endpoint_values.txt`, read by `make_system_figures.py` | One aggregate trace per logged condition and one run per physical scene/method; run-level seeds and repeated physical trials are absent | Panels a--f show team-size traces, exact team-size crossings, unequal trace horizons, KTH traces, exact KTH crossings and paired physical endpoints. No interpolation, smoothing, uncertainty band or matched-horizon speed-up is introduced. |
| Main Fig. 6 (`figure4_qualitative.png` plus `fig5_exploration.pdf`) | Two progression blocks above two descriptive time-series panels for one MRPB layout | All nine records under `peer_review_data/mrpb/all_metrics/`, listed below | Retained aggregate traces only. The historical campaign was configured for 50 seeds per method, but run-level files are unavailable; no spread, confidence interval or statistical test can be reconstructed | The two supplied progression blocks are extracted unchanged from `source_assets/figure4_base.png`. The narrow old plot strip is discarded; nine source-backed coverage traces and six source-backed predictive occupied-cell-precision traces are placed at full 183-mm width below the plate without interpolation, smoothing or resampling. |
| Main Fig. 7 (`figure5_composite.png`) | Arena-context strip above the descriptive physical progression and endpoints | `peer_review_data/tables_and_figures/physical_endpoint_values.txt` for the reported endpoints | One MSO and one frontier run in each of three arenas; no repeated-trial interval or test | The former Supplementary arena views are promoted above the original physical plate. Only the ambiguous metric heading in the supplied plate is corrected to `Occupied-cell precision`; images and numerical bars are otherwise unchanged. |
| Supplementary Fig. 2 | Single-panel descriptive KTH coverage trace | `peer_review_data/scaling_kth/A3_ours_multi_ours_orb_coverage.csv` | One retained aggregate trace containing 45 logged rows. The condition was configured for five seeds, but run-level records are unavailable; no spread or test is shown | The unsourced occupied-cell-precision panel is removed. The retained coverage rows are plotted without smoothing or subsampling. |
| Supplementary Fig. 5 | Archived single-robot physical diagnostic | Eight CSVs and the extraction script under `peer_review_data/physical/single_robot/` | Two included runs for MSO, MapEx and UPEN; one included IG-Hector run after the failed first run was excluded | Original supplied figure retained without pixel changes. The caption defines the legacy metric as unmasked map-raster free-cell support, not physical arena coverage, and discloses unequal sample counts and stopping rules. |

## Exact MRPB mapping

The main-text panels are generated by
`peer_review_data/mrpb/plot_exploration_results.py`, which reads the retained
records directly and exports the full-width vector used in the manuscript.
Panel c reads `coverage` from all nine retained CSV records:

- `ours_multi_ours_orb.csv` (`MSO multi`)
- `ours_multi_ours_orb_nomerge.csv` (`MSO no merge`)
- `ours_multi_orb.csv` (`MSO multi, ORB`)
- `nearest-multi-our-orb.csv` (`Frontier multi`)
- `ours_single.csv` (`MSO single`)
- `MapEx_single.csv` (`MapEx`)
- `upen_single.csv` (`UPEN`)
- `ig-hector_single.csv` (`IG-Hector`)
- `nearest.csv` (`Frontier single`)

Panel d reads `predicted_map_quality`, displayed as `Occupied-cell precision`,
from the six records with retained predictive completion output:

- `ours_multi_ours_orb.csv` (`MSO multi`)
- `ours_multi_ours_orb_nomerge.csv` (`MSO no merge`)
- `ours_multi_orb.csv` (`MSO multi, ORB`)
- `ours_single.csv` (`MSO single`)
- `MapEx_single.csv` (`MapEx`)
- `upen_single.csv` (`UPEN`)

The quality fields in `nearest-multi-our-orb.csv`, `ig-hector_single.csv` and
`nearest.csv` are not used as predictive precision: these methods do not
provide a retained predictive completion output, and the stored fields are
all-zero unavailable-value sentinels rather than measured precision traces.
Coverage remains available and is shown for all three. Every displayed line
uses all rows with `step <= 1800` in file order; there is no interpolation,
smoothing, resampling or synthetic endpoint.

The `ours_multi_orb.csv` and `ours_multi_ours_orb_nomerge.csv` records are
numerically identical in both displayed fields at every retained step from 0
through 1800, so the corresponding `MSO multi, ORB` and `MSO no merge` curves
coincide. Sparse markers make both identities inspectable without adding an
offset or altering any value.

## Exclusions and corrections

- The former Supplementary Fig. 3b occupied-cell-precision series has no
  retained machine-readable source.  It is excluded from the figure, caption
  and analysis; the source-backed KTH coverage panel remains.
- The approximate hard-coded arrays previously used for the main MRPB time
  series were removed from the packaged plotting source. The corrected display
  and its independently runnable archive plot are built directly from the nine
  packaged aggregate CSVs listed above. The legacy `MSO multi, improved ORB
  (OBS)` series has no retained CSV and is therefore not restored from its
  approximate arrays.
- The all-zero predictive-quality fields in the two frontier records and the
  IG-Hector record are not interpreted as measured zero precision. Their
  exclusion from predictive precision is explicit; all nine source-backed
  coverage records remain displayed, and no row from a displayed series is
  selectively removed.
- `Accuracy` and vague `prediction accuracy` terminology is not used for the
  source-backed quantitative displays.  The precise map metric is
  `Occupied-cell precision`, and Fig. 3 uses the neutral field name `Profiler
  operation entries`. The unchanged author-supplied Fig. 1 page retains its
  original profiler wording; because that convention was not recovered, it is
  not used as MAC or FLOP evidence in the quantitative figure or text.

## Image-integrity record

| Asset | Raw/frozen source | Crop or local edit | Tone/colour adjustment | Stitching or reuse |
|---|---|---|---|---|
| `figure1.png` | Packaged `source_assets/figs.drawio`, page `fig1` | No crop or local edit; exporter alpha is composited over true white | None | Direct page export only; no panel reuse |
| `figure2_composite.png` | Packaged `source_assets/figs.drawio`, page `fig2`; `source_assets/supplementary_figure1.png` | No content crop; direct white-canvas export, deterministic scaling and two external row headers | None | Two supplied schematics stacked in one main-text display |
| `figure4_qualitative.png` | `source_assets/figure4_base.png` | Cropped at pixel x = 13,330, the supplied boundary before the obsolete plot strip; external panel headers added | None | Both progression blocks retained; full-width vector metrics placed separately by LaTeX |
| `figure5.png` | `source_assets/figure5_base.png` | One bounded heading rectangle only | None | Existing supplied composite retained |
| `figure5_composite.png` | `figure5.png`; `supplementary_scene1.png`--`supplementary_scene3.png` | No experimental crop; deterministic scaling and external scene-label bands | None | Three supplied arena views stacked above the supplied physical plate |
| `supplementary_figure3.png` | KTH aggregate CSV | No image crop; regenerated single panel | Standard Python line rendering only | No panel reuse |

No brightness, contrast, gamma or pseudocolour adjustment is applied.  The
physical and progression images have no calibrated scale bar and are used only
as descriptive context, not for metric extraction.

## Export and display checks

- `export_drawio_figures.py` validates the `fig1`/`fig2` page order and expected
  dimensions, exports both pages without element edits, checks true-white
  corner pixels and writes 600-dpi PNG and LZW TIFF companions. The assembled
  method figure and other bounded composites have PNG and LZW TIFF companions
  generated by `assemble_main_composites.py`.
- Main Figs. 3, 4 and 8 each contain six panels and are exported as editable
  SVG/PDF, 300-dpi PNG and 600-dpi LZW TIFF files.
- The exact MRPB metric panels, comprising nine coverage and six predictive
  occupied-cell-precision traces, are exported as SVG, PDF, 600 dpi PNG and
  600 dpi LZW TIFF at a declared 183-mm width.
- `peer_review_data/mrpb/plot_exploration_results.py` exports full-width SVG,
  PDF, 600 dpi PNG and 600 dpi LZW TIFF companions under
  `peer_review_data/mrpb/figs/`; its physical endpoint plot is read directly
  from `peer_review_data/tables_and_figures/physical_endpoint_values.txt`.
- Supplementary Fig. 2 is exported as SVG, PDF, 600 dpi PNG and 600 dpi LZW
  TIFF by `peer_review_data/scaling_kth/plot_kth_curves.py`.
- Figure labels, legends and axes were checked panel by panel for clipping and
  collision in the source exports and again in the compiled manuscript.
- Standalone PDF glyph audits pass at 6.02 pt for the resource profile and
  6.20 pt for both the transform audit and scaling figure.  Their actual minima
  after placement in the 160-mm review manuscript are 5.052 pt, 5.433 pt and
  5.497 pt, respectively; no placed text span is below 5 pt.  The full-width
  MRPB metric panels remain at 6.00 pt, and the former sub-5-pt effective MRPB
  strip is no longer used in the manuscript.
- The main manuscript and SI compile without overfull boxes, undefined labels,
  missing citations or float-size warnings.  All affected PDF pages were
  rendered at final placement and inspected for clipping and overlap.

## Remaining evidence limits

The corrected figures do not recover missing run-level seeds, the historical
training split manifest, the manuscript checkpoint-selection record, a KTH
occupied-cell-precision series, deployed transform decisions, or repeated
physical trials.  Consequently, all affected curves and image sequences remain
descriptive; no confidence bands, significance marks, causal effects, deployed
registration-accuracy claim or large-team physical-generalisation claim is
made.
