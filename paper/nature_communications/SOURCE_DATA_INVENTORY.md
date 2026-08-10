# Nature Communications source-data inventory

The submission-time source-data package contains a final-display manifest at
`peer_review_data/FINAL_DISPLAY_MAP.tsv`, with one row for every quantitative or
source-bearing main or supplementary display.

## Present in the source-data packages

- Main Fig. 1: page `fig1` of `figure_source/source_assets/figs.drawio` and the
  direct white-canvas export route in `figure_source/export_drawio_figures.py`.
  The display is explanatory; its retained profiler wording is not used to
  establish a MAC/FLOP convention.
- Main Fig. 2: page `fig2` of the same packaged `figs.drawio`, the supplied registration schematic
  and the deterministic stacking route in
  `figure_source/assemble_main_composites.py`. Both source schematics are
  retained without content edits.
- Main Fig. 3: `figure_source/source_data/model_profile.csv` and
  `runtime_profile.csv`. Panels a--f show profiler operation entries versus
  FID, parameters versus LPIPS, deterministic comparator-to-MSO resource
  ratios, a 4-model by 5-metric raw-value matrix, a 5-configuration by 7-field
  direction-adjusted percentage-change matrix and all recurring/encounter
  timing records on one unified logarithmic axis. Detailed raw values are
  reproduced in Supplementary Table 5 and Supplementary Table 2.
- Main Table 1: pairwise SR, RMSE, SSIM and Dice overlap summary. Two legacy
  distance columns were removed because their coordinate and aggregation
  conventions could not be verified.
- Main Fig. 4: portable source maps, fixed event manifest, 300 event-level
  transform records, the manuscript-specific decision summary and 150 fixed
  erroneous-transform proposals. Panels a--f show translation/yaw errors,
  pre-commit decisions, injection scores, gate-score strata and staged-RANSAC
  usage; detailed values are reproduced in Supplementary Table 6.
- Main Table 2: fusion-component ablation point estimates.
- Main Fig. 6: all nine retained aggregate MRPB CSV records and a rendering
  script that draws nine coverage traces and six predictive occupied-cell-
  precision traces directly, without interpolation, smoothing, resampling or
  synthetic endpoints. `MSO multi, ORB` and `MSO no merge` are identical
  through step 1800 and therefore overlap. The legacy OBS trace has no retained
  CSV and is not restored from approximate arrays. Run-level seed files are
  absent.
- Main Fig. 7: physical endpoint values and retained processing scripts.
- Main Fig. 8: retained aggregate auxiliary team-size and KTH coverage traces
  plus the six physical endpoint records. Panels a--f show team-size traces,
  exact team-size crossings, unequal retained horizons, KTH traces, exact KTH
  crossings and paired physical endpoints. The auxiliary $N=2,3,5$ records use
  their native logged-step axes and are not treated as a controlled scaling
  comparison with the main MRPB trace.
- Supplementary Fig. 5: eight retained single-robot physical CSV records, a
  four-method descriptive summary and the historical extraction/rendering
  script. The failed first IG-Hector run is retained but excluded from the
  plotted aggregate and explicitly disclosed.
- Supplementary Fig. 1: available team-size coverage traces.
- Supplementary Fig. 2: the retained KTH coverage trace. The unretained
  occupied-cell-precision series is not displayed or analysed.
- Supplementary Tables 2--4: numerical values in the manuscript and bandwidth
  extraction script.
- `figure_source/`: the direct white-canvas diagram exporter, Python quantitative
  rendering script, explicit figure contract, figure-specific CSV inputs and
  editable SVG/PDF outputs for the three six-panel quantitative figures.
- The companion public repository revision contains the hash-verified
  8,304-row preserved-archive inventory, two architecture-compatible recovered
  checkpoint candidates, the declared reconstructed training/evaluation path,
  and canonical outputs from the archived two-robot registration replay.
- The 18 raw bags, their 18 metadata files, a data README and the checksum
  inventory are in a [public read-only Google Drive folder](https://drive.google.com/drive/folders/1mbCuIISidEy87mmWPbZTtiRKfW54Fhii?usp=sharing).
  Upload and unauthenticated-access checks passed.

## Required before submission

- Recover and include the historical fitting, validation and held-out split
  manifest, or retain the manuscript's explicit limitation that it is absent.
- Retain the explicit distinction between the missing historical training
  entry/split/checkpoint-selection record and the newly provided reconstructed
  workflow and recovered checkpoint candidates. Neither candidate is identified
  as the checkpoint underlying Supplementary Table 5.
- Include any available run-level MRPB files for all reported seeds; otherwise
  keep the manuscript's explicit aggregate-data limitation.
- Retain the recovered deployment-source revision and frozen replay manifest in
  the repository. The replay is a negative offline audit and cannot be presented
  as a successful online-registration or recovery experiment.
- Do not restore the conference draft's 200-clip temporal-prior ablation as a
  numerical result unless its clip manifest or aggregate source record is
  recovered; the present submission explicitly discloses its omission.
- Export the material above as the journal's final Source Data workbook or zipped
  machine-readable folder. Displays whose source records are incomplete must
  retain the limitations stated in the manuscript and figure legends.
