# Peer-review data archive

This archive contains the retained processed data and evaluation records that can
be supplied with the manuscript at first submission. It is deliberately
separated from third-party datasets and large ROS bag files.
The 18 raw bags and 18 metadata companions are available separately in a
[public read-only Google Drive folder](https://drive.google.com/drive/folders/1mbCuIISidEy87mmWPbZTtiRKfW54Fhii?usp=sharing).

## Contents

- `FINAL_DISPLAY_MAP.tsv`: one row per quantitative or source-bearing final
  display, with its retained source, script, reported unit and principal limit.
- `RAW_ROSBAG_SHA256SUMS.md`: exact relative paths, byte counts and SHA-256
  values for the 18 raw bags and 18 metadata files distributed separately.
- `tables_and_figures/`: retained experiment summary and a machine-readable text
  transcription of the quantitative tables and physical endpoint values.
- `transform_validation/`: portable event assets and manifest, event-level
  registrar output, evaluator summaries, the reconstructed reference registrar, and the fixed
  wrong-transform injection audit used for the main transform figure and
  Supplementary Table 6. Its `README.md` gives the
  exact thresholds, cluster field, bootstrap settings, random seed, commands,
  and expected counts.
- `mrpb/`: retained aggregate MRPB CSV records and a plotting script that reads
  those records directly. The plotted points are the exact retained aggregate
  values and are shown without interpolation or smoothing; run-level records
  are unavailable.
- `scaling_kth/`: retained coverage traces and plotting scripts used in the main
  team-size/KTH figure and Supplementary Figs. 1 and 2.
- `physical/`: bandwidth extraction and physical-log processing scripts. Its
  `single_robot/` subfolder contains all eight retained per-run CSVs, the
  historical extraction script and a summary of the unequal included-run
  counts used in Supplementary Fig. 5. The large source rosbags are linked
  separately rather than bundled here.

## Evidence boundaries

The MRPB files are exact retained aggregate traces rather than a complete set of
run-level files for every configured seed; no smoothing is applied when they are
plotted. The team-size and KTH folders likewise retain aggregate coverage only;
the KTH occupied-cell-precision series is not available in machine-readable
form. The direct-transform test is an offline reference implementation
reconstructed from the manuscript specification and is not a replay of the
deployed executable. The wrong-transform test bypasses feature matching and
evaluates a simulated pre-commit state transition. Physical endpoint
comparisons contain one run per method and scene. The auxiliary single-robot
metric is unmasked map-raster free-cell support rather than arena coverage;
the failed first IG-Hector run is retained but excluded from the plotted
aggregate and the resulting unequal sample counts are disclosed.

The public raw-data folder contains the identifiable physical bags and the same
SHA-256 inventory included here. It does not supply the unrecovered historical
training entry, manuscript checkpoint, split manifest, seed-level simulation
records or unlogged deployment decisions; those absences remain explicit in the
manuscript.
