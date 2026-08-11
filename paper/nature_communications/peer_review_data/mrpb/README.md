# MRPB retained records

`all_metrics/` contains the retained aggregate CSV logs from the MRPB campaign.
`plot_exploration_results.py` reads the plotted aggregate series directly from
these CSV files. It plots the recorded points without interpolation or
smoothing; the displayed values therefore match the retained aggregate records
rather than a visual reconstruction. At recorded step 1500, the CSV values are
0.7048 for full MSO and 0.6042 for MSO without merge.

The coverage panel includes all nine retained configuration records. The
occupied-cell-precision panel includes the six configurations with predictive
completion outputs. Observed-only precision fields are not presented as
prediction quality, and the all-zero frontier-multirobot field remains an
unavailable-value sentinel. The retained classical-ORB multirobot and no-merge
records are numerically identical over the displayed interval; both are drawn
with distinct line/marker encodings without displacing either record.

Run-level files for the configured seeds and time-resolved dispersion summaries
were not retained. The script therefore cannot reconstruct per-run variation or
seed-level uncertainty, and the manuscript treats the aggregate curves as
descriptive.
