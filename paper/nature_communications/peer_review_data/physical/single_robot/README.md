# Archived single-robot physical diagnostic

This folder contains the eight retained per-run CSV records and the historical
rendering script for the four-method single-robot comparison shown as
Supplementary Fig. 5. The records were extracted from the corresponding ROS 2
bags at 0.05 m grid resolution.

The figure's legacy `Explored free area` label is the number of free cells in
the complete unmasked `/map` raster multiplied by the squared grid resolution.
It is not a region-of-interest-clipped arena area or a coverage fraction. The
value can therefore exceed the hand-measured arena footprint and must not be
interpreted as physical area explored.

MSO, MapEx and UPEN each contribute two successful runs. The 12.747 s
`ighector_trial1` record is retained here for completeness but was excluded
from the plotted aggregate because the run failed; IG-Hector therefore has one
included run. Stopping behaviour also differs across planners. The display is
a provenance diagnostic, not a replicated or matched-horizon efficacy test.
`summary.tsv` states the included-run counts and the plotted descriptive means.

The Python script reads the original bag directories as well as these derived
CSVs; the raw bags and their metadata are distributed through the data link
reported in the manuscript and repository.
