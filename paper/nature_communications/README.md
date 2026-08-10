# Nature Communications working manuscript snapshot

This directory contains the current public working snapshot of the MSO
manuscript, Supplementary Information, displayed figures, editable figure
sources and machine-readable peer-review data.

The snapshot is intentionally labelled as a working manuscript. It predates
the final results of the full-objective benchmark recorded in
`../../experiments/unified_12_method_benchmark/STATUS_2026-08-10.md`; those
results must not be added to the manuscript until the complete teacher and all
five student runs have finished and passed the frozen analyses.

## Contents

- `manuscript.tex` and `manuscript.pdf`: current main manuscript source and
  rendered PDF;
- `supplementary_information.tex` and `supplementary_information.pdf`:
  current Supplementary Information source and rendered PDF;
- `references.bib`: bibliography database;
- displayed raster/vector files at the paths used directly by the TeX sources;
- `figure_source/`: editable drawing/plot scripts, source assets and compact
  source tables;
- `peer_review_data/`: retained processed data, event-level audit records and
  reproduction scripts;
- data-access and source-inventory documents at the directory root.

Administrative submission material (cover letter, reviewer suggestions and
editor-facing forms), LaTeX build debris, temporary QA renders and obsolete
upload ZIP files are deliberately excluded from the public repository.

## Building the manuscript

The TeX source uses the Springer Nature journal template. Obtain the current
`sn-jnl.cls` and `sn-nature.bst` from the official Springer Nature template
distribution, place them in this directory, and then run:

```bash
latexmk -pdf manuscript.tex
latexmk -pdf supplementary_information.tex
```

The checked-in PDFs are the canonical rendered snapshot for this commit. The
third-party template files are not redistributed here and remain under their
upstream licence.

## Scope and provenance

The manuscript is evidence-bounded: recovered weights are candidates rather
than authenticated manuscript checkpoints; the registration replay is an
offline audit; and the available physical-team evidence does not establish a
matched, communication-impaired, three-or-more-robot physical campaign.

Source and documentation owned by the MSO authors follow the repository
licence. Third-party assets, external datasets, candidate weights and raw ROS
bags retain their separate rights and provenance boundaries.
