# Exploratory audits reconstructed from saved probability maps

This directory contains three bounded offline audits. They start from archived
probability maps and use a newly reconstructed reference registrar plus simple
grid-planning proxies. They do not rerun training, recover the historical online
implementation, or execute a complete robot system. Results must therefore be
described as exploratory saved-map audits rather than system-level validation.

The registration-input audit reconstructs this component chain:

```text
saved probability map or observed-only control
    -> reconstructed reference registration
    -> observed occupied-support gate
    -> isolated atomic measured-map commit
    -> categorical occupancy state
    -> four-connected farthest-frontier grid proxy
```

The provisional-planning audit fixes one observed-only transform, gate decision,
measured-map commit, start cell, and planner proxy. Prediction is visible only as
a disposable planner layer and is never persisted as measured occupancy.

The frontier-ranking audit is narrower still. Both arms use the same measured
reachable-frontier candidates, traversability mask, obstacle inflation, and path
search. Prediction may rank those candidates but cannot change path feasibility
or collision checking.

## Public peer-review data

The portable transform-validation subtree is checked into
`paper/nature_communications/peer_review_data/transform_validation/`. Its
manifest is `manifest.json`; all raster paths are relative to that manifest's
directory and are resolved there by each runner. The complete paper snapshot
has a byte-level `SHA256SUMS` inventory.

All runners default to this public repository path. An explicit manifest path
is also supported for a copied subtree or a different workspace layout.

## Statistical unit and challenge semantics

Repeated synthetic perturbations share base frames. Raw event rows are retained
only for traceability; every numerical summary first averages within
`base_frame_cluster`, and every exact sign test operates on those cluster means.
No event-level significance test is reported.

The 50 known-positive `predicted_raw` perturbations form 30 base-frame clusters.
The other 50 low-support temporal-mismatch cases form 9 clusters and are treated
as **indeterminate abstention challenges**, not verified negatives. An accepted
challenge is reported as a challenge accept, not an error or false accept. This
manifest contains no explicit known-error injection, so a false-accept rate is
not estimable from it.

The manifest's selection-lock field is a source self-declaration, not proof of
preregistration. The runners do not use that declaration for inference; all
three analyses are explicitly exploratory.

## Run

From the repository root:

```powershell
py -3 -m pip install -r integrated_offline\requirements-lock.txt

py -3 integrated_offline\run_paired_ablation.py `
  --output integrated_offline\results

py -3 integrated_offline\run_provisional_planning_ablation.py `
  --output integrated_offline\planning_results

py -3 integrated_offline\run_frontier_ranking_ablation.py `
  --output integrated_offline\frontier_ranking_results
```

For a cleanly extracted bundle, pass its relative-path manifest explicitly:

```powershell
py -3 integrated_offline\run_paired_ablation.py `
  C:\path\to\peer_review_data\transform_validation\manifest.json `
  --output integrated_offline\results
```

The first command writes `paired_events.csv` with raw audit records and
`paired_differences.csv` with one row per base-frame cluster. The planning and
ranking commands analogously write `planning_events.csv` plus
`planning_differences.csv`, and `ranking_events.csv` plus
`ranking_differences.csv`. Each result directory also contains `summary.json`,
`REPORT.md`, `run_manifest.json`, and `SHA256SUMS`.

`reference_registrar.py` is documented by `REFERENCE_REGISTRAR.md`. Those are
the correct local filenames; the registrar is a new reconstruction rather than
the historical matcher.

Run the contract, matcher-fallback, relative-path, and clean-extract tests with:

```powershell
py -3 -m unittest discover -s integrated_offline\tests -v
```

The portability test copies the public transform-validation subtree to a
temporary directory, resolves relative raster paths, and executes one paired
audit record from that copy.

## Map and planner contracts

The persistent map begins as the target measured occupancy grid. An accepted
transform may fill only unknown target cells from the warped measured source;
it cannot overwrite measured target cells. A rejected proposal returns a copy
with an identical state hash and zero changed cells.

For the provisional layer, raw value 0 is unsupported, values 1--127 are
provisional free, and values 128--255 are provisional occupied. The threshold
follows the archived 0.5 rule. Supported target and aligned-source probabilities
use a conservative maximum, after which measured cells remain authoritative.
The provisional layer is discarded after planning.

For each isolated perturbation, the grid proxy chooses a start in the largest
traversable component, inflates occupied cells by two 0.05 m cells, and computes
a four-connected shortest route to a reachable frontier. The partial oracle is
the union of two measured maps aligned by the known synthetic transform; it is
not complete environment ground truth.

All cluster units come from one archived A3 scene. These audits cannot establish
physical scaling, communication robustness, online exploration, population-level
safety, or complete-system performance.
