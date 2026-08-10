# Exploratory provisional-planning audit

This reconstructed offline audit starts from saved probability maps and isolates prediction at a planner proxy input. Both arms share one observed-only reference transform, observed-support decision, atomic measured-map commit, persistent-state hash, start cell, and planner proxy. Prediction is never used by registration or the gate and is never persisted as measured occupancy.

## Shared upstream path

The 50 perturbations form 30 clustered `base_frame_cluster` analysis units. They remain correlated snapshots from one scene and are not independent environments. The observed-only gate's mean within-cluster acceptance rate was 93.3%; its mean within-cluster rate of accepted transforms outside 0.25 m / 5 deg was 0.0%.

## Cluster-aware planner result

| Planner input | Route-available mean cluster rate | Median cluster-mean reachable frontiers | Median cluster-mean route length, m | Collision mean cluster rate | Unknown-traversal mean cluster rate | Fully known-free mean cluster rate |
|---|---:|---:|---:|---:|---:|---:|
| Measured persistent only | 100.0% | 1071.8 | 22.613 | 23.9% | 58.1% | 28.1% |
| Measured + disposable prediction | 100.0% | 986.8 | 22.613 | 26.1% | 62.5% | 26.9% |

Across 30 clusters, the disposable layer added a median 21001.5 cells to the within-cluster support mean and changed the within-cluster reachable-free mean by a median -78.1 cells. The median cluster-mean route-length difference was 0.000 m.

Provisional-minus-measured cluster-mean route availability was positive/negative/zero in 0/0/30 clusters (two-sided exact sign p=1.000). Collision-route signs were 1/0/29 (p=1.000); unknown-traversal signs were 2/0/28 (p=0.500). No event-level significance test is reported.

These results do not demonstrate a safe or statistically resolved planning benefit from prediction. Support gain is not equivalent to explored area, and the oracle is only the union of two saved measured maps rather than complete environment ground truth.

## Exploratory construction

- Only the 50 known-positive `predicted_raw` A3 perturbations are used; they aggregate to 30 `base_frame_cluster` units.
- A3 uses one common 1020 x 424 canvas and 0.05 m resolution. All predicted and observed file hashes and shapes are checked before processing.
- Raw prediction value 0 is unsupported/unknown, 1--127 is provisional free, and 128--255 is provisional occupied. The 128 threshold follows the archived 0.5 rule. The design is declared exploratory, not preregistered or prelocked.
- Accepted source probabilities are warped only by the shared observed-only estimate. Target and source probabilities use a conservative maximum on overlapping support. Measured cells then overwrite the provisional layer.
- The measured persistent map is hashed before and after both planner calls. Prediction is disposable and cannot mutate this state.

## Boundaries

The cluster units remain snapshots from one archived A3 scene. The planner is a deterministic grid proxy, not the historical MRPB planner. This is not an online, physical, temporal-consensus, communication-impairment, or exploration-coverage experiment, and it is not a complete system execution.

Input manifest SHA-256: `47037e16c500977dd12e716d497423d6d2ddfb1dc3437765a323ac52b590b183`.
