# Exploratory frontier-ranking audit

This reconstructed offline audit starts from saved probability maps and gives prediction only one role: ranking the same measured-map reachable frontier candidates. Registration, gating, measured-map commit, candidate generation, obstacle inflation, traversability, and path search are identical between arms and do not read prediction.

The 50 perturbations aggregate to 30 clustered `base_frame_cluster` analysis units. They reduce repeated-frame pseudoreplication but remain correlated snapshots from one scene; they are not independent environments. The shared observed-only gate's mean within-cluster acceptance rate was 93.3%.

## Cluster-aware selected-target and path result

| Ranking | Median cluster-mean hidden-free yield | Median cluster-mean hidden occupied | Median cluster-mean route length, m | Collision mean cluster rate | Unknown-traversal mean cluster rate | Fully known-free mean cluster rate |
|---|---:|---:|---:|---:|---:|---:|
| Measured farthest | 1.0 | 2.0 | 22.613 | 23.9% | 58.1% | 28.1% |
| Predicted-free ranking | 7.8 | 2.0 | 16.113 | 32.2% | 47.5% | 38.3% |

The predicted-minus-baseline hidden-free cluster mean was positive/negative/zero in 17/9/4 clusters. The two-sided exact cluster sign test gives p=0.169; the median cluster-mean difference was 1.0 cells, but the mean was -14.9 cells. Thus the event-level positive impression does not survive cluster-aware inference, and the cluster mean is negative.

Collision-route cluster-mean signs were 9/5/16 (two-sided exact sign p=0.424); unknown-traversal signs were 6/9/15 (p=0.607). No event-level significance test is reported.

The prediction-ranked goal changed the cluster-mean route length by a median -6.175 m. The hidden-occupied cluster-mean difference had median 0.8 cells and mean 7.7 cells. This is a utility--risk trade-off, not a safety benefit.

## Exploratory design

- The candidate set is exactly the current measured-only planner's reachable frontier mask. Its hash is identical between paired arms.
- The baseline chooses maximum measured path distance. Prediction chooses maximum predicted-free support on measured-unknown cells in a 1.0 m disk, then maximum measured path distance and row/column order for ties.
- The 1.0 m radius follows the existing frontier-clustering scale (`DBSCAN eps=1.0`) and equals 20 cells at 0.05 m resolution. The design is exploratory, not preregistered or prelocked.
- Both routes are reconstructed only on the same measured persistent map. Prediction is never persisted and never changes collision checking.
- Oracle hidden-free yield counts measured-unknown cells labelled free by the two-map ground-truth-transform union within the same 1.0 m disk. It is a partial offline oracle, not explored area or complete environment truth.

## Boundaries

The 30 cluster units remain correlated snapshots from one archived A3 scene and are not independent environments. The planner is a deterministic audit proxy, not historical MRPB navigation. This reconstructed audit cannot establish online exploration, physical scaling, communication robustness, population-level safety, or complete-system performance.

Input manifest SHA-256: `47037e16c500977dd12e716d497423d6d2ddfb1dc3437765a323ac52b590b183`.
