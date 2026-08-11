# Exploratory reconstructed component-chain audit

This is a single-scene exploratory audit reconstructed from saved probability maps. It does not rerun the historical trainer or historical online implementation, and it is not a complete system execution, an online robot experiment, or evidence of real-world exploration improvement.

## Reconstructed audit chain

`saved probability map / observed-only control -> reconstructed reference registration -> observed-support gate -> isolated atomic measured-map commit -> categorical occupancy state -> deterministic farthest-frontier grid proxy`

The raw trace contains 100 paired perturbations grouped by `base_frame_cluster`: 30 known-positive clusters and 9 indeterminate abstention-challenge clusters. All summaries below first average repeated perturbations within a cluster; exact sign tests then use cluster means. No event-level significance test is reported.

## Cluster-aware result

| Arm | Correct-positive accept, mean cluster rate | Incorrect-positive accept, mean cluster rate | Positive abstention, mean cluster rate | Challenge accept, mean cluster rate | Collision route, mean positive-cluster rate | Unknown traversal, mean positive-cluster rate |
|---|---:|---:|---:|---:|---:|---:|
| Predicted maps | 95.0% | 1.7% | 3.3% | 0.0% | 19.7% | 57.5% |
| Observed only | 93.3% | 0.0% | 6.7% | 0.0% | 23.9% | 58.1% |

Across the 30 known-positive clusters, the predicted-minus-observed correct-accept cluster mean was positive/negative/zero in 1/1/28 clusters (two-sided exact sign p=1.000). The corresponding incorrect-positive accept signs were 1/0/29 (p=1.000).

The median predicted-minus-observed difference between cluster-mean known-coverage gains was 129.8 cells and the mean was 1753.7 cells. Collision-route cluster-mean signs were 1/2/27 (p=1.000).

The low-support temporal-mismatch cases are indeterminate abstention challenges, not verified negatives. Challenge-accept cluster-mean signs were 0/0/9 (p=1.000); these accepts are not counted as errors or false accepts. This manifest contains 0 explicit known-error-injection clusters, so a false-accept rate is not estimable here.

The cluster-aware result does not demonstrate a safety advantage from prediction. The clusters are snapshots from one archived scene, not independent environments.

## Pose errors on known-positive clusters with rigid candidates

| Arm | Cluster-mean translation median / P90, m | Cluster-mean absolute yaw median / P90, deg |
|---|---:|---:|
| Predicted maps | 0.049 / 0.097 | 0.071 / 0.143 |
| Observed only | 0.039 / 0.080 | 0.039 / 0.090 |

## Boundaries

- The predictor stage consists only of saved probability maps; it is not a recovered training-to-inference execution.
- Each event has an isolated persistent state. The synthetic perturbations are not a coherent temporal trajectory, so three-frame consensus is not invented.
- The planner is a deterministic four-connected farthest-frontier audit proxy with 0.10 m obstacle inflation, not the historical MRPB planner or navigation stack.
- MRPB predicted and observed rasters are excluded because their dynamic canvases differ and the retained files lack the metric origins needed for a valid pairing.
- Physical bags are excluded from this causal analysis because they contain no transform, gate, commit, or planner-event topics.

Input manifest SHA-256: `47037e16c500977dd12e716d497423d6d2ddfb1dc3437765a323ac52b590b183`.
