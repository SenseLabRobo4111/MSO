# Transform validation report

Correctness thresholds: **tau_t = 0.25 m and tau_yaw = 5 deg**. Errors use `E = inv(T_gt) @ T_hat` under the `T_i<-j` convention.

Uncertainty: **95% percentile cluster bootstrap**, cluster `cluster_id`, 10000 replicates, seed 20260807. Clusters, not individual encounters, are sampled with replacement.

## A. SE(2) accuracy on returned positive candidates

| Input | Method | GT+ N | Returned/rigid-valid | Translation m: median [IQR]; P90; median 95% CI | Yaw deg: median [IQR]; P90; median 95% CI |
| --- | --- | --- | --- | --- | --- |
| observed | OFFLINE_RIGID_OCCUPIED_SUPPORT_GATE_0.20 | 50 | 50/50 | 1.000 [1.000, 1.000]; 1.000; [1.000, 1.000] | 15.000 [15.000, 15.000]; 15.000; [15.000, 15.000] |
| predicted | OFFLINE_RIGID_OCCUPIED_SUPPORT_GATE_0.20 | 50 | 50/50 | 1.000 [1.000, 1.000]; 1.000; [1.000, 1.000] | 15.000 [15.000, 15.000]; 15.000; [15.000, 15.000] |
| predicted_raw | OFFLINE_RIGID_OCCUPIED_SUPPORT_GATE_0.20 | 50 | 50/50 | 1.000 [1.000, 1.000]; 1.000; [1.000, 1.000] | 15.000 [15.000, 15.000]; 15.000; [15.000, 15.000] |

## B. Selective reliability

| Input | Method | Pos/neg | Accepted | Rejected | TA | FA (% accepted) | FR (% positive) | TR | Gate violations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observed | OFFLINE_RIGID_OCCUPIED_SUPPORT_GATE_0.20 | 50/0 | 0 | 50 | 0 | 0 (NA%; CI NA) | 50 (100.0%; CI [100.000, 100.000]) | 0 | 0 |
| predicted | OFFLINE_RIGID_OCCUPIED_SUPPORT_GATE_0.20 | 50/0 | 0 | 50 | 0 | 0 (NA%; CI NA) | 50 (100.0%; CI [100.000, 100.000]) | 0 | 0 |
| predicted_raw | OFFLINE_RIGID_OCCUPIED_SUPPORT_GATE_0.20 | 50/0 | 0 | 50 | 0 | 0 (NA%; CI NA) | 50 (100.0%; CI [100.000, 100.000]) | 0 | 0 |

## C. Wrong-transform injection safety

| Input | Method | Injected N | Accept/reject | Safe reject | Incorrect commit | Recovered/observed commits | Detection latency events med/P90 | Recovery events med/P90 | Contaminated cycles med/P90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observed | OFFLINE_RIGID_OCCUPIED_SUPPORT_GATE_0.20 | 50 | 0/50 | 50 (100.0%; CI [100.000, 100.000]) | 0 (0.0%; CI [0.000, 0.000]) | 0/0 | 0.000/0.000 | NA/NA | NA/NA |
| predicted | OFFLINE_RIGID_OCCUPIED_SUPPORT_GATE_0.20 | 50 | 0/50 | 50 (100.0%; CI [100.000, 100.000]) | 0 (0.0%; CI [0.000, 0.000]) | 0/0 | 0.000/0.000 | NA/NA | NA/NA |
| predicted_raw | OFFLINE_RIGID_OCCUPIED_SUPPORT_GATE_0.20 | 50 | 0/50 | 50 (100.0%; CI [100.000, 100.000]) | 0 (0.0%; CI [0.000, 0.000]) | 0/0 | 0.000/0.000 | NA/NA | NA/NA |

FA is reported per accepted encounter; FR is reported per GT-positive encounter. A pre-commit injection rejection has both `accepted=false` and `committed=false`. CIs are cluster-bootstrap percentile intervals.
