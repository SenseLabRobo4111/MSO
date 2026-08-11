# Transform validation report

Correctness thresholds: **tau_t = 0.25 m and tau_yaw = 5 deg**. Errors use `E = inv(T_gt) @ T_hat` under the `T_i<-j` convention.

Uncertainty: **95% percentile cluster bootstrap**, cluster `cluster_id`, 10000 replicates, seed 20260807. Clusters, not individual encounters, are sampled with replacement.

## A. SE(2) accuracy on returned positive candidates

| Input | Method | GT+ N | Returned/rigid-valid | Translation m: median [IQR]; P90; median 95% CI | Yaw deg: median [IQR]; P90; median 95% CI |
| --- | --- | --- | --- | --- | --- |
| observed | MANUSCRIPT_RECONSTRUCTED_REFERENCE | 50 | 50/50 | 0.040 [0.024, 0.061]; 0.072; [0.036, 0.046] | 0.023 [0.011, 0.040]; 0.080; [0.014, 0.029] |
| predicted | MANUSCRIPT_RECONSTRUCTED_REFERENCE | 50 | 50/45 | 0.052 [0.031, 0.079]; 0.143; [0.038, 0.064] | 0.052 [0.028, 0.118]; 0.193; [0.036, 0.082] |
| predicted_raw | MANUSCRIPT_RECONSTRUCTED_REFERENCE | 50 | 50/49 | 0.047 [0.029, 0.078]; 0.108; [0.037, 0.062] | 0.066 [0.023, 0.110]; 0.146; [0.043, 0.089] |

## B. Selective reliability

| Input | Method | Pos/neg | Accepted | Rejected | TA | FA (% accepted) | FR (% positive) | TR | Gate violations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observed | MANUSCRIPT_RECONSTRUCTED_REFERENCE | 50/50 | 50 | 50 | 50 | 0 (0.0%; CI [0.000, 0.000]) | 0 (0.0%; CI [0.000, 0.000]) | 50 | 0 |
| predicted | MANUSCRIPT_RECONSTRUCTED_REFERENCE | 50/50 | 52 | 48 | 43 | 9 (17.3%; CI [0.000, 36.924]) | 7 (14.0%; CI [0.000, 31.373]) | 41 | 0 |
| predicted_raw | MANUSCRIPT_RECONSTRUCTED_REFERENCE | 50/50 | 47 | 53 | 46 | 1 (2.1%; CI [0.000, 6.977]) | 3 (6.0%; CI [0.000, 19.565]) | 50 | 0 |

## C. Wrong-transform injection safety

No injected events were present.

FA is reported per accepted encounter; FR is reported per GT-positive encounter. A pre-commit injection rejection has both `accepted=false` and `committed=false`. CIs are cluster-bootstrap percentile intervals.
