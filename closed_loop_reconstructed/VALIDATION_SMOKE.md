# Archived validation smoke record

This record predates the current persistent-map endpoint, recovery, summary-hash, and strict event-coverage contracts. It documents an earlier validation-only fixture run and has not been promoted into current evidence. The current implementation is tested only with synthetic validation/temp fixtures in `tests/`; no held-out campaign has been run.

Date: 2026-08-08 UTC

Environment: private headless validation host, Python 3 environment, NumPy
2.2.6 and OpenCV 4.11.0. Host-specific paths are not published.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
python3 \
  -m closed_loop_reconstructed.run_campaign \
  --source-root <validation-source-root> \
  --output-root <temporary-output-root> \
  --split val --limit-floorplans 1 --seeds 11 \
  --arms prediction_on observed_only \
  --predictor fixture --registrar reference --maximum-ticks 20
```

Scope: one validation floor (`0510030937_A_40_1_106`), seed 11, three robots, both paired arms, 20 ticks. No held-out bitmap was opened and no held-out run was launched.

Results:

- Two runs completed in 1.9881 s total (prediction arm 1.1424 s; observed control 0.7967 s).
- The paired random-stream manifest digest was identical in both arms.
- Both arms executed 60 motion attempts with zero collision rejects.
- Prediction arm: 198 events, six encounter registrations, six gate rejects, six hash-preserving noncommits.
- Observed control: 201 events, six encounter registrations, five gate rejects, one gate accept; the incorrect accepted candidate was stopped before commit by temporal consistency. There were six hash-preserving noncommits and zero unsafe registration commits.
- Under the earlier verifier, both final event chains passed and all rejected commits preserved persistent-map revision and hash. Those archived logs have not been claimed to satisfy the stricter current verifier.

This fixture result checks execution and safety invariants only. It is not evidence of prediction benefit, checkpoint quality, historical equivalence, or held-out performance. The 40 m / 640 / 256 prediction geometry remains a configurable validation-smoke default pending the replacement-training audit; protocol freezing and held-out execution are disabled.
