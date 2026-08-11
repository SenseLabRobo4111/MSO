# Prediction output contract

Each method and seed must write one immutable run directory containing:

- `run_manifest.json`: method, seed, source revision, environment hash,
  training manifest hash, selected checkpoint hash, selection metric and timing;
- `metrics_by_epoch.tsv`: every completed epoch, without deleted or rewritten
  rows;
- `predictions.tsv`: one row for every validation or test sample;
- `probabilities/<sample_id>.npy`: a finite float32 256 x 256 occupied-cell
  probability raster in [0, 1];
- `resource_profile.json`: parameter count, profiler convention, latency sample
  count, peak VRAM and training time;
- `SHA256SUMS`: every result file except the checksum list itself.

`predictions.tsv` columns are fixed as:

```text
method	seed	split	sample_id	probability_path	probability_sha256	inference_ms
```

The common evaluator, not an individual method adapter, performs thresholding,
measured-cell copy-back and all reported metrics. Adapters cannot omit difficult
samples or use method-specific evaluation crops.

The benchmark is prospective over the preserved 6,841/1,463 archive. It is not
the unrecovered historical 5,385/1,356 experiment and is not described as
building-disjoint.
