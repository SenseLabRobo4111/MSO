#!/usr/bin/env bash
set -Eeuo pipefail

root=/mnt/data1/MSO_12method_benchmark_20260808
launcher="$root/launcher"
common="$root/common_data/manifest.tsv"

for marker in UNET_CAMPAIGN_COMPLETE MSO_CAMPAIGN_COMPLETE ADAPTER_CAMPAIGNS_COMPLETE; do
  while ! test -f "$root/$marker"; do
    sleep 60
  done
done

methods=(MSO U-Net LaMa-Fourier MI-GAN PartialConv GatedConv EdgeConnect AOT-GAN MAT ZITS++ HINT)
for method in "${methods[@]}"; do
  safe_name="${method//+/_plus_}"
  for seed in 11 23 37 53 71; do
    run="$root/runs/$safe_name/seed_$seed"
    for split in validation test; do
      output="$root/evaluation/$safe_name/seed_$seed/$split"
      if ! test -f "$output/summary.json"; then
        docker run --rm \
          --cpus=4 \
          --memory=24g \
          --volume /mnt/data1:/mnt/data1:ro \
          --volume "$root:$root:rw" \
          --volume "$launcher:/work:ro" \
          --entrypoint python \
          mso-registration:20260808 \
          /work/evaluate_predictions.py \
          --common-manifest "$common" \
          --prediction-manifest "$run/predictions_${split}.tsv" \
          --output "$output" \
          --split "$split"
      fi
      (cd "$output" && sha256sum -c SHA256SUMS)
    done
  done
done

docker run --rm \
  --volume "$root:$root:rw" \
  --volume "$launcher:/work:ro" \
  --entrypoint python \
  mso-registration:20260808 \
  /work/aggregate_prediction_results.py \
  --root "$root" \
  --output "$root/aggregate"

printf 'UNIFIED_EVALUATION_COMPLETE\n' > "$root/UNIFIED_EVALUATION_COMPLETE"
