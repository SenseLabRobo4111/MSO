#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/data1/MSO_12method_benchmark_20260808"
exec > >(tee -a "${ROOT}/logs/adapter_campaigns.log") 2>&1
while ! test -f "${ROOT}/ADAPTER_SMOKES_COMPLETE"; do
  sleep 30
done
date -u +'%Y-%m-%dT%H:%M:%SZ'
methods=(LaMa-Fourier MI-GAN PartialConv GatedConv EdgeConnect AOT-GAN MAT ZITS++ HINT)
for method in "${methods[@]}"; do
  safe_name="${method//+/_plus_}"
  for seed in 11 23 37 53 71; do
    output="${ROOT}/runs/${safe_name}/seed_${seed}"
    if ! test -f "${output}/completion.json"; then
      docker run --rm --gpus all --shm-size=16g \
        -v "${ROOT}:${ROOT}:rw" \
        pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel \
        python "${ROOT}/launcher/train_unified_adapter.py" \
          --method "${method}" \
          --seed "${seed}" \
          --model-data-root "${ROOT}/model_data" \
          --model-manifest "${ROOT}/model_data/model_manifest.tsv" \
          --config "${ROOT}/launcher/training_config.json" \
          --output "${output}" \
          --device cuda
    fi
    (cd "${output}" && sha256sum -c SHA256SUMS)
  done
done
date -u +'%Y-%m-%dT%H:%M:%SZ' | tee "${ROOT}/ADAPTER_CAMPAIGNS_COMPLETE"
