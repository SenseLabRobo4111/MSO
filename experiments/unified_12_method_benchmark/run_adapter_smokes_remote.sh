#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/data1/MSO_12method_benchmark_20260808"
exec > >(tee -a "${ROOT}/logs/adapter_smokes.log") 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ'
methods=(LaMa-Fourier MI-GAN PartialConv GatedConv EdgeConnect AOT-GAN MAT ZITS++ HINT)
for method in "${methods[@]}"; do
  safe_name="${method//+/_plus_}"
  output="${ROOT}/smoke/${safe_name}_seed11"
  docker run --rm --gpus all --shm-size=16g \
    -v "${ROOT}:${ROOT}:rw" \
    pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel \
    python "${ROOT}/launcher/train_unified_adapter.py" \
      --method "${method}" \
      --seed 11 \
      --model-data-root "${ROOT}/model_data" \
      --model-manifest "${ROOT}/model_data/model_manifest.tsv" \
      --config "${ROOT}/launcher/training_config.json" \
      --output "${output}" \
      --device cuda \
      --smoke-only
done
date -u +'%Y-%m-%dT%H:%M:%SZ' | tee "${ROOT}/ADAPTER_SMOKES_COMPLETE"
