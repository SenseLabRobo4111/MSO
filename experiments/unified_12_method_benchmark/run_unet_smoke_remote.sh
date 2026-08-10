#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/data1/MSO_12method_benchmark_20260808"
OUTPUT="${ROOT}/smoke/U-Net_seed11"
exec > >(tee -a "${ROOT}/logs/unet_smoke.log") 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ'
docker run --rm --gpus all --shm-size=16g \
  -v "${ROOT}:${ROOT}:rw" \
  pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel \
  python "${ROOT}/launcher/train_unified_adapter.py" \
    --method U-Net \
    --seed 11 \
    --model-data-root "${ROOT}/model_data" \
    --model-manifest "${ROOT}/model_data/model_manifest.tsv" \
    --config "${ROOT}/launcher/training_config.json" \
    --output "${OUTPUT}" \
    --device cuda \
    --smoke-only
date -u +'%Y-%m-%dT%H:%M:%SZ' | tee "${ROOT}/UNET_SMOKE_COMPLETE"
