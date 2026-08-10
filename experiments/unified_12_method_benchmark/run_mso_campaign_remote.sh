#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/data1/MSO_12method_benchmark_20260808"
exec > >(tee -a "${ROOT}/logs/mso_campaign.log") 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ'
smoke="${ROOT}/smoke/MSO_seed11"
if ! test -f "${smoke}/completion.json"; then
  docker run --rm --gpus all --shm-size=16g \
    -e MSO_ADAPTER_ROOT="${ROOT}/sources/MSO_public" \
    -v "${ROOT}:${ROOT}:rw" \
    pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel \
    python "${ROOT}/launcher/train_unified_adapter.py" \
      --method MSO \
      --seed 11 \
      --model-data-root "${ROOT}/model_data" \
      --model-manifest "${ROOT}/model_data/model_manifest.tsv" \
      --config "${ROOT}/launcher/training_config.json" \
      --output "${smoke}" \
      --device cuda \
      --smoke-only
fi
for seed in 11 23 37 53 71; do
  output="${ROOT}/runs/MSO/seed_${seed}"
  if ! test -f "${output}/completion.json"; then
    docker run --rm --gpus all --shm-size=16g \
      -e MSO_ADAPTER_ROOT="${ROOT}/sources/MSO_public" \
      -v "${ROOT}:${ROOT}:rw" \
      pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel \
      python "${ROOT}/launcher/train_unified_adapter.py" \
        --method MSO \
        --seed "${seed}" \
        --model-data-root "${ROOT}/model_data" \
        --model-manifest "${ROOT}/model_data/model_manifest.tsv" \
        --config "${ROOT}/launcher/training_config.json" \
        --output "${output}" \
        --device cuda
  fi
  (cd "${output}" && sha256sum -c SHA256SUMS)
done
date -u +'%Y-%m-%dT%H:%M:%SZ' | tee "${ROOT}/MSO_CAMPAIGN_COMPLETE"
