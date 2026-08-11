#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/data1/MSO_12method_benchmark_20260808"
OUTPUT="${ROOT}/model_data"
exec > >(tee -a "${ROOT}/logs/model_cache.log") 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ'
python3 "${ROOT}/launcher/materialize_256_cache.py" \
  --common-manifest "${ROOT}/common_data/manifest.tsv" \
  --output-root "${OUTPUT}" \
  --workers 12
(cd "${OUTPUT}" && sha256sum -c SHA256SUMS)
date -u +'%Y-%m-%dT%H:%M:%SZ' | tee "${ROOT}/MODEL_CACHE_COMPLETE"
