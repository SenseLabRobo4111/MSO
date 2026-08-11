#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/data1/MSO_12method_benchmark_20260808"
exec > >(tee -a "${ROOT}/logs/common_data.log") 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ'
python3 "${ROOT}/launcher/prepare_common_manifests.py"
(cd "${ROOT}/common_data" && sha256sum -c SHA256SUMS)
date -u +'%Y-%m-%dT%H:%M:%SZ' | tee "${ROOT}/COMMON_DATA_COMPLETE"
