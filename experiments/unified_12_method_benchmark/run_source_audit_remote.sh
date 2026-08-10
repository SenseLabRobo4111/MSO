#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/data1/MSO_12method_benchmark_20260808"
mkdir -p "${ROOT}/logs"
exec > >(tee -a "${ROOT}/logs/source_audit.log") 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ'
python3 "${ROOT}/launcher/audit_sources.py"
date -u +'%Y-%m-%dT%H:%M:%SZ' | tee "${ROOT}/SOURCE_AUDIT_COMPLETE"
