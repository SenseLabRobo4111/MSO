#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/data1/MSO_12method_benchmark_20260808"
DATASET="/mnt/data1/SenseMapData/SenseMapDatasets"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT}/logs"
SOURCE_DIR="${ROOT}/sources"
LOCK_DIR="${ROOT}/locks"

mkdir -p "${LOG_DIR}" "${SOURCE_DIR}" "${LOCK_DIR}"
exec > >(tee -a "${LOG_DIR}/bootstrap.log") 2>&1

date -u +'%Y-%m-%dT%H:%M:%SZ'
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 --version
docker info --format 'docker={{.ServerVersion}}'

test -d "${DATASET}/train"
test -d "${DATASET}/test"
TRAIN_COUNT="$(find "${DATASET}/train" -mindepth 1 -maxdepth 1 -type d | wc -l)"
TEST_COUNT="$(find "${DATASET}/test" -mindepth 1 -maxdepth 1 -type d | wc -l)"
test "${TRAIN_COUNT}" -eq 6841
test "${TEST_COUNT}" -eq 1463
printf 'train_samples=%s\ntest_samples=%s\n' "${TRAIN_COUNT}" "${TEST_COUNT}" \
  | tee "${LOCK_DIR}/dataset_counts.txt"

find "${DATASET}" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${LOCK_DIR}/dataset_files.sha256"
sha256sum "${LOCK_DIR}/dataset_files.sha256" \
  > "${LOCK_DIR}/dataset_inventory.sha256"

cp "${SCRIPT_DIR}/baseline_manifest.tsv" "${LOCK_DIR}/baseline_manifest.tsv"
sha256sum "${LOCK_DIR}/baseline_manifest.tsv" \
  > "${LOCK_DIR}/baseline_manifest.sha256"

tail -n +2 "${LOCK_DIR}/baseline_manifest.tsv" \
  | while IFS=$'\t' read -r method family repository commit role; do
      case "${repository}" in
        https://*)
          destination="${SOURCE_DIR}/${method}"
          if test -d "${destination}/.git"; then
            git -C "${destination}" fetch --all --tags --prune
          else
            git clone --filter=blob:none "${repository}" "${destination}"
          fi
          git -C "${destination}" checkout --detach "${commit}"
          actual="$(git -C "${destination}" rev-parse HEAD)"
          test "${actual}" = "${commit}"
          printf '%s\t%s\n' "${method}" "${actual}" \
            >> "${LOCK_DIR}/resolved_revisions.tsv"
          ;;
      esac
    done

find "${SOURCE_DIR}" -mindepth 2 -maxdepth 2 -name .git -type d \
  | sort \
  | while read -r git_dir; do
      repo_dir="$(dirname "${git_dir}")"
      git -C "${repo_dir}" status --short
    done \
  > "${LOCK_DIR}/source_worktrees.status"
test ! -s "${LOCK_DIR}/source_worktrees.status"

python3 - <<'PY'
from pathlib import Path
from PIL import Image
import json

root = Path('/mnt/data1/SenseMapData/SenseMapDatasets')
rows = []
for split in ('train', 'test'):
    samples = sorted(p for p in (root / split).iterdir() if p.is_dir())
    for sample in samples[:32]:
        obs = Image.open(sample / 'local_obs_0.png')
        target = Image.open(sample / 'local_map_0.png')
        rows.append({
            'split': split,
            'sample': sample.name,
            'obs_mode': obs.mode,
            'obs_size': list(obs.size),
            'target_mode': target.mode,
            'target_size': list(target.size),
        })
out = Path('/mnt/data1/MSO_12method_benchmark_20260808/locks/image_contract_probe.json')
out.write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

sha256sum "${LOCK_DIR}"/* > "${ROOT}/BOOTSTRAP_SHA256SUMS"
date -u +'%Y-%m-%dT%H:%M:%SZ' | tee "${ROOT}/BOOTSTRAP_COMPLETE"
