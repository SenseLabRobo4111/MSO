#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON_BIN:?Set PYTHON_BIN to the experiment Python interpreter}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the frozen source snapshot}"
: "${GT_ROOT:?Set GT_ROOT to the 156-floorplan source root}"
: "${CAMPAIGN_ROOT:?Set CAMPAIGN_ROOT to a new empty campaign directory}"

if [[ -e "${CAMPAIGN_ROOT}/CAMPAIGN_COMPLETE" ]]; then
  printf 'Campaign is already complete: %s\n' "${CAMPAIGN_ROOT}" >&2
  exit 2
fi
if [[ -d "${CAMPAIGN_ROOT}" ]] && [[ -n "$(find "${CAMPAIGN_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  printf 'Refusing non-empty campaign root: %s\n' "${CAMPAIGN_ROOT}" >&2
  exit 2
fi

mkdir -p "${CAMPAIGN_ROOT}/logs" "${CAMPAIGN_ROOT}/runs"
printf '%s\n' "$$" > "${CAMPAIGN_ROOT}/campaign.pid"
printf 'PYTHON_BIN=%q SOURCE_ROOT=%q GT_ROOT=%q CAMPAIGN_ROOT=%q bash %q\n' \
  "${PYTHON_BIN}" "${SOURCE_ROOT}" "${GT_ROOT}" "${CAMPAIGN_ROOT}" "$0" \
  > "${CAMPAIGN_ROOT}/launch_command.txt"

CURRENT_STAGE=initializing
record_state() {
  "${PYTHON_BIN}" - "${CAMPAIGN_ROOT}/campaign_state.json" "$1" "${CURRENT_STAGE}" "$2" <<'PY'
import datetime
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "detail": sys.argv[4],
    "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}
on_failure() {
  code=$?
  record_state failed "exit_code=${code}"
  exit "${code}"
}
trap on_failure ERR

record_state running started
"${PYTHON_BIN}" - "${CAMPAIGN_ROOT}/environment.json" <<'PY'
import json
import platform
import pathlib
import sys
import torch
import numpy
payload = {
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "torch": torch.__version__,
    "numpy": numpy.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cuda_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
}
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
find "${SOURCE_ROOT}" -type f \( -name '*.py' -o -name '*.json' -o -name '*.sh' \) -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${CAMPAIGN_ROOT}/source_sha256.txt"

DATASET_ROOT="${CAMPAIGN_ROOT}/dataset_48_per_floorplan"
CURRENT_STAGE=generate_dataset
record_state running generating_7488_samples
"${PYTHON_BIN}" "${SOURCE_ROOT}/generate_floorplan_dataset.py" \
  --source-root "${GT_ROOT}" \
  --output-root "${DATASET_ROOT}" \
  --samples-per-floorplan 48 \
  2>&1 | tee "${CAMPAIGN_ROOT}/logs/01_generate_dataset.log"

CURRENT_STAGE=validate_dataset
record_state running validating_split_and_hashes
"${PYTHON_BIN}" "${SOURCE_ROOT}/validate_generated_dataset.py" \
  --dataset-root "${DATASET_ROOT}" \
  --manifest "${DATASET_ROOT}/manifest.csv" \
  --expected-buildings 146 \
  --expected-floorplans 156 \
  --expected-samples 7488 \
  --verify-images \
  2>&1 | tee "${CAMPAIGN_ROOT}/logs/02_validate_dataset.log"
sha256sum "${DATASET_ROOT}/manifest.csv" "${DATASET_ROOT}/floorplan_split.csv" \
  "${DATASET_ROOT}/dataset_summary.json" > "${CAMPAIGN_ROOT}/dataset_artifact_sha256.txt"

TEACHER_RUN="${CAMPAIGN_ROOT}/runs/teacher_seed_101"
CURRENT_STAGE=teacher_seed_101
record_state running training_teacher
"${PYTHON_BIN}" "${SOURCE_ROOT}/train_reconstructed.py" \
  --stage teacher \
  --dataset-root "${DATASET_ROOT}" \
  --manifest "${DATASET_ROOT}/manifest.csv" \
  --output-dir "${TEACHER_RUN}" \
  --seed 101 \
  2>&1 | tee "${CAMPAIGN_ROOT}/logs/03_teacher_seed_101.log"
sha256sum "${TEACHER_RUN}/best.pt" "${TEACHER_RUN}/last.pt" \
  "${TEACHER_RUN}/metrics.csv" "${TEACHER_RUN}/completion.json" \
  > "${TEACHER_RUN}/artifact_sha256.txt"

for seed in 11 23 37 53 71; do
  STUDENT_RUN="${CAMPAIGN_ROOT}/runs/student_seed_${seed}"
  CURRENT_STAGE="student_seed_${seed}"
  record_state running "training_student_seed_${seed}"
  "${PYTHON_BIN}" "${SOURCE_ROOT}/train_reconstructed.py" \
    --stage student \
    --teacher-checkpoint "${TEACHER_RUN}/best.pt" \
    --dataset-root "${DATASET_ROOT}" \
    --manifest "${DATASET_ROOT}/manifest.csv" \
    --output-dir "${STUDENT_RUN}" \
    --seed "${seed}" \
    2>&1 | tee "${CAMPAIGN_ROOT}/logs/04_student_seed_${seed}.log"
  sha256sum "${STUDENT_RUN}/best.pt" "${STUDENT_RUN}/last.pt" \
    "${STUDENT_RUN}/metrics.csv" "${STUDENT_RUN}/completion.json" \
    > "${STUDENT_RUN}/artifact_sha256.txt"
done

CURRENT_STAGE=complete
record_state complete all_training_stages_completed
touch "${CAMPAIGN_ROOT}/CAMPAIGN_COMPLETE"
trap - ERR
