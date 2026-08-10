#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON_BIN:?Set PYTHON_BIN}"
: "${EVAL_SOURCE_ROOT:?Set EVAL_SOURCE_ROOT}"
: "${CAMPAIGN_ROOT:?Set CAMPAIGN_ROOT}"

POST_ROOT="${CAMPAIGN_ROOT}/postprocess"
if [[ -d "${POST_ROOT}" ]] && [[ -n "$(find "${POST_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  printf 'Refusing non-empty postprocess root: %s\n' "${POST_ROOT}" >&2
  exit 2
fi
mkdir -p "${POST_ROOT}/logs" "${POST_ROOT}/evaluations"
printf '%s\n' "$$" > "${POST_ROOT}/postprocess.pid"
find "${EVAL_SOURCE_ROOT}" -type f \( -name '*.py' -o -name '*.sh' \) -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${POST_ROOT}/source_sha256.txt"

write_state() {
  "${PYTHON_BIN}" - "${POST_ROOT}/state.json" "$1" "$2" <<'PY'
import datetime
import json
import pathlib
import sys
payload = {
    "status": sys.argv[2],
    "detail": sys.argv[3],
    "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}
on_failure() {
  code=$?
  write_state failed "exit_code=${code}"
  exit "${code}"
}
trap on_failure ERR
write_state waiting waiting_for_training_campaign

while [[ ! -e "${CAMPAIGN_ROOT}/CAMPAIGN_COMPLETE" ]]; do
  if grep -q '"status": "failed"' "${CAMPAIGN_ROOT}/campaign_state.json" 2>/dev/null; then
    write_state stopped training_campaign_failed
    exit 3
  fi
  sleep 60
done

write_state running evaluating_five_selected_students
DATASET_ROOT="${CAMPAIGN_ROOT}/dataset_48_per_floorplan"
SUMMARY_PATHS=()
for seed in 11 23 37 53 71; do
  OUTPUT="${POST_ROOT}/evaluations/student_seed_${seed}"
  "${PYTHON_BIN}" "${EVAL_SOURCE_ROOT}/evaluate_reconstructed.py" \
    --dataset-root "${DATASET_ROOT}" \
    --manifest "${DATASET_ROOT}/manifest.csv" \
    --checkpoint "${CAMPAIGN_ROOT}/runs/student_seed_${seed}/best.pt" \
    --output-dir "${OUTPUT}" \
    --bootstrap-replicates 10000 \
    2>&1 | tee "${POST_ROOT}/logs/student_seed_${seed}_test.log"
  sha256sum "${OUTPUT}/summary.json" "${OUTPUT}/per_sample_metrics.csv" \
    > "${OUTPUT}/artifact_sha256.txt"
  SUMMARY_PATHS+=("${OUTPUT}/summary.json")
done

"${PYTHON_BIN}" "${EVAL_SOURCE_ROOT}/summarize_multiseed.py" \
  --summaries "${SUMMARY_PATHS[@]}" \
  --output-dir "${POST_ROOT}/multiseed" \
  2>&1 | tee "${POST_ROOT}/logs/multiseed_summary.log"
sha256sum "${POST_ROOT}/multiseed/multiseed_summary.json" \
  "${POST_ROOT}/multiseed/multiseed_summary.csv" \
  > "${POST_ROOT}/multiseed/artifact_sha256.txt"

touch "${POST_ROOT}/POSTPROCESS_COMPLETE"
write_state complete evaluation_and_multiseed_summary_complete
trap - ERR
