#!/usr/bin/env bash
set -Eeuo pipefail

root=/mnt/data1/MSO_12method_benchmark_20260808
launcher="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$root/sources/MSO_public"
model_root="$root/model_data"
common_root="$root/common_data"
raw_root=/mnt/data1/SenseMapData/SenseMapDatasets
archive=/mnt/data1/ACEsplat/iros.zip
archive_prefix=iros/MapEx/lama
historical_checkpoint=/mnt/data1/SenseExpo/iros/SenseMap/src/robot_client/robot_client/config/model-epoch-deconv.ckpt
resnet_source=/mnt/data1/SenseExpo/resnetpl_weights/encoder_epoch_20.pth
base_image=pytorch/pytorch@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9
registration_image=sha256:93226e0f94504726e99477a643910dfc7f9cf209a45dd4d8bd21b08a2debb2ee
bundle_manifest="$launcher/FULL_MSO_V4_BUNDLE_SHA256SUMS"

expected_bundle="${EXPECTED_BUNDLE_SHA256:-}"
if [[ ! "$expected_bundle" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'EXPECTED_BUNDLE_SHA256 is missing or invalid\n' >&2
  exit 2
fi
if test "$(sha256sum "$bundle_manifest" | awk '{print $1}')" != "$expected_bundle"; then
  printf 'V4 bundle manifest hash differs\n' >&2
  exit 3
fi
(cd "$launcher" && sha256sum -c "$(basename "$bundle_manifest")")

exec 9>"$root/.full_mso_paper_reconstruction_v4.lock"
if ! flock -n 9; then
  printf 'another V4 reconstruction runner owns the lock\n' >&2
  exit 4
fi

bundle_short="${expected_bundle:0:12}"
image_tag="mso-full-v4:$bundle_short"
preflight="$root/full_mso_preflight_v4_$bundle_short"
campaign="$root/full_mso_paper_equation_reconstruction_v4_$bundle_short"
lama_root="$preflight/lama"
resnet_root="$preflight/resnetpl_weights"
resnet_relative=ade20k/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
resnet_staged="$resnet_root/$resnet_relative"
topology_contract="$preflight/recovered_topology.json"
derived_id_file="$preflight/DERIVED_IMAGE_ID"
v3_teacher_complete="$root/full_mso_paper_equation_reconstruction_75074e7b4a61/runs/teacher_seed_101/RUN_COMPLETE.json"

mkdir -p "$preflight"
test "$(sha256sum "$model_root/model_manifest.tsv" | awk '{print $1}')" = \
  e6423f446cb59bc5b2ca7ed51cc7cec1474fb0fec31651463f4420c816137e0f
test "$(sha256sum "$common_root/manifest.tsv" | awk '{print $1}')" = \
  25c7ef3db9c8adf85dd24727b88f79cf6e485f281fdf380ed5ce7026bd091760
test "$(sha256sum "$historical_checkpoint" | awk '{print $1}')" = \
  0e7f949925b9f6310f93e36b10152e9729a8cacef15c9061d9e361f1b7c649ad
test "$(sha256sum "$resnet_source" | awk '{print $1}')" = \
  d7dcb0234a2c1fd23d490d48c2c2fc5c39dc2b0ce39085b2f6f7e867fdd5d304
if ! test -f "$resnet_staged"; then
  mkdir -p "$(dirname "$resnet_staged")"
  test ! -e "$resnet_staged.tmp"
  ln "$resnet_source" "$resnet_staged.tmp"
  mv "$resnet_staged.tmp" "$resnet_staged"
fi

if ! docker image inspect "$image_tag" >/dev/null 2>&1; then
  docker build --pull=false --tag "$image_tag" \
    --file "$launcher/Dockerfile.full_mso" "$launcher"
fi
derived_id="$(docker image inspect "$image_tag" --format '{{.Id}}')"
base_id="$(docker image inspect "$base_image" --format '{{.Id}}')"
printf '%s\n' "$derived_id" > "$derived_id_file.tmp"
mv "$derived_id_file.tmp" "$derived_id_file"

docker run --rm --network none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g \
  --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges \
  --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPYCACHEPREFIX=/tmp/pycache \
  --mount type=bind,src="$launcher",dst=/work,readonly \
  --entrypoint python "$derived_id" -m py_compile \
  /work/train_paper_mso_reconstructed_v4.py \
  /work/train_paper_mso_reconstructed.py \
  /work/mso_full_components.py \
  /work/verify_recovered_components.py \
  /work/train_unified_adapter.py \
  /work/unified_models.py \
  /work/aggregate_full_mso_reconstruction_v4.py \
  /work/plot_full_mso_reconstruction_v4.py

if ! test -f "$lama_root/saicinpainting/training/losses/adversarial.py"; then
  dependency_tmp="$preflight/lama.tmp"
  test ! -e "$dependency_tmp"
  mkdir -p "$dependency_tmp"
  mapfile -t dependency_files < <(awk -F'  ' '{print $2}' "$launcher/lama_dependency_sha256.tsv")
  archive_files=()
  for relative in "${dependency_files[@]}"; do
    archive_files+=("$archive_prefix/$relative")
  done
  unzip -q "$archive" "${archive_files[@]}" -d "$dependency_tmp"
  mv "$dependency_tmp/$archive_prefix" "$lama_root"
  rm -rf "$dependency_tmp/iros"
  rmdir "$dependency_tmp"
fi
(cd "$lama_root" && sha256sum -c "$launcher/lama_dependency_sha256.tsv")

if ! test -f "$topology_contract"; then
  docker run --rm --network none --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,size=8g \
    --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --mount type=bind,src="$launcher",dst=/work,readonly \
    --mount type=bind,src="$source_root",dst=/src/MSO_public,readonly \
    --mount type=bind,src="$historical_checkpoint",dst=/checkpoint.ckpt,readonly \
    --mount type=bind,src="$preflight",dst=/output \
    --entrypoint python "$derived_id" \
    /work/verify_recovered_components.py \
    --trusted-checkpoint /checkpoint.ckpt \
    --source-root /src/MSO_public \
    --output /output/recovered_topology.json
fi
python3 - "$topology_contract" "$launcher/verify_recovered_components.py" <<'PY'
import hashlib, json, pathlib, sys
contract = json.loads(pathlib.Path(sys.argv[1]).read_text())
source_sha = hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest()
assert contract["status"] == "recovered_topology_verified_weights_not_reused"
assert contract["verifier_sha256"] == source_sha
assert contract["historical_weights_reused_for_initialisation"] is False
PY

while ! test -f "$v3_teacher_complete"; do
  printf 'waiting read-only for frozen V3 teacher RUN_COMPLETE: %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 60
done
python3 - "$v3_teacher_complete" <<'PY'
import hashlib, json, pathlib, sys
marker_path = pathlib.Path(sys.argv[1])
root = marker_path.parent
marker = json.loads(marker_path.read_text())
completion = json.loads((root / "completion.json").read_text())
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert marker["completed"] is True
assert completion["completed"] is True
assert completion["stage"] == "teacher"
assert int(completion["seed"]) == 101
assert int(completion["last_epoch"]) == 499
assert marker["completion_json_sha256"] == digest(root / "completion.json")
assert marker["sha256_manifest_sha256"] == digest(root / "SHA256SUMS")
PY
printf 'observed frozen V3 teacher completion; V4 remains a separate run\n'

while true; do
  free_mib="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if test "$free_mib" -ge 81920; then
    break
  fi
  printf 'waiting for at least 80 GiB free GPU memory; now %s MiB\n' "$free_mib"
  sleep 60
done
free_bytes="$(df --output=avail -B1 "$root" | tail -1 | tr -d ' ')"
if test "$free_bytes" -lt 322122547200; then
  printf 'less than 300 GiB free disk; refusing snapshot-rich V4 campaign\n' >&2
  exit 7
fi

mkdir -p "$campaign/logs" "$campaign/runs" "$campaign/evaluation"
exec > >(tee -a "$campaign/logs/campaign.log") 2>&1
failure_marker="$campaign/FAILED"
complete_marker="$campaign/FULL_MSO_PAPER_EQUATION_RECONSTRUCTION_V4_COMPLETE"
on_exit() {
  rc=$?
  if test "$rc" -ne 0; then
    printf 'exit=%s utc=%s\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$failure_marker.tmp"
    mv "$failure_marker.tmp" "$failure_marker"
  fi
}
trap on_exit EXIT
if test -f "$complete_marker"; then
  printf 'V4 reconstruction already complete\n'
  exit 0
fi
rm -f "$failure_marker"
{
  printf 'bundle_sha256=%s\n' "$expected_bundle"
  printf 'derived_image_id=%s\n' "$derived_id"
  printf 'base_image_id=%s\n' "$base_id"
  printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi
} > "$campaign/logs/environment_start.txt"

run_stage() {
  local stage="$1"
  local seed="$2"
  local output="$3"
  local smoke="$4"
  local teacher_checkpoint="${5:-}"
  local teacher_sha="${6:-}"
  mkdir -p "$output"
  local mounts=(
    --mount type=bind,src="$launcher",dst=/work,readonly
    --mount type=bind,src="$model_root",dst=/data/model,readonly
    --mount type=bind,src="$source_root",dst=/src/MSO_public,readonly
    --mount type=bind,src="$lama_root",dst=/deps/lama,readonly
    --mount type=bind,src="$resnet_root",dst=/deps/resnet,readonly
    --mount type=bind,src="$topology_contract",dst=/contracts/topology.json,readonly
    --mount type=bind,src="$output",dst=/output
  )
  local teacher_args=()
  if test -n "$teacher_checkpoint"; then
    test "$(sha256sum "$teacher_checkpoint" | awk '{print $1}')" = "$teacher_sha"
    mounts+=(--mount type=bind,src="$teacher_checkpoint",dst=/teacher/best_primary.pt,readonly)
    teacher_args=(
      --teacher-checkpoint /teacher/best_primary.pt
      --teacher-checkpoint-sha256 "$teacher_sha"
    )
  fi
  local mode_args=(--resume)
  local command_prefix=()
  if test "$smoke" = true; then
    mode_args=(--smoke-only)
    command_prefix=(timeout --signal=TERM --kill-after=2m 30m)
  fi
  "${command_prefix[@]}" docker run --rm --gpus 'device=0' \
    --network none --read-only --tmpfs /tmp:rw,nosuid,nodev,size=8g \
    --shm-size=24g --user 1000:1000 --cap-drop ALL \
    --security-opt no-new-privileges --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONPYCACHEPREFIX=/tmp/pycache --env PYTHONHASHSEED="$seed" \
    --env MSO_CONTAINER_IMAGE_ID="$derived_id" \
    --env MSO_BUNDLE_SHA256="$expected_bundle" \
    "${mounts[@]}" --entrypoint python "$derived_id" \
    /work/train_paper_mso_reconstructed_v4.py \
    --stage "$stage" --seed "$seed" \
    --model-data-root /data/model \
    --model-manifest /data/model/model_manifest.tsv \
    --config /work/full_mso_config_v4.json \
    --source-root /src/MSO_public \
    --topology-contract /contracts/topology.json \
    --dependency-manifest /work/lama_dependency_sha256.tsv \
    --lama-root /deps/lama --resnet-weights-root /deps/resnet \
    --output /output --device cuda \
    "${teacher_args[@]}" "${mode_args[@]}"
}

verify_completion() {
  local run="$1"
  local stage="$2"
  local seed="$3"
  local expected_teacher_sha="${4:-}"
  (cd "$run" && sha256sum -c SHA256SUMS)
  python3 - "$run" "$stage" "$seed" "$expected_teacher_sha" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
value = json.loads((root / "completion.json").read_text())
run_manifest = json.loads((root / "run_manifest.json").read_text())
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert value["completed"] is True
assert value["stage"] == sys.argv[2]
assert int(value["seed"]) == int(sys.argv[3])
assert value["protocol_id"] == "mso_paper_equation_reconstruction_v4"
assert run_manifest["protocol_id"] == value["protocol_id"]
assert int(value["last_epoch"]) == 499
assert value["selection_metric"] == "validation_unknown_occupied_micro_f1_at_0.5"
assert value["best_primary_checkpoint_sha256"] == digest(root / "best_primary.pt")
assert value["best_calibration_checkpoint_sha256"] == digest(root / "best_calibration.pt")
assert value["last_checkpoint_sha256"] == digest(root / "last.pt")
assert value["selection_manifest_sha256"] == digest(root / "selection_manifest.json")
selection = json.loads((root / "selection_manifest.json").read_text())
assert selection["source"] == "independent_full_metrics_tsv_recomputation"
assert selection["protocol_id"] == value["protocol_id"]
assert selection["stage"] == value["stage"]
assert int(selection["seed"]) == int(value["seed"])
assert selection["primary"]["epoch"] == value["best_primary_epoch"]
assert selection["calibration"]["epoch"] == value["best_calibration_epoch"]
assert selection["primary"]["checkpoint_sha256"] == value["best_primary_checkpoint_sha256"]
assert selection["calibration"]["checkpoint_sha256"] == value["best_calibration_checkpoint_sha256"]
assert "best_validation_unknown_bce" not in value
assert value["best_primary_validation_unknown_bce"] == selection["primary"]["validation_metrics"]["unknown_bce"]
if value["stage"] == "student":
    expected_teacher_sha = sys.argv[4]
    assert len(expected_teacher_sha) == 64
    assert all(character in "0123456789abcdef" for character in expected_teacher_sha)
    assert value["parent_teacher_checkpoint_kind"] == "validation_best_primary_generator"
    assert value["parent_teacher_checkpoint_sha256"] == expected_teacher_sha
    assert run_manifest["parent_teacher_checkpoint_kind"] == "validation_best_primary_generator"
    assert run_manifest["parent_teacher_checkpoint_sha256"] == expected_teacher_sha
    assert selection["run_provenance"]["parent_teacher_checkpoint_kind"] == "validation_best_primary_generator"
    assert selection["run_provenance"]["parent_teacher_checkpoint_sha256"] == expected_teacher_sha
else:
    assert sys.argv[4] == ""
    assert value["parent_teacher_checkpoint_kind"] is None
    assert value["parent_teacher_checkpoint_sha256"] is None
assert len(value["periodic_snapshot_sha256"]) == 50
for relative, expected in value["periodic_snapshot_sha256"].items():
    assert digest(root / relative) == expected
marker = json.loads((root / "RUN_COMPLETE.json").read_text())
assert marker["completed"] is True
assert marker["completion_json_sha256"] == digest(root / "completion.json")
assert marker["sha256_manifest_sha256"] == digest(root / "SHA256SUMS")
PY
}

run_smoke() {
  local stage="$1"
  local seed="$2"
  local final="$3"
  local teacher_checkpoint="${4:-}"
  local teacher_sha="${5:-}"
  local temporary="${final}.tmp"
  test ! -e "$temporary"
  mkdir -p "$(dirname "$final")"
  run_stage "$stage" "$seed" "$temporary" true "$teacher_checkpoint" "$teacher_sha"
  test -f "$temporary/smoke.json"
  mv "$temporary" "$final"
}

teacher_smoke="$campaign/smoke/teacher_seed_101"
if ! test -f "$teacher_smoke/smoke.json"; then
  test ! -e "$teacher_smoke"
  run_smoke teacher 101 "$teacher_smoke"
fi
teacher_run="$campaign/runs/teacher_seed_101"
if test -f "$teacher_run/RUN_COMPLETE.json"; then
  verify_completion "$teacher_run" teacher 101
else
  run_stage teacher 101 "$teacher_run" false
  verify_completion "$teacher_run" teacher 101
fi
teacher_checkpoint="$teacher_run/best_primary.pt"
teacher_sha="$(sha256sum "$teacher_checkpoint" | awk '{print $1}')"
python3 - "$teacher_run/completion.json" "$teacher_sha" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert value["best_primary_checkpoint_sha256"] == sys.argv[2]
PY

student_smoke="$campaign/smoke/student_seed_11"
if ! test -f "$student_smoke/smoke.json"; then
  test ! -e "$student_smoke"
  run_smoke student 11 "$student_smoke" "$teacher_checkpoint" "$teacher_sha"
fi

for seed in 11 23 37 53 71; do
  run="$campaign/runs/student_seed_$seed"
  if test -f "$run/RUN_COMPLETE.json"; then
    verify_completion "$run" student "$seed" "$teacher_sha"
  else
    run_stage student "$seed" "$run" false "$teacher_checkpoint" "$teacher_sha"
    verify_completion "$run" student "$seed" "$teacher_sha"
  fi
  for split in validation test; do
    parent="$campaign/evaluation/student_seed_$seed"
    output="$parent/$split"
    if test -d "$output"; then
      (cd "$output" && sha256sum -c SHA256SUMS)
      continue
    fi
    test ! -e "$output"
    temporary="${output}.tmp"
    test ! -e "$temporary"
    mkdir -p "$parent"
    docker run --rm --network none --read-only --cpus=4 --memory=24g \
      --tmpfs /tmp:rw,nosuid,nodev,size=2g --user 1000:1000 \
      --cap-drop ALL --security-opt no-new-privileges \
      --env PYTHONDONTWRITEBYTECODE=1 \
      --mount type=bind,src="$launcher",dst=/work,readonly \
      --mount type=bind,src="$common_root",dst=/data/common,readonly \
      --mount type=bind,src="$raw_root",dst="$raw_root",readonly \
      --mount type=bind,src="$run",dst=/run,readonly \
      --mount type=bind,src="$parent",dst=/evaluation \
      --entrypoint python "$registration_image" \
      /work/evaluate_predictions.py \
      --common-manifest /data/common/manifest.tsv \
      --prediction-manifest "/run/predictions_${split}.tsv" \
      --output "/evaluation/${split}.tmp" --split "$split"
    (cd "$temporary" && sha256sum -c SHA256SUMS)
    mv "$temporary" "$output"
  done
done

aggregate="$campaign/aggregate"
if ! test -d "$aggregate"; then
  aggregate_tmp="${aggregate}.tmp"
  test ! -e "$aggregate_tmp"
  /home/haohua/miniconda3/envs/occbev/bin/python \
    "$launcher/aggregate_full_mso_reconstruction_v4.py" \
    --campaign "$campaign" --uniform-aggregate "$root/aggregate" \
    --output "$aggregate_tmp"
  (cd "$aggregate_tmp" && sha256sum -c SHA256SUMS)
  mv "$aggregate_tmp" "$aggregate"
fi
(cd "$aggregate" && sha256sum -c SHA256SUMS)

figures="$campaign/visualizations"
if ! test -d "$figures"; then
  figures_tmp="${figures}.tmp"
  test ! -e "$figures_tmp"
  /home/haohua/miniconda3/envs/occbev/bin/python \
    "$launcher/plot_full_mso_reconstruction_v4.py" \
    --aggregate "$aggregate" --output "$figures_tmp"
  (cd "$figures_tmp" && sha256sum -c SHA256SUMS)
  mv "$figures_tmp" "$figures"
fi
(cd "$figures" && sha256sum -c SHA256SUMS)

{
  printf 'bundle_sha256=%s\n' "$expected_bundle"
  printf 'derived_image_id=%s\n' "$derived_id"
  printf 'teacher_best_primary_sha256=%s\n' "$teacher_sha"
  printf 'complete_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$complete_marker.tmp"
mv "$complete_marker.tmp" "$complete_marker"
trap - EXIT
printf 'V4 reconstruction complete UTC %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
