#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/data1/MSO_12method_benchmark_20260808"
while ! test -f "${ROOT}/UNET_CAMPAIGN_COMPLETE"; do
  sleep 30
done
bash "${ROOT}/launcher/run_mso_campaign_remote.sh"
bash "${ROOT}/launcher/run_adapter_smokes_remote.sh"
bash "${ROOT}/launcher/run_adapter_campaigns_remote.sh"
bash "${ROOT}/launcher/run_evaluation_remote.sh"
