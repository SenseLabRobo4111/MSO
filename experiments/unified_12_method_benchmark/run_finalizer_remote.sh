#!/usr/bin/env bash
set -Eeuo pipefail

root=/mnt/data1/MSO_12method_benchmark_20260808
launcher="$root/launcher"

while ! test -f "$root/UNIFIED_EVALUATION_COMPLETE"; do
  sleep 60
done

(cd "$root/aggregate" && sha256sum -c SHA256SUMS)

if test -e "$root/final_visualizations"; then
  printf 'refusing to overwrite existing final_visualizations\n' >&2
  exit 1
fi

python3 "$launcher/plot_aggregate_results.py" \
  --aggregate "$root/aggregate" \
  --output "$root/final_visualizations"

(cd "$root/final_visualizations" && sha256sum -c SHA256SUMS)
printf 'FINAL_VISUALIZATIONS_COMPLETE\n' > "$root/FINAL_VISUALIZATIONS_COMPLETE"
