#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/data1/MSO_12method_benchmark_20260808"
docker pull pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel \
  2>&1 | tee "${ROOT}/logs/pytorch_image_pull.log"
docker pull tensorflow/tensorflow:2.19.0-gpu \
  2>&1 | tee "${ROOT}/logs/tensorflow_image_pull.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' | tee "${ROOT}/ENV_IMAGES_COMPLETE"
