#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/data1/MSO_12method_benchmark_20260808"
while ! test -f "${ROOT}/ENV_IMAGES_COMPLETE"; do
  sleep 5
done

docker run --rm --gpus all \
  -v "${ROOT}:${ROOT}:rw" \
  pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel \
  python -c "import json,torch; x=torch.randn(2,3,256,256,device='cuda'); y=torch.nn.Conv2d(3,16,3,padding=1,device='cuda')(x); print(json.dumps({'framework':'torch','version':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'shape':list(y.shape),'finite':bool(torch.isfinite(y).all())}))" \
  | tee "${ROOT}/compatibility/pytorch_gpu_smoke.json"

docker run --rm --gpus all \
  -v "${ROOT}:${ROOT}:rw" \
  tensorflow/tensorflow:2.19.0-gpu \
  python -c "import json,tensorflow as tf; x=tf.random.normal([2,256,256,3]); y=tf.keras.layers.Conv2D(16,3,padding='same')(x); print(json.dumps({'framework':'tensorflow','version':tf.__version__,'gpus':[d.name for d in tf.config.list_physical_devices('GPU')],'shape':list(y.shape),'finite':bool(tf.reduce_all(tf.math.is_finite(y)).numpy())}))" \
  | tee "${ROOT}/compatibility/tensorflow_gpu_smoke.json"

sha256sum "${ROOT}/compatibility/"*_gpu_smoke.json \
  > "${ROOT}/compatibility/GPU_SMOKE_SHA256SUMS"
date -u +'%Y-%m-%dT%H:%M:%SZ' | tee "${ROOT}/GPU_SMOKE_COMPLETE"
