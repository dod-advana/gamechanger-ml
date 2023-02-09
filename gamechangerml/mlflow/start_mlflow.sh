#!/bin/sh
set -e
mlflow server \
      --backend-store-uri file:/mnt/mlruns \
      --default-artifact-root file:/mnt/mlrun \
      --host 0.0.0.0 \
      --port 5050

