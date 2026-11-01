#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
DATA="${DATA:-/mnt/c/Users/wuche/Downloads/LAV-DF}"
COHORT="${COHORT:-/home/evan/lavdf5000_visual_manifest/videos.jsonl}"
FEATURES="${FEATURES:-$PWD/features/rgb_flow_train_dev}"
RUN="${RUN:-$PWD/runs/umma_rgb_flow_s42}"
DEVICE="${DEVICE:-cuda:0}"
BATCH="${BATCH:-2}"
# Run full pipeline only after setup and pilot pass. Extraction resumes matching clips.
python extract_rgb_flow.py --data-root "$DATA" --cohort-videos "$COHORT" \
  --out "$FEATURES" --splits train dev --sample-fps 10 --device "$DEVICE" --encode-batch-size 32
python run_experiment.py smoke --manifest "$FEATURES/manifest.json" \
  --input-mode rgb-flow --device "$DEVICE" --batch-size "$BATCH" --steps 2 --amp none
python run_experiment.py train --manifest "$FEATURES/manifest.json" \
  --input-mode rgb-flow --out "$RUN" --device "$DEVICE" --batch-size "$BATCH" \
  --workers 2 --epochs 15 --warmup-epochs 5 --lr 0.001 --weight-decay 0.05 \
  --patience 8 --seed 42 --amp none
