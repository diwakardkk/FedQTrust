#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-output/scientific_results}"
DEVICE="${DEVICE:-cuda}"
MODE="${MODE:-paper}"
ROUNDS="${ROUNDS:-}"
SEEDS="${SEEDS:-}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
MODELS="${MODELS:-classical_cnn,fedqcnn}"
DATASETS="${DATASETS:-pathmnist,octmnist,pneumoniamnist,retinamnist,breastmnist}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
SEED="${SEED:-42}"
STRICT_INFRA="${STRICT_INFRA:-true}"

echo "[FedQTrust] Scientific results suite"
echo "[FedQTrust] OUTPUT_DIR=$OUTPUT_DIR DEVICE=$DEVICE MODE=$MODE"

echo "[1/5] Unit tests"
python -m pytest -q

echo "[2/5] Smoke test"
python -m fedqtrust smoke-test --download-data --device "$DEVICE" --output-dir "$OUTPUT_DIR/smoke_test"

if [ "$MODE" = "paper" ]; then
  TRAIN_ROUNDS="${ROUNDS:-100}"
  RAW_DIR="$OUTPUT_DIR/raw_training"
  echo "[3/5] Run real training and save raw metrics"
  train_args=(
    -m fedqtrust run-gpu-once
    --download-data
    --device "$DEVICE"
    --output-dir "$RAW_DIR"
    --rounds "$TRAIN_ROUNDS"
    --batch-size "$BATCH_SIZE"
    --learning-rate "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --num-workers "$NUM_WORKERS"
    --models "$MODELS"
    --datasets "$DATASETS"
    --seed "$SEED"
  )

  if [ -n "$MAX_TRAIN_SAMPLES" ]; then
    train_args+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
  fi

  if [ -n "$MAX_EVAL_SAMPLES" ]; then
    train_args+=(--max-eval-samples "$MAX_EVAL_SAMPLES")
  fi

  if [ "$DEVICE" != "cpu" ]; then
    train_args+=(--amp --require-cuda)
  fi

  python "${train_args[@]}"

  echo "[4/5] Read real metrics and generate plots, tables, statistics, and summaries"
  real_args=(
    -m fedqtrust real-results-suite
    --source-dir "$RAW_DIR"
    --output-dir "$OUTPUT_DIR"
    --device "$DEVICE"
  )

  if [ "$STRICT_INFRA" = "true" ]; then
    real_args+=(--strict-infra)
  fi

  python "${real_args[@]}"

  echo "[5/5] Audit real-result completeness"
  audit_args=(-m fedqtrust audit-real-results --output-dir "$OUTPUT_DIR")
  if [ "$STRICT_INFRA" = "true" ]; then
    audit_args+=(--strict-infra)
  fi
  python "${audit_args[@]}"
else
  echo "[3/5] Generate fast test/sample publication artifacts"
  args=(
    -m fedqtrust publication-suite
    --mode "$MODE"
    --device "$DEVICE"
    --output-dir "$OUTPUT_DIR"
    --download-data
  )

  if [ -n "$ROUNDS" ]; then
    args+=(--rounds "$ROUNDS")
  fi

  if [ -n "$SEEDS" ]; then
    args+=(--seeds "$SEEDS")
  fi

  python "${args[@]}"

  echo "[4/5] Audit sample artifact completeness"
  python -m fedqtrust audit-publication --output-dir "$OUTPUT_DIR"
  echo "[5/5] Sample mode complete"
fi

echo "[DONE] Saved results bundle under $OUTPUT_DIR"
