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
RUN_E1="${RUN_E1:-false}"
RUN_ALL_REAL="${RUN_ALL_REAL:-false}"
E1_ROUNDS="${E1_ROUNDS:-${ROUNDS:-100}}"
E1_SEEDS="${E1_SEEDS:-42,123,456,789,999}"
E1_DATASETS="${E1_DATASETS:-pathmnist,octmnist,pneumoniamnist,retinamnist,breastmnist}"
E1_METHODS="${E1_METHODS:-FedAvg,FedProx,FedQCNN,Krum,FLTrust,PQS-BFL,SecEdge-MC,FedQTrust}"
E1_ATTACKS="${E1_ATTACKS:-benign,label_flip,sign_flip,gaussian,lie,free_riding,combined}"
E1_CLIENTS="${E1_CLIENTS:-10}"
E1_CLIENTS_PER_ROUND="${E1_CLIENTS_PER_ROUND:-5}"
E1_LOCAL_EPOCHS="${E1_LOCAL_EPOCHS:-1}"
E1_MAX_TRAIN_SAMPLES="${E1_MAX_TRAIN_SAMPLES:-}"
E1_MAX_EVAL_SAMPLES="${E1_MAX_EVAL_SAMPLES:-}"

echo "[FedQTrust] Scientific results suite"
echo "[FedQTrust] OUTPUT_DIR=$OUTPUT_DIR DEVICE=$DEVICE MODE=$MODE"

echo "[1/5] Unit tests"
python -m pytest -q

echo "[2/5] Smoke test"
python -m fedqtrust smoke-test --download-data --device "$DEVICE" --output-dir "$OUTPUT_DIR/smoke_test"

if [ "$MODE" = "paper" ]; then
  if [ "$RUN_ALL_REAL" = "true" ]; then
    PAPER_ROUNDS="${ROUNDS:-100}"
    echo "[3/3] Run complete real paper E1-E9 workflow"
    paper_args=(
      -m fedqtrust run-paper-real
      --download-data
      --device "$DEVICE"
      --output-dir "$OUTPUT_DIR"
      --rounds "$PAPER_ROUNDS"
      --e3-rounds "${E3_ROUNDS:-200}"
      --seeds "$E1_SEEDS"
      --datasets "$E1_DATASETS"
      --methods "$E1_METHODS"
      --attacks "$E1_ATTACKS"
      --clients "$E1_CLIENTS"
      --clients-per-round "$E1_CLIENTS_PER_ROUND"
      --batch-size "$BATCH_SIZE"
      --learning-rate "$LEARNING_RATE"
      --weight-decay "$WEIGHT_DECAY"
      --num-workers "$NUM_WORKERS"
    )
    if [ -n "$E1_MAX_TRAIN_SAMPLES" ]; then
      paper_args+=(--max-train-samples "$E1_MAX_TRAIN_SAMPLES")
    fi
    if [ -n "$E1_MAX_EVAL_SAMPLES" ]; then
      paper_args+=(--max-eval-samples "$E1_MAX_EVAL_SAMPLES")
    fi
    if [ "$DEVICE" != "cpu" ]; then
      paper_args+=(--amp --require-cuda)
    fi
    python "${paper_args[@]}"
    echo "[DONE] Saved complete real paper bundle under $OUTPUT_DIR"
    exit 0
  fi

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

  if [ "$RUN_E1" = "true" ]; then
    echo "[4/6] Run real E1 attack/baseline experiments"
    e1_args=(
      -m fedqtrust run-e1-real
      --download-data
      --device "$DEVICE"
      --output-dir "$OUTPUT_DIR/e1_attack_resilience"
      --rounds "$E1_ROUNDS"
      --seeds "$E1_SEEDS"
      --datasets "$E1_DATASETS"
      --methods "$E1_METHODS"
      --attacks "$E1_ATTACKS"
      --clients "$E1_CLIENTS"
      --clients-per-round "$E1_CLIENTS_PER_ROUND"
      --local-epochs "$E1_LOCAL_EPOCHS"
      --batch-size "$BATCH_SIZE"
      --learning-rate "$LEARNING_RATE"
      --weight-decay "$WEIGHT_DECAY"
      --num-workers "$NUM_WORKERS"
    )
    if [ -n "$E1_MAX_TRAIN_SAMPLES" ]; then
      e1_args+=(--max-train-samples "$E1_MAX_TRAIN_SAMPLES")
    fi
    if [ -n "$E1_MAX_EVAL_SAMPLES" ]; then
      e1_args+=(--max-eval-samples "$E1_MAX_EVAL_SAMPLES")
    fi
    if [ "$DEVICE" != "cpu" ]; then
      e1_args+=(--amp --require-cuda)
    fi
    python "${e1_args[@]}"
    READ_STEP="[5/6]"
    AUDIT_STEP="[6/6]"
  else
    READ_STEP="[4/5]"
    AUDIT_STEP="[5/5]"
  fi

  echo "$READ_STEP Read real metrics and generate plots, tables, statistics, and summaries"
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

  echo "$AUDIT_STEP Audit real-result completeness"
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
