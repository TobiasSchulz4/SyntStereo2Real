#!/usr/bin/env bash
set -e

CONFIG_TRANSLATION="configs/config_translation_driving_kitti.yaml"
CONFIG_AANET="configs/config_aanet_training.yaml"
OUTPUT_ROOT="experiments"
CHECKPOINT_DIR="${OUTPUT_ROOT}/checkpoints/driving_to_kitti"
TRANSLATED_DIR="datasets/translated/driving_to_kitti"
EVAL_ROOT="datasets/aanet_eval/kitti"
DEVICE=${DEVICE:-cuda}

mkdir -p "${CHECKPOINT_DIR}" "${TRANSLATED_DIR}"

echo "[1/4] Training translator (Driving -> KITTI)"
python -m src.trainers.train_translation \
  --config "${CONFIG_TRANSLATION}" \
  --output_root "${CHECKPOINT_DIR}" \
  --device "${DEVICE}"

echo "[2/4] Translating synthetic dataset"
CHECKPOINT_PATH=$(ls -t "${CHECKPOINT_DIR}"/checkpoints/checkpoint_*.pt | head -n 1)
if [ -z "${CHECKPOINT_PATH}" ]; then
  echo "No checkpoint found in ${CHECKPOINT_DIR}/checkpoints" >&2
  exit 1
fi
python -m src.trainers.translate_dataset \
  --config "${CONFIG_TRANSLATION}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output_root "${TRANSLATED_DIR}" \
  --device "${DEVICE}" \
  --use_edge

echo "[3/4] Training AANet on translated dataset"
python -m src.trainers.train_aanet --config "${CONFIG_AANET}"

echo "[4/4] Evaluating AANet predictions"
python -m src.trainers.eval_aanet --root "${EVAL_ROOT}"
