random_number=$((RANDOM % 100 + 1200))
NUM_GPUS=8
STEP="0100000"
MODEL_SIZE='XL'
CFG_SCALE=1.0
CLS_CFG_SCALE=1.0
GH=1.0
PATCH_SIZE=2

# Experiments to iterate over
EXPERIMENTS=(
  XL2-baseline
  XL2-cosine
  XL2-uniform
)

# Number of sampling steps to evaluate
NUM_STEPS_SET=(250 50)

for NUM_STEP in "${NUM_STEPS_SET[@]}"; do
  for EXP in "${EXPERIMENTS[@]}"; do
    SAVE_PATH="exps/${EXP}"
    
    python ./evaluations/evaluator.py \
        --ref_batch evaluations/VIRTUAL_imagenet256_labeled.npz \
        --sample_batch ${SAVE_PATH}/checkpoints/SiT-${MODEL_SIZE}-${PATCH_SIZE}-${STEP}-size-256-vae-ema-cfg-${CFG_SCALE}-seed-0-sde-${NUM_STEP}steps.npz \
        --save_path ${SAVE_PATH}/checkpoints \
        --cfg_cond 1 \
        --step ${STEP} \
        --num_steps ${NUM_STEP} \
        --cfg ${CFG_SCALE} \
        --cls_cfg ${CLS_CFG_SCALE} \
        --gh ${GH}
  done
done