random_number=$((RANDOM % 100 + 1200))
NUM_GPUS=8
STEP="0100000"
MODEL_SIZE='B'
CFG_SCALE=1.0
GH=1.0

export NCCL_P2P_DISABLE=1

# Experiments to iterate over
EXPERIMENTS=(
  b2-baseline
  b2-cosine
  b2-exponential
  b2-linear
  b2-quadratic
  b2-uniform
)

# Number of sampling steps to evaluate
NUM_STEPS_SET=(250 50)

for NUM_STEP in "${NUM_STEPS_SET[@]}"; do
  for EXP in "${EXPERIMENTS[@]}"; do
    SAVE_PATH="exps/${EXP}"
    
    python -m torch.distributed.launch --master_port=$random_number --nproc_per_node=$NUM_GPUS generate.py \
        --model SiT-${MODEL_SIZE}/2 \
        --num-fid-samples 50000 \
        --ckpt ${SAVE_PATH}/checkpoints/${STEP}.pt \
        --path-type=linear \
        --projector-embed-dims=768 \
        --per-proc-batch-size=64 \
        --mode=sde \
        --num-steps=${NUM_STEP} \
        --cfg-scale=${CFG_SCALE} \
        --guidance-high=${GH} \
        --sample-dir ${SAVE_PATH}/checkpoints
  done
done
