NUM_GPUS=8
random_number=$((RANDOM % 100 + 1200))
SIZE=XL

# Linear
accelerate launch --multi_gpu --num_processes $NUM_GPUS train.py \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-${SIZE}/2" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --output-dir="exps" \
    --exp-name="${SIZE}2-linear" \
    --batch-size=256 \
    --data-dir="dataset" \
    --cfm-schedule="linear" \
    --cfm-coeff=0.1 \

# Quadratic
accelerate launch --multi_gpu --num_processes $NUM_GPUS train.py \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-${SIZE}/2" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --output-dir="exps" \
    --exp-name="${SIZE}2-quadratic" \
    --batch-size=256 \
    --data-dir="dataset" \
    --cfm-schedule="quadratic" \
    --cfm-coeff=0.15 \

# Exponential
accelerate launch --multi_gpu --num_processes $NUM_GPUS train.py \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-${SIZE}/2" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --output-dir="exps" \
    --exp-name="${SIZE}2-exponential" \
    --batch-size=256 \
    --data-dir="dataset" \
    --cfm-schedule="exponential" \
    --cfm-coeff=0.12 \

