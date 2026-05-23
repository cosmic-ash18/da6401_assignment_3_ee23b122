#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-da6401-a3-report}
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-64}
DMODEL=${DMODEL:-256}
LAYERS=${LAYERS:-3}
HEADS=${HEADS:-8}
DFF=${DFF:-1024}
DEVICE=${DEVICE:-auto}

if [ "$DEVICE" = "auto" ]; then
  DEVICE_ARG=""
else
  DEVICE_ARG="--device $DEVICE"
fi

mkdir -p checkpoints
COMMON="--use_wandb --wandb_project $PROJECT --epochs $EPOCHS --batch_size $BATCH --d_model $DMODEL --N $LAYERS --num_heads $HEADS --d_ff $DFF $DEVICE_ARG"

python3 train.py $COMMON --run_name q21_noam_baseline --experiment_tag q21_noam --checkpoint checkpoints/q21_noam_best.pth --last_checkpoint checkpoints/q21_noam_last.pth --log_attention_sample --log_grad_steps 1000 --log_prediction_confidence --compute_val_bleu
python3 train.py $COMMON --run_name q21_fixed_lr --experiment_tag q21_fixed_lr --fixed_lr --lr 1e-4 --checkpoint checkpoints/q21_fixed_lr_best.pth --last_checkpoint checkpoints/q21_fixed_lr_last.pth
python3 train.py $COMMON --run_name q22_no_scale_attention --experiment_tag q22_no_scale --no_scale_attention --checkpoint checkpoints/q22_no_scale_best.pth --last_checkpoint checkpoints/q22_no_scale_last.pth --log_grad_steps 1000
python3 train.py $COMMON --run_name q24_learned_positional --experiment_tag q24_learned_pe --learned_positional --checkpoint checkpoints/q24_learned_pe_best.pth --last_checkpoint checkpoints/q24_learned_pe_last.pth --compute_val_bleu
python3 train.py $COMMON --run_name q25_no_label_smoothing --experiment_tag q25_no_smoothing --smoothing 0.0 --checkpoint checkpoints/q25_no_smoothing_best.pth --last_checkpoint checkpoints/q25_no_smoothing_last.pth --log_prediction_confidence
