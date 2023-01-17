#!/bin/bash
#SBATCH --job-name=embe/osr_vit_boston_2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --gres=gpu:1

WANDB_API_KEY=$(awk '/api.wandb.ai/ {f=1} f && /password/ {print $2;f=0}' ~/.netrc)

docker run --rm -it \
    --shm-size=50g \
    -v /cluster/home/embe/osr4h/osr_vit:/osr-vit \
    -v /cluster/data/embe/boston:/osr-vit/data/boston \
    --name ${SLURM_JOB_NAME/\//_} \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e SLURM_JOB_NAME=$SLURM_JOB_NAME \
    embe/osr_vit:latest \
    python train_detector.py \
        --exp-name osrdetector \
        --n-gpu 1 --image-size 224 --batch-size 128 --num-workers 2 --train-steps 4047 --lr 0.01 --wd 1e-5 \
        --dataset Boston \
        --num-classes 7 --random-seed 26 \
        --checkpoint-path /osr-vit/experiments/save/osrclassifier_Boston_b16_bs128_lr0.01_wd1e-05_nc7_rs26_230104_064440/checkpoints/ckpt_epoch_current.pth \
        --leave-out-class 0
