#!/bin/bash
#SBATCH --job-name=embe/osr_vit_svhn_1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --gres=gpu:1

WANDB_API_KEY=$(awk '/api.wandb.ai/ {f=1} f && /password/ {print $2;f=0}' ~/.netrc)

docker run --rm \
    -v /cluster/home/embe/osr4h/osr_vit:/osr-vit \
    --name embe_osr_vit_svhn_1 \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    embe/osr_vit:latest \
    python train_classifier.py \
        --exp-name osrclassifier \
        --n-gpu 1 --batch-size 64 --num-workers 4 --train-steps 4590 --lr 0.01 --wd 1e-5 \
        --dataset SVHN \
        --random-seed 69 \
        --checkpoint-path ./pretrained_model/imagenet21k+imagenet2012_ViT-B_16.pth
