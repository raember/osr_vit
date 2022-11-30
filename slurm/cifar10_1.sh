#!/bin/bash
#SBATCH --job-name=embe/osr_vit_cifar10_1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --gres=gpu:1

WANDB_API_KEY=$(awk '/api.wandb.ai/ {f=1} f && /password/ {print $2;f=0}' ~/.netrc)

docker run --rm -it \
    --shm-size=50g \
    -v /cluster/home/embe/osr4h/osr_vit:/osr-vit \
    -v cifar-10:/osr-vit/data/cifar-10-batches-py \
    --name ${SLURM_JOB_NAME/\//_} \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e SLURM_JOB_NAME=$SLURM_JOB_NAME \
    embe/osr_vit:latest \
    python train_classifier_batch.py \
        --batch-size 128 \
        --checkpoint-path ./pretrained_model/imagenet21k+imagenet2012_ViT-B_16.pth \
        --dataset CIFAR10 \
        --num-classes 10
