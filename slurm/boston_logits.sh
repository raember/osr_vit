#!/bin/bash
#SBATCH --job-name=embe/osr_vit_boston_logits
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=250G
#SBATCH --gres=gpu:5

#WANDB_API_KEY=$(awk '/api.wandb.ai/ {f=1} f && /password/ {print $2;f=0}' ~/.netrc)

docker run --rm -it \
    --shm-size=50g \
    -v /cluster/home/embe/osr4h/osr_vit:/osr-vit \
    -v /cluster/data/embe/boston:/osr-vit/data/boston \
    --name ${SLURM_JOB_NAME/\//_} \
    -e SLURM_JOB_NAME=$SLURM_JOB_NAME \
    embe/osr_vit:latest \
    python save_embedding.py \
        --exp-name osrdetector \
        --batch-size 64 \
        --in-dataset Boston \
        --in-num-classes 7 \
        --out-dataset Boston \
        --out-num-classes 8 \
        --num-workers 8
