import argparse
import json
import random
import sys
from pathlib import Path
import os

import train_detector
from src.config import get_train_config
from src.utils import setup_device

parser = argparse.ArgumentParser("Train gooddetector")

# basic config
parser.add_argument("--dataset", type=str, default='ImageNet', help="dataset for fine-tunning/evaluation")
parser.add_argument("--num-classes", type=int, default=6, help="number of classes in dataset")
parser.add_argument("--batch-size", type=int, default=32, help="batch size")

train_steps = 4590
config = parser.parse_args()
if config.dataset == "MNIST":
    random_seed = 42
elif config.dataset == "SVHN":
    random_seed = 7
elif config.dataset == "CIFAR10":
    random_seed = 24
elif config.dataset == "TinyImageNet":
    random_seed = 533
elif config.dataset == "Boston":
    random_seed = 420
    train_steps = 172798 // config.batch_size * 3
else:
    random_seed = 335

random.seed(random_seed)
seeds = random.sample(range(1000), 8 if config.dataset == 'Boston' else 5)
commands = []
experiments_dir = os.path.join(os.getcwd(), 'experiments/save')#specify the root dir

for seed, leave_out in zip(seeds, range(8)):
    path = Path("./")
    for dir in os.listdir(experiments_dir):
        if len(dir.split("_")) != 10:
            continue
        exp_name, dataset, model_arch, _, _, _, num_classes, random_seed, _, _ = dir.split("_")
        if exp_name == 'osrclassifier' and dataset == config.dataset and config.num_classes == int(num_classes[2:]) and seed == int(random_seed[2:]):
            ckpt_path = Path(experiments_dir, dir, "checkpoints", "ckpt_epoch_current.pth")
            if not ckpt_path.is_file():
                continue
            orig_config = json.load(open(ckpt_path.parent.parent / 'config.json'))
            sys.argv = [
                sys.argv[0],
                '--exp-name', 'osrdetector',
                '--n-gpu', '1',
                # '--tensorboard',
                '--image-size', '224',  # 128, 160, 224, 384, 448
                '--batch-size', str(config.batch_size),
                '--num-workers', '2',
                '--train-steps', str(train_steps),
                '--lr', '0.01',
                '--wd', '1e-5',
                '--dataset', config.dataset,
                '--num-classes', str(config.num_classes),
                '--random-seed', str(seed),
                '--checkpoint-path', str(ckpt_path),
                '--leave-out-class', str(orig_config['leave_out_class'])
            ]
            config = get_train_config()
            # device
            device, device_ids = setup_device(config.n_gpu)
            train_detector.main(config, device, device_ids)