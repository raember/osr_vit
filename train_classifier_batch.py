import argparse
import random
import sys

import train_classifier
from src.config import get_train_config
from src.utils import setup_device

parser = argparse.ArgumentParser("Train classifier")

# basic config
parser.add_argument("--checkpoint-path", type=str, default=None, help="model checkpoint to load weights")
parser.add_argument("--dataset", type=str, default='ImageNet', help="dataset for fine-tunning/evaluation")
parser.add_argument("--num-classes", type=int, default=6, help="number of classes in dataset")
parser.add_argument("--batch-size", type=int, default=32, help="batch size")
parser.add_argument("--leave-out", type=int, default=-1, help="class to leave out")

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
    train_steps = 172798 // config.batch_size * 3  # Aim for 3 epochs
else:
    random_seed = 335

random.seed(random_seed)
seeds = random.sample(range(1000), 8 if config.dataset == 'Boston' else 5)
leave_out = range(8)
if config.leave_out != -1:
    seeds = [seeds[config.leave_out]]
    leave_out = [config.leave_out]
args = [
    sys.argv[0],
    '--exp-name', 'osrclassifier',
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
    '--checkpoint-path', config.checkpoint_path,
]
for seed, leave_out in zip(seeds, leave_out):
    sys.argv = [
        *args,
        '--random-seed', str(seed),
    ]
    if config.dataset == 'Boston':
        sys.argv.append('--leave-out-class')
        sys.argv.append(str(leave_out))
    config = get_train_config()
    # device
    device, device_ids = setup_device(config.n_gpu)
    train_classifier.main(config, device, device_ids)