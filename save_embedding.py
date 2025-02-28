import argparse
import codecs
import json
import shutil

import torch
from tqdm import tqdm

from src.config import *
from src.model import OODTransformer
import random
from torch.utils.data import DataLoader
from src.dataset import *
from torch.nn import functional as F
import sklearn.metrics as skm
from src.utils import write_json

def parse_option():
    parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')

    parser.add_argument("--exp-name", type=str, default="ft", help="experiment name")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for data loader')
    parser.add_argument('--in-dataset', default='cifar10', required=False, help='cifar10 | cifar100 | stl10 | ImageNet30')
    parser.add_argument("--in-num-classes", type=int, default=1000, help="number of classes in dataset")
    parser.add_argument('--out-dataset', default='cifar10', required=False, help='cifar10 | cifar100 | stl10 | ImageNet30')
    parser.add_argument("--out-num-classes", type=int, default=1000, help="number of classes in dataset")
    parser.add_argument("--data-dir", type=str, default='./data', help='data folder')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--image-size", type=int, default=224, help="input image size", choices=[128, 160, 224, 384, 448])

    opt = parser.parse_args()

    return opt

def run_model(model, loader, softmax=False):
    #run the resnet model
    total = 0
    out_list = []
    tgt_list = []
    cls_list = []
    logit_list = []
    for images, target in tqdm(loader):
        total += images.size(0)
        images = images.cuda()
        output, classifier = model(images,feat_cls=True)

        out_list.append(output.data.cpu())
        cls_list.append(F.softmax(classifier, dim=1).data.cpu())
        tgt_list.append(target)
        logit_list.append(classifier.data.cpu())

    return  torch.cat(out_list), torch.cat(tgt_list), torch.cat(cls_list), torch.cat(logit_list)

def euclidean_dist(x, support_mean):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = support_mean.size(0)
    d = x.size(1)
    if d != support_mean.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    support_mean = support_mean.unsqueeze(0).expand(n, m, d)

    #return torch.pow(x - support_mean, 2).sum(2)
    return ((x - support_mean)*(x-support_mean)).sum(2)

def get_distances(in_list, out_list, classes_mean):

    print('Compute euclidean distance for in and out distribution data')
    test_dists = euclidean_dist(in_list, classes_mean)
    out_dists = euclidean_dist(out_list, classes_mean)

    return test_dists, out_dists

def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc

CHECKS = {}

def main(opt, model):
    file = Path('logs/vit_loc_' + str(opt.lo_classes) + '/test_logits.json')
    if file.exists():
        print(f"Already processed for LOC {opt.lo_classes}")
        return
    ckpt = torch.load(opt.ckpt_file, map_location=torch.device("cpu"))
    # if ckpt['epoch'] != 3:
    #     return
    # global CHECKS
    # CHECKS[opt.lo_classes] = [*CHECKS.get(opt.lo_classes, []), opt.ckpt_file]
    # print(ckpt['epoch'], opt.lo_classes)
    # return
    # load networks
    #model = opt.model
    missing_keys = model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.cuda()
    model.eval() 
    print('load model: ' + opt.ckpt_file)

    classes_mean = ckpt['classes_mean']

    # load ID dataset
    print('load in target data: ', opt.in_dataset)
    if opt.in_dataset == "CUB":
        import pickle
        with open("src/cub_osr_splits.pkl", "rb") as f:
            splits = pickle.load(f)
            known_classes = splits['known_classes']
            train_dataset = eval("get{}Dataset".format(opt.in_dataset))(image_size=opt.image_size, split='train', data_path=opt.data_dir, known_classes=known_classes)
            in_dataset = eval("get{}Dataset".format(opt.in_dataset))(image_size=opt.image_size, split='in_test', data_path=opt.data_dir, known_classes=known_classes)

            unknown_classes = splits['unknown_classes'][opt.mode]
            out_dataset = eval("get{}Dataset".format(opt.in_dataset))(image_size=opt.image_size, split='in_test', data_path=opt.data_dir, known_classes=unknown_classes)
    
    else:
        random.seed(opt.random_seed)
        if opt.in_dataset == "MNIST" or opt.in_dataset == "SVHN" or opt.in_dataset == "CIFAR10":
            total_classes = 10
        elif opt.in_dataset == "CIFAR100":
            total_classes = 100
        elif opt.in_dataset == "TinyImageNet":
            total_classes = 200
        elif opt.in_dataset == "Boston":
            total_classes = 8

        known_classes = random.sample(range(0, total_classes), opt.in_num_classes)

        in_dataset = eval("get{}Dataset".format(opt.in_dataset))(image_size=opt.image_size, split='in_test', data_path=opt.data_dir, known_classes=known_classes)

        # load OOD dataset
        print('load out target data: ', opt.out_dataset)
        if opt.in_dataset == opt.out_dataset and opt.out_dataset != 'Boston':
            unknown_classes =  list(set(range(total_classes)) -  set(known_classes))
            out_dataset = eval("get{}Dataset".format(opt.in_dataset))(image_size=opt.image_size, split='out_test', data_path=opt.data_dir, known_classes=known_classes)
        else:
            random.seed(opt.random_seed)
            if opt.out_dataset == "MNIST" or opt.out_dataset == "CIFAR10":
                out_total_classes = 10
            elif opt.out_dataset == "CIFAR100":
                out_total_classes = 100
            elif opt.out_dataset == "CIFAR+10":
                out_total_classes = 20
                opt.out_dataset = 'CIFAR100'
            elif opt.out_dataset == "TinyImageNet":
                out_total_classes = 200
            elif opt.in_dataset == "Boston":
                out_total_classes = 8
            unknown_classes = random.sample(range(0, out_total_classes), opt.out_num_classes)
            out_dataset = eval("get{}Dataset".format(opt.out_dataset))(image_size=opt.image_size, split='in_test', data_path=opt.data_dir, known_classes=unknown_classes)

    test_data_len = min(len(in_dataset), len(out_dataset))
    random.seed(opt.random_seed)
    in_index = random.sample(range(len(in_dataset)), test_data_len)
    in_dataset = torch.utils.data.Subset(in_dataset, in_index)
    random.seed(opt.random_seed)
    out_index = random.sample(range(len(out_dataset)), test_data_len)
    out_dataset = torch.utils.data.Subset(out_dataset, out_index)

    in_dataloader = DataLoader(in_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    out_dataloader = DataLoader(out_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    torch.multiprocessing.set_sharing_strategy('file_system')
    print(f"Running model trained on LOC {opt.lo_classes}")
    #in_emb, in_targets, in_sfmx, in_logits = run_model(model,in_dataloader)

    out_emb, out_targets, out_sfmx, out_logits = run_model(model,out_dataloader)
    #print(in_emb.shape, out_emb.shape, in_targets.shape, out_targets.shape, classes_mean.shape)
    embs = torch.cat([out_emb, classes_mean], axis=0).numpy()
    targets = torch.cat([out_targets], axis=0).numpy()
    logits = torch.cat([out_logits], axis=0).numpy()

    out_dists = euclidean_dist(out_emb, classes_mean)
    ood_lbl = torch.argmax(out_sfmx, dim=1).cpu()
    ood_score = [dist[ood_lbl[i]].cpu().tolist() for i, dist in enumerate(out_dists)]
    #np.savez("data.npz", embs=embs, targets=targets)
    #np.savez("data.npz", embs=embs, targets=targets)
    #np.savez("data.npz", embs=embs, targets=targets)nss
    store_data = dict()
    store_data["dataset"] = opt.in_dataset
    store_data["leave_out_class"] = opt.lo_classes
    store_data["ood_score"] = ood_score
    store_data["labels"] = targets.tolist()
    store_data["logits"] = logits.tolist()
    store_data["checkpoint_path"] = opt.ckpt_file
    file.parent.mkdir(parents=True, exist_ok=True)
    with codecs.open(str(file), 'w', encoding='utf-8') as fp:
        json.dump(store_data, fp,
                  separators=(',', ':'),
                  sort_keys=True)
    
    



    

    #return {'train_acc': train_acc, 'in_acc': in_acc, 'auroc': auroc, 'known_classes': sorted(known_classes), 'unknown_classes': sorted(unknown_classes)}

    


    

def run_ood_distance(opt):
    experiments_dir = os.path.join(os.getcwd(), 'experiments/save')#specify the root dir
    experiments_dir = os.path.join(os.getcwd(), f'out/detectors/{opt.out_dataset}')#specify the root dir
    for dir in os.listdir(experiments_dir):
        # if dir == 'osrdetector_data':
        #     continue
        # exp_name, dataset, model_arch, _, _, _, num_classes, random_seed, _, _ = dir.split("_")
        # if dataset != opt.in_dataset or exp_name != 'osrdetector':
        #     continue
        # if opt.exp_name == exp_name and opt.in_dataset == dataset and opt.in_num_classes == int(num_classes[2:]):
        opt = eval("get_{}_config".format('b16'))(opt)
        model = OODTransformer(
                 image_size=(opt.image_size, opt.image_size),
                 patch_size=(opt.patch_size, opt.patch_size),
                 emb_dim=opt.emb_dim,
                 mlp_dim=opt.mlp_dim,
                 num_heads=opt.num_heads,
                 num_layers=opt.num_layers,
                 num_classes=opt.in_num_classes,
                 attn_dropout_rate=opt.attn_dropout_rate,
                 dropout_rate=opt.dropout_rate,
                 )

        ckpt_file = Path(experiments_dir, dir, "checkpoints", "ckpt_epoch_current.pth")
        ckpt_file = Path(experiments_dir, dir)
        if not ckpt_file.is_file():
            continue
        print(f"Attempting {ckpt_file}")
        #cfg = json.load(open(os.path.join(experiments_dir, dir, "config.json")))
        opt.lo_classes = int(ckpt_file.stem.split('_')[1])  # cfg['leave_out_class']
        opt.ckpt_file = str(ckpt_file)
        opt.random_seed = 42  # int(random_seed[2:])

        main(opt, model)
    # print(CHECKS)
    # for loc, ck in CHECKS.items():
    #     p = Path(f"out/detectors/{opt.out_dataset}/LOC_{loc}_ckpt.pth")
    #     p.parent.mkdir(parents=True, exist_ok=True)
    #     shutil.copy(ck[0], p)




if __name__ == '__main__':
    #parse argument
    opt = parse_option()
    run_ood_distance(opt)