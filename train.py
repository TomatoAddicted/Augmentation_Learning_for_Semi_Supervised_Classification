from datetime import datetime
from typing import Union

print("Version: 1.0.5.4: Tracking dada process")
import argparse
import logging
import math
import os
import sys
import random
import shutil
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm

from dataset.datasets import get_dataloaders
from dataset.load_augment import visualize_dada_process
from utils import AverageMeter, accuracy, visPlot
from dataset.utils import TransformFixMatch, save_create_dir
#from dataset.randaugment import fixmatch_augment_dict
import neptune.new as neptune
from config import dataset_path, ssd_path, get_dataset_params

from dada.search_relax.architect import Architect
from dada.search_relax.model_search import Network

from dada.search_relax.train_search_paper import print_genotype, avg_op_pos

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

best_acc = 0
best_avg_class_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def calculate_weights(targets):
    targets = torch.Tensor(targets).int()
    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
    weight = 1. / class_sample_count.double()
    samples_weight = torch.tensor([weight[t] for t in targets])
    return samples_weight


def get_balanced_class_sampler(args, targets):
    count = torch.zeros(args.num_classes)
    for i in range(args.num_classes):
        count[i] = len(np.where(np.array(targets) == i)[0])
    N = float(sum(count))
    class_weights = N / count
    sample_weights = np.zeros(len(targets))
    for i in range(args.num_classes):
        sample_weights[np.array(targets) == i] = class_weights[i]
    assert sample_weights.min() > 0, "Some samples have a weight of zero, they will never be used"
    return torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(targets))  # args.batch_size)


"""
def get_balanced_class_sampler(args, targets):
    weights = calculate_weights(targets)
    assert weights.min() > 0, "Some samples have a weight of zero, they will never be used"
    return torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))  # args.batch_size)
"""


def create_model(args, log=False):
    if args.arch == 'wideresnet':
        import models.wideresnet as models
        model = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=0,
                                        num_classes=args.num_classes,
                                        input_channels=args.image_channels,
                                        )
    elif args.arch == 'resnext':
        import models.resnext as models
        model = models.build_resnext(cardinality=args.model_cardinality,
                                     depth=args.model_depth,
                                     width=args.model_width,
                                     num_classes=args.num_classes)
    elif args.arch == 'resnet':
        # my old implementation:
        # import torchvision.models as models
        # model = eval("models.resnet" + str(args.model_depth) + "()")
        # Quanfu's classifier:
        from qfan.models import build_model
        args.input_channels = 3
        args.backbone_net = "resnet" + str(args.model_depth)
        model, _ = build_model(args)
    elif args.arch == "simple":
        from models.wideresnet import build_simple_net
        model = build_simple_net(args.num_classes, args.cropsize, log)
    if log:
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6))
    return model


class Trainer():
    def __init__(self):
        self.read_args()
        global best_acc, best_avg_class_acc  # TODO: in self speichern

        self.init_logger()
        self.init_device()
        self.init_writer()
        self.log_device()
        self.set_seed()

    def read_args(self):
        # TODO: Quanfus Parser raus und relevante args unten einfügen
        # parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')

        from qfan.opts import arg_parser
        parser = arg_parser()
        # parser.add_argument('--gpu-id', default='0', type=int,
        #                    help='id(s) for CUDA_VISIBLE_DEVICES')
        parser.add_argument('--gpu-id', default='0', type=str,
                            help='id(s) for CUDA_VISIBLE_DEVICES')
        parser.add_argument('--num-workers', type=int, default=32,  # 4,
                            help='number of workers')
        parser.add_argument('--dataset', default='cifar10', type=str,
                            choices=['cifar10', 'cifar100', 'mini_imagenet', 'imagenet', 'euro_sat', 'plant_village',
                                     'isic', 'mini_plant_village', 'mini_isic', 'chestx', 'mini_chestx', 'cem_holes',
                                     'cem_squares', 'toy', 'clipart', 'infograph', 'painting', 'quickdraw', 'real',
                                     'sketch', 'sketch10'],
                            help='dataset name')
        parser.add_argument('--num-labeled', type=int, default=4000,
                            help='number of labeled data')
        parser.add_argument("--expand-labels", action="store_true",
                            help="expand labels to fit eval steps")
        parser.add_argument('--arch', default='resnet', type=str,
                            choices=['wideresnet', 'resnext', 'resnet', 'simple'],
                            help='dataset name')
        # parser.add_argument('--total-steps', default=2 ** 20, type=int,
        #                    help='number of total steps to run')
        parser.add_argument('--num-epochs', default=100, type=int,
                            help='number of epochs to run')
        parser.add_argument('--eval-step', default=1024, type=int,
                            help='number of eval steps to run')
        parser.add_argument('--start-epoch', default=0, type=int,
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('--batch-size', default=64, type=int,
                            help='train batchsize')
        parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                            help='initial learning rate')
        parser.add_argument('--warmup', default=0, type=float,
                            help='warmup epochs (unlabeled data based)')
        parser.add_argument('--wdecay', default=5e-4, type=float,
                            help='weight decay')
        parser.add_argument('--nesterov', action='store_true', default=True,
                            help='use nesterov momentum')
        parser.add_argument('--use-ema', action='store_true', default=False,
                            help='use EMA model')
        parser.add_argument('--ema-decay', default=0.999, type=float,
                            help='EMA decay rate')
        parser.add_argument('--mu', default=7, type=int,
                            help='coefficient of unlabeled batch size')
        parser.add_argument('--lambda-u', default=1, type=float,
                            help='coefficient of unlabeled loss')
        parser.add_argument('--T', default=1, type=float,
                            help='pseudo label temperature')
        parser.add_argument('--threshold', default=0.95, type=float,
                            help='pseudo label threshold')
        parser.add_argument('--out', default='result',
                            help='directory to output the result')
        parser.add_argument('--resume', default='', type=str,
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--seed', default=666, type=int,
                            help="random seed")
        parser.add_argument("--amp", action="store_true",
                            help="use 16-bit (mixed) precision through NVIDIA apex AMP")
        parser.add_argument("--opt_level", type=str, default="O1",
                            help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                 "See details at https://nvidia.github.io/apex/amp.html")
        parser.add_argument("--local_rank", type=int, default=-1,
                            help="For distributed training: local_rank")
        parser.add_argument('--no-progress', action='store_true',
                            help="don't use progress bar")
        parser.add_argument('--plt_env', default='class-Plot', type=str,
                            help='Define environment name for neptune.ai.')
        parser.add_argument('--log_file', default='Logfile',
                            help='Define a name for the log')
        # parser.add_argument('--augment_method', default='randaugment', type=str,
        #                    choices=['randaugment', 'ctaugment'],
        #                    help='Chose augmentation mode')
        parser.add_argument('--nofixmatch', action='store_true', default=False,
                            help='Set to apply regular supervised training without fixmatch')
        # parser.add_argument('--vanilla', action='store_true', default=False,
        #                    help='Set to deactivate augmentations for labeled data')
        parser.add_argument('--balance', action='store_true', default=False,
                            help='Set to balance labeled data')
        parser.add_argument('--cropsize', default=224, type=int,
                            help='size of crops after augmentation')
        parser.add_argument('--augment', default='strong', type=str,
                            choices=['weak', 'strong', 'none', 'paws', 'learn', 'dada', 'load', 'randload']
                                    + ['AutoContrast', 'Brightness', 'Color', 'Contrast', 'Cutout',
                                       'Equalize', 'Invert', 'Posterize', 'Rotate', 'Sharpness', 'ShearX',
                                       'ShearY', 'Solarize', 'SolarizeAdd', 'TranslateX', 'TranslateY', 'Black'],
                            help='chose augmentatino used for learning (fixmatch and noe does not work)')
        parser.add_argument('--n_aug', default=2, type=int,
                            help='amount of subpolicies applied to each image')
        parser.add_argument('--k_aug', default=20, type=int,
                            help='amount of subpoliciies loaded (top k policies will be used for training)')
        parser.add_argument('--m_mode', default='uniform', type=str,
                            choices=['uniform', 'normal', 'fix'],
                            help='mode for magnitude sampling')
        parser.add_argument('--tune', action='store_true', default=False,
                            help='NOT USED!!! set only when running the hyper parameter tuning script')
        parser.add_argument('--parallel', action='store_true', default=False,
                            help='activate parallel training. gpu_id will set the index of the highest gpu used')
        parser.add_argument('--noneptune', action='store_true', default=False,
                            help='activate to disable neptune')
        parser.add_argument('--supervised-warmup', default=0, type=float,
                            help='warmup epochs (only using supervised training for first x epochs)')

        # Arguments from DADA
        parser.add_argument("--policy_file", type=str, default="None",
                            help="use only for load! path to policy file "
                                 "(if 'None' file with generic name in working dir will be used)")
        parser.add_argument('--temperature', type=float, default=0.1, help="temperature (for dada augmentation)")
        parser.add_argument('--num_policies', type=int, default=105, help="number of policies used for learning")
        parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
        parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
        # parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
        ##parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--val_portion', type=float, default=0.5,
                            help='portion of validation data, used for augmentation learning')
        parser.add_argument('--mask_arch', action='store_true', default=False,
                            help='applies threshold on confidence to unsupervised samples before fed into architect')
        parser.add_argument('--min_mask_arch', type=float, default=0,
                            help='minimum amount of samples surpassing threshold for arch step. Arch step is skipped'
                                 'if mask is too sparse')
        parser.add_argument('--stop_model', type=int, default=0, help="if != 0: stop model training after x operations "
                                                                      "(Dada continues)")
        parser.add_argument('--estimate_mean_std', action='store_true', default=False,
                            help='activate to calculate mean and std. Training will not start')
        parser.add_argument('--k_dada', default=2, type=int,
                            help='amount of operations per sub-policy')
        parser.add_argument('--dada_fix_m', action='store_true', default=False,
                            help='activate to exclude m from dada augmentation training (will be fixed to 0.5)')
        parser.add_argument('--dada_fix_p', action='store_true', default=False,
                            help='activate to exclude m from dada augmentation training (will be fixed to 1)')
        parser.add_argument('--weight_policies', action='store_true', default=False,
                            help='loaded subpolicies are sampled by weight(only relevant if augment=load)')
        parser.add_argument('--orig_weights', action='store_true', default=False,
                            help='sub-policy weights are not sharpened (only relevant if weight policies)')

        parser.add_argument('--T_weights', default=0.05, type=float,
                            help='Temperature parameter for weight sharpening od subpolicies. When T is small, '
                                 'weights are sharpened more')

        self.args = parser.parse_args()
        self.args.epochs = self.args.num_epochs
        self.args = get_dataset_params(self.args, dataset_path)
        if self.args.augment == "none" and not self.args.nofixmatch:
            # print("you selected Fixmatch with augmentation none. This makes no sense as fixmatch relies on "
            #      "augmentation. Using strong augment instead")
            raise RuntimeError(
                "you selected Fixmatch with augmentation none. This makes no sense as fixmatch relies on "
                "augmentation.")
            # self.args.augment = "strong"

        # direction to store/load augmentation file (only for dada)

        policy_filename = self.args.dataset + (
            '_supervised_augment.txt' if self.args.nofixmatch else '_fixmatch_augment.txt')
        # storing is now proceeded in afolder for dada (to save all states of augment)
        if self.args.augment == "dada":
            self.policy_folder = os.path.join('/' if self.args.out[0] == '/' else '', *self.args.out.split('/')[:-1],
                                                 "policy_files")
            # creating folder if not exists:
            save_create_dir(self.policy_folder)
            self.args.policy_path = os.path.join(self.policy_folder, policy_filename)
        else:  # for load augment
            if self.args.policy_file == "None":
                self.args.policy_path = os.path.join('/' if self.args.out[0] == '/' else '', *self.args.out.split('/')[:-1],
                                                    policy_filename)
            else:
                self.args.policy_path = self.args.policy_file


        # set mode variable (fixmatch or supervised training)
        self.mode = "supervised" if self.args.nofixmatch else "fixmatch"

        #
        if self.mode == "supervised":  # For supervised training there is no mask
            self.args.mask_arch = False

        if not self.args.mask_arch:
            self.args.min_mask_arch = 0

        if self.args.estimate_mean_std:
            # we want to calculate mean and std on the original images (not augmented or normalized)
            self.args.augment = "mean_std"
            self.args.nofixmatch = False

        if self.args.k_dada == 1 and self.args.num_policies > 15:
            print("Warnung, you set k_dada to one, so num_policies is reduced to 15")
            self.args.num_policies = 15

        if self.args.weight_policies:  # when weighting policies, all should be loaded
            self.args.k_aug = 105

        self.args.sharpen_weights = not self.args.orig_weights


    def init_neptune(self, mode):
        self.args.neptune = not self.args.noneptune
        if self.args.neptune:
            try:
                os.system('neptune sync')
            except:
                print("neptune sync failed!")

            self.run = neptune.init(project='Cryo-EM/FixMatch',  # 'zipped1/Cryo-EM',
                                    tags=[self.args.plt_env, mode, self.args.dataset, self.args.augment + '_augment',
                                          str(self.args.num_labeled) + '_labels', str(self.args.cropsize) + '_crop'],
                                    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0MTVjZjhlYS0wNjFhLTRhNzItOTNkYy04ZGQxNjZmZTg5MWIifQ==')
            # old_api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmYjYyOGM0My1hNGU5LTQwNGQtOWIwMy0yMDRlMzg2YTEwZGIifQ==')

            self.run['params'] = {'initial lr': self.args.lr, 'dataset': self.args.dataset,
                                  'num labeled': self.args.num_labeled,
                                  'architecture': self.args.arch, 'batch size': self.args.batch_size,
                                  'augment': self.args.augment,
                                  'cropsize': self.args.cropsize, 'balance': self.args.balance,
                                  'epochs': self.args.epochs,
                                  'gpu_ids': self.args.gpu_id, 'num_workers': self.args.num_workers,
                                  'parallel': self.args.parallel,
                                  'no fixmatch': self.args.nofixmatch, 'arch_lr': self.args.arch_learning_rate,
                                  'pretrained': self.args.imagenet_pretrained, 'threshold': self.args.threshold,
                                  'n_aug': self.args.n_aug, 'k_aug': self.args.k_aug, 'm_mode': self.args.m_mode,
                                  'T_weights': self.args.T_weights, "weight_policies": self.args.weight_policies,
                                  'args': self.args}
            if self.args.augment == "load":
                self.run["Augmentation weights"].upload_files(self.args.policy_path)
        else:
            print("Neptune is inactive! Results won't be uploaded")
            self.run = None

    def upload_epoch_neptune(self, test_acc, test_loss, avg_class_acc, avg_color_pos):
        self.run["metrics/train/lr"].log(self.scheduler.get_last_lr()[0])

        self.run["metrics/train/loss"].log(self.losses.avg)
        self.run["metrics/train/acc_labeled"].log(self.labeled_accuracy.avg)
        self.run["metrics/train/acc_unlabeled"].log(self.unlabeled_accuracy.avg)
        if self.args.augment == "dada":
            self.run["metrics/train/acc_unrolled"].log(self.unrolled_accuracy.avg)
            self.run["metrics/train/color_pos"].log(avg_color_pos)
            self.run["metrics/train/color_thr_ratio"].log(self.color_thr_ratios.avg)
        self.run["metrics/train/thr_ratio"].log(self.thr_ratios.avg)
        self.run['"metrics/val/loss"'].log(test_loss)
        self.run['"metrics/val/acc"'].log(test_acc)
        self.run['"metrics/val/avg_class_acc"'].log(avg_class_acc)

    def init_logger(self):
        # create log folder if not exists
        if not os.path.exists(dataset_path + '/logs/' + self.args.dataset):
            os.makedirs(dataset_path + '/logs/' + self.args.dataset)

        try:
            logfile = dataset_path + '/logs/' + self.args.dataset + '/' + self.args.log_file + '.log'

            fh = logging.FileHandler(logfile)
        except:
            logfile = dataset_path + '/logs/' + self.args.dataset + '/' + self.args.log_file + 'alternative.log'

            fh = logging.FileHandler(logfile)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    def init_device(self):
        if self.args.local_rank == -1:
            main_gpu = eval("[" + self.args.gpu_id + "]")[0]
            device = torch.device('cuda', main_gpu)  # torch.device('cuda', args.gpu_id)
            self.args.world_size = 1
            self.args.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device('cuda', self.args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            self.args.world_size = torch.distributed.get_world_size()
            self.args.n_gpu = 1

        self.args.device = device
        # args.device = torch.device("cpu")  # For debugging

    def log_device(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if self.args.local_rank in [-1, 0] else logging.WARN)

        logger.warning(
            f"Process rank: {self.args.local_rank}, "
            f"device: {self.args.device}, "
            f"n_gpu: {self.args.n_gpu}, "
            f"distributed training: {bool(self.args.local_rank != -1)}, "
            f"16-bits training: {self.args.amp}", )

        logger.info(dict(self.args._get_kwargs()))

    def set_seed(self):
        pass  # seed is currently disabled
        #if self.args.seed is not None:
        #    set_seed(self.args)

    def init_writer(self):
        if self.args.local_rank in [-1, 0]:
            os.makedirs(self.args.out, exist_ok=True)
            self.args.writer = SummaryWriter(self.args.out)

        if self.args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

    def update_writer(self, test_acc, test_loss, avg_class_acc, epoch):
        self.args.writer.add_scalar('train/1.train_loss', self.losses.avg, epoch)
        self.args.writer.add_scalar('train/2.train_loss_x', self.losses_x.avg, epoch)
        self.args.writer.add_scalar('train/3.train_loss_u', self.losses_u.avg, epoch)
        self.args.writer.add_scalar('train/4.thr_ratio', self.thr_ratios.avg, epoch)
        self.args.writer.add_scalar('train/5.mask', self.mask_probs.avg, epoch)  # 4. and 5. might be the same
        self.args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
        self.args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
        self.args.writer.add_scalar('test/3.avg_class_acc', avg_class_acc, epoch)

    def close_writer(self):
        self.args.writer.close()

    def init_model(self):
        # TODO: Network einfach self.args übergeben
        if self.args.augment == "dada":
            if self.args.k_dada == 1:
                from dada.search_relax.primitives_single import sub_policies as sub_policies_raw
            elif self.args.k_dada == 2:
                from dada.search_relax.primitives import sub_policies as sub_policies_raw
            else:
                raise RuntimeError(f"Invalid value for k_dada: {self.args.k_dada}")

            self.args.sub_policies = random.sample(sub_policies_raw, self.args.num_policies)
            self.criterion = nn.CrossEntropyLoss()
            self.criterion = self.criterion.to(self.args.device)
            if self.args.parallel:
                self.args.device_ids = eval("[" + self.args.gpu_id + "]")
            else:
                self.args.device_ids = None

            # TODO: redo arch name (wideresnet does not work yet)
            self.model = Network(args=self.args, model_name=self.args.arch + str(self.args.model_depth),
                                 num_classes=self.args.num_classes,
                                 sub_policies=self.args.sub_policies, use_cuda=True, use_parallel=self.args.parallel,
                                 temperature=self.args.temperature, criterion=self.criterion, device=self.args.device,
                                 device_ids=self.args.device_ids, create_model_funct=create_model)
            self.args.magnitudes = self.model.magnitudes.to("cpu")
            self.architect = Architect(self.model, self.args)
        else:
            self.model = create_model(self.args, log=True)
            self.architect = None

        if self.args.parallel:
            # args.device_ids = [0, 1, 2, 3][:args.gpu_id + 1]
            self.args.device_ids = eval("[" + self.args.gpu_id + "]")
            if self.args.augment == "dada":
                self.model.model = torch.nn.DataParallel(self.model.model, device_ids=self.args.device_ids)
                # model._criterion = torch.nn.DataParallel(model._criterion, device_ids=args.device_ids)
            else:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.args.device_ids)

        self.model.to(self.args.device)

        self.model.zero_grad()  # TODO: Diese Zeile war zwischen scheduler und trainigs start (Verschiebung problem?)

    def init_augmentation(self):
        self.augmentor = TransformFixMatch(self.args)

    def init_data(self):
        data_loaders = get_dataloaders(self.args, dataset_path,
                                       self.augmentor)  # DATASET_GETTERS[args.dataset](args, dataset_path, augmentor)
        self.train_labeled_loader = data_loaders["labeled"]
        self.train_unlabeled_loader = data_loaders["unlabeled"]
        self.test_loader = data_loaders["test"]
        self.val_labeled_loader = data_loaders["val"]

    def init_optimizer_and_scheduler(self):
        if self.args.nofixmatch:
            self.args.eval_step = len(self.train_labeled_loader.dataset) // self.args.batch_size
        else:
            self.args.eval_step = len(self.train_unlabeled_loader.dataset) // (self.args.batch_size * self.args.mu)

        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.args.wdecay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0},
            # {'params': [model.p], 'weight_decay': 0.0}
            # {'params': [unlabeled_trainloader.dataset.transform.strong.transforms[-1].m], 'weight_decay': 0.0}
        ]

        # args.epochs = math.ceil(args.total_steps / args.eval_step)
        self.args.total_steps = self.args.eval_step * self.args.num_epochs
        self.optimizer = optim.SGD(grouped_parameters, lr=self.args.lr,
                                   momentum=0.9, nesterov=self.args.nesterov)
        # optimizer = torch.optim.Adam(grouped_parameters, #[self.model.augment_parameters()[1]] + self.model.augment_parameters()[3:],
        #                                lr=args.lr, betas=(0.5, 0.999),
        #                                  weight_decay=args.wdecay)

        # scheduler = ExponentialLR(optimizer, gamma=0.9)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.args.warmup, self.args.total_steps)

        self.args.start_epoch = 0

        if self.args.resume:
            logger.info("==> Resuming from checkpoint..")
            assert os.path.isfile(
                self.args.resume), "Error: no checkpoint directory found!"
            self.args.out = os.path.dirname(self.args.resume)
            checkpoint = torch.load(self.args.resume, map_location='cuda:0')
            best_acc = checkpoint['best_acc']
            self.args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def log_intro_training(self):
        logger.info("***** Running training *****")
        logger.info(f"  Task = {self.args.dataset}@{self.args.num_labeled}")
        logger.info(f"  Num Epochs = {self.args.epochs}")
        logger.info(f"  Batch size per GPU = {self.args.batch_size}")
        logger.info(
            f"  Total train batch size = {self.args.batch_size * self.args.world_size}")
        logger.info(f"  Total optimization steps = {self.args.total_steps}")

    def log_epoch_accuracies(self, best_acc, best_avg_class_acc):
        logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
        logger.info('Mean top-1 acc: {:.2f}\n'.format(
            np.mean(self.test_accs[-20:])))
        logger.info('Best avg class acc: {:.2f}'.format(best_avg_class_acc))
        logger.info(
            'Mean avg class acc: {:.2f}\n'.format(sum(self.avg_class_accs[-20:]) / len(self.avg_class_accs[-20:])))

    def init_average_meters(self):
        self.test_accs = []
        self.avg_class_accs = []
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.labeled_accuracy = AverageMeter()
        self.unrolled_accuracy = AverageMeter()
        self.end = time.time()
        self.losses_u = AverageMeter()
        self.mask_probs = AverageMeter()
        self.unlabeled_accuracy = AverageMeter()
        self.thr_ratios = AverageMeter()
        self.color_thr_ratios = AverageMeter()

    def reset_average_meters(self):
        self.batch_time.reset()
        self.data_time.reset()
        self.losses.reset()
        self.losses_x.reset()
        self.losses_u.reset()
        self.mask_probs.reset()
        self.labeled_accuracy.reset()
        self.unlabeled_accuracy.reset()
        self.unrolled_accuracy.reset()
        self.thr_ratios.reset()
        self.color_thr_ratios.reset()

    def update_p_bar(self, epoch, batch_idx):
        if not self.args.no_progress:
            self.p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Acc_l: {acc_l:.2f}, Acc_u: {acc_u:.2f},"
                " Acc_v: {acc_v:.2f}, LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. "
                "Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=self.args.epochs,
                    batch=batch_idx + 1,
                    iter=self.args.eval_step,
                    acc_l=self.labeled_accuracy.avg,
                    acc_u=self.unlabeled_accuracy.avg,
                    acc_v=self.unrolled_accuracy.avg if self.args.augment == "dada" else 0.,
                    lr=self.scheduler.get_last_lr()[0],
                    data=self.data_time.avg,
                    bt=self.batch_time.avg,
                    loss=self.losses.avg,
                    loss_x=self.losses_x.avg,
                    loss_u=self.losses_u.avg,
                    mask=self.mask_probs.avg))
            self.p_bar.update()

    def init_epoch_dada(self):
        self.model.sample()
        self.train_unlabeled_loader.dataset.weights_index = self.model.ops_weights_b.to("cpu")
        self.train_unlabeled_loader.dataset.probabilities_index = self.model.probabilities_b.to("cpu")
        self.train_unlabeled_loader.dataset.magnitudes = self.model.magnitudes.to("cpu")
        self.color_applied = 'Color' in self.train_unlabeled_loader.dataset.ops_names[
            self.train_unlabeled_loader.dataset.weights_index]

    def dada_step(self, inputs_u_s, targets_u, inputs_val, targets_val, mask=None, thr_ratio=None):
        self.model.train()
        self.model.set_augmenting(True)
        inputs_val = Variable(inputs_val, requires_grad=False).to(self.args.device)
        targets_val = Variable(targets_val, requires_grad=False).to(self.args.device,
                                                                    non_blocking=True)  # .cuda(non_blocking=True)

        if self.args.mask_arch:
            # Apply mask to unsupervised targets and samples before arch step
            masked_inputs_u_s = inputs_u_s[mask == 1]
            masked_targets_u = targets_u[mask == 1]
            #print("Color applied: ", self.color_applied)
            #print("Targets", targets_u)
            #print("Masked targets", masked_targets_u)
            #print(f"Passing vectors: inputs_u: {masked_inputs_u_s.shape}")

            self.architect.step(masked_inputs_u_s.to(self.args.device), masked_targets_u,
                                inputs_val, targets_val,
                                self.scheduler.get_last_lr()[0], self.optimizer, unrolled=self.args.unrolled)
        else:
            self.architect.step(inputs_u_s.to(self.args.device), targets_u, inputs_val, targets_val,
                                self.scheduler.get_last_lr()[0], self.optimizer, unrolled=self.args.unrolled)

        self.model.sample()
        if self.mode == "fixmatch":
            self.train_unlabeled_loader.dataset.weights_index = self.model.ops_weights_b
            self.train_unlabeled_loader.dataset.probabilities_index = self.model.probabilities_b
            self.train_unlabeled_loader.dataset.magnitudes = self.model.magnitudes
        else:  # supervised:
            self.train_labeled_loader.dataset.weights_index = self.model.ops_weights_b
            self.train_labeled_loader.dataset.probabilities_index = self.model.probabilities_b
            self.train_labeled_loader.dataset.magnitudes = self.model.magnitudes

    def save_model(self, test_acc, is_best, epoch):
        model_to_save = self.model
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            'ema_state_dict': None,
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, is_best, self.args.out)

    def train(self, mode):
        if mode == "fixmatch":
            self.train_fixmatch()
        if mode == "supervised":
            self.train_supervised()

    def train_fixmatch(self):
        global best_acc, best_avg_class_acc  # TODO: wie oben global weg
        self.init_average_meters()
        labeled_iter = iter(self.train_labeled_loader)

        self.model.train()

        # TODO: wieder weg, diese Zeilen sind nur um einmal die confusion matrix zu kriegen
        if self.args.augment == "dada":
            test_model = self.model.model
        else:
            test_model = self.model
        test(self.args, self.test_loader, test_model, 0)

        for epoch in range(self.args.start_epoch, self.args.epochs):
            if not self.args.no_progress:
                self.p_bar = tqdm(range(self.args.eval_step),
                                  disable=self.args.local_rank not in [-1, 0])
            if self.args.augment == "dada":
                self.init_epoch_dada()
                val_iter = iter(self.val_labeled_loader)

            for batch_idx, unlabeled_batch in enumerate(self.train_unlabeled_loader, 0):
                (inputs_u_w, inputs_u_s), _ = unlabeled_batch

                # Apply Fixmatch (supervised and unsupervised) step
                # TODO: replace try-catch
                try:
                    inputs_x, targets_x = labeled_iter.next()
                except:
                    labeled_iter = iter(self.train_labeled_loader)
                    inputs_x, targets_x = labeled_iter.next()
                self.data_time.update(time.time() - self.end)
                batch_size = inputs_x.shape[0]  # TODO: we have that in args

                inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(self.args.device)
                targets_x = targets_x.to(self.args.device)
                logits = self.model(inputs)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits

                pseudo_label = torch.softmax(logits_u_w.detach() / self.args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.args.threshold).float()
                thr_ratio = sum(mask) / len(mask)

                # IF YOU WANT TO CHANGE LOSS CRITERION: Change it in the model as well (for DADA)
                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
                Lu = (F.cross_entropy(logits_u_s, targets_u,
                                      reduction='none') * mask).mean()
                if epoch >= self.args.supervised_warmup:
                    loss = Lx + self.args.lambda_u * Lu
                else:  # start with e.g. 15 epochs of supervised training only
                    loss = Lx
                loss.backward()

                self.losses.update(loss.item())
                self.losses_x.update(Lx.item())
                self.losses_u.update(Lu.item())
                # This line avoids a bug I searched for a week ...
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                if not (self.args.stop_model != 0 and epoch >= self.args.stop_model):
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.model.zero_grad()

                self.labeled_accuracy.update(accuracy(logits_x, targets_x)[0], len(logits_x))
                self.thr_ratios.update(thr_ratio)

                if epoch >= self.args.supervised_warmup:
                    self.unlabeled_accuracy.update(accuracy(logits_u_s, targets_u)[0])
                else:
                    self.unlabeled_accuracy.update(0.)

                # Apply DADA step (if required)
                if self.args.augment == "dada":
                    if epoch >= self.args.supervised_warmup and thr_ratio > self.args.min_mask_arch:
                        print(f"DADA step: thr_ratio={thr_ratio}, min_mask={self.args.min_mask_arch}")
                        # load val images (used for val during arch step)
                        try:
                            inputs_val, targets_val = val_iter.next()
                        except:
                            val_iter = iter(self.val_labeled_loader)
                            inputs_val, targets_val = val_iter.next()
                        # apply DADA
                        self.dada_step(inputs_u_s, targets_u, inputs_val, targets_val, mask, thr_ratio)
                        # count how many images with color augment surpassed threshold
                        if self.color_applied:
                            color_thr_ratio = sum(mask) / len(mask)
                        else:
                            color_thr_ratio = 0
                        # Track accuracy (and color pos)
                        self.unrolled_accuracy.update(self.architect.unrolled_acc[0])
                        self.color_thr_ratios.update(color_thr_ratio)
                    else:  # Track accuracy as zero if no arch step was made
                        self.unrolled_accuracy.update(0.)
                        self.color_thr_ratios.update(0.)

                self.batch_time.update(time.time() - self.end)
                self.end = time.time()
                self.mask_probs.update(mask.mean().item())

                if not self.args.no_progress:
                    self.update_p_bar(epoch, batch_idx)
                # END OF ITERATION

            # Following only happens once per epoch
            if not self.args.no_progress:
                self.p_bar.close()

            if self.args.augment == "dada":
                genotype = self.model.genotype()
                # writing the genotype (the learned sub policies) into a txt file
                print_genotype(genotype, self.args, self.run, epoch)  # uploads current augmentation file to neptune
                avg_color_pos = avg_op_pos(self.args.policy_path, 'Color', epoch)
                test_model = self.model.model
            else:
                test_model = self.model
                avg_color_pos = None

            test_loss, test_acc, avg_class_acc = test(self.args, self.test_loader, test_model, epoch)
            self.test_accs.append(test_acc)
            self.avg_class_accs.append(avg_class_acc)

            self.update_writer(test_acc, test_loss, avg_class_acc, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            best_avg_class_acc = max(avg_class_acc, best_avg_class_acc)

            self.save_model(test_acc, is_best, epoch)

            self.log_epoch_accuracies(best_acc, best_avg_class_acc)

            if self.args.neptune:
                self.upload_epoch_neptune(test_acc, test_loss, avg_class_acc, avg_color_pos)

            self.reset_average_meters()
            # End of epoch

        # Following line is only run once after training is over
        self.close_writer()

    def train_supervised(self):
        global best_acc, best_avg_class_acc  # TODO: wie oben global weg
        self.init_average_meters()

        self.model.train()

        for epoch in range(self.args.start_epoch, self.args.epochs):
            if not self.args.no_progress:
                self.p_bar = tqdm(range(self.args.eval_step),
                                  disable=self.args.local_rank not in [-1, 0])
            if self.args.augment == "dada":
                self.init_epoch_dada()
                val_iter = iter(self.val_labeled_loader)

            for batch_idx, labeled_batch in enumerate(self.train_labeled_loader, 0):
                inputs, targets = labeled_batch
                inputs = inputs.to(self.args.device)
                self.data_time.update(time.time() - self.end)
                #batch_size = inputs.shape[0]  # TODO: we have that in args

                targets = targets.to(self.args.device)
                logits = self.model(inputs)

                # IF YOU WANT TO CHANGE LOSS CRITERION: Change it in the model as well (for DADA)
                Lx = F.cross_entropy(logits, targets, reduction='mean')
                loss = Lx
                loss.backward()

                self.losses.update(loss.item())
                self.losses_x.update(Lx.item())

                # This line avoids a bug I searched for a week ...
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                if not (self.args.stop_model != 0 and epoch >= self.args.stop_model):
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.model.zero_grad()

                self.labeled_accuracy.update(accuracy(logits, targets)[0], len(logits))

                # Apply DADA step (if required)
                if self.args.augment == "dada":
                    # load val images (used for val during arch step)
                    try:
                        inputs_val, targets_val = val_iter.next()
                    except:
                        val_iter = iter(self.val_labeled_loader)
                        inputs_val, targets_val = val_iter.next()
                    # apply DADA
                    self.dada_step(inputs, targets, inputs_val, targets_val)

                    # Track accuracy (and color pos)
                    self.unrolled_accuracy.update(self.architect.unrolled_acc[0])



                self.batch_time.update(time.time() - self.end)
                self.end = time.time()

                if not self.args.no_progress:
                    self.update_p_bar(epoch, batch_idx)
                # END OF ITERATION

            # The Following Code only happens once per epoch
            if not self.args.no_progress:
                self.p_bar.close()

            if self.args.augment == "dada":
                genotype = self.model.genotype()
                # writing the genotype (the learned sub policies) into a txt file
                print_genotype(genotype, self.args, self.run)  # uploads current augmentation file to neptune
                avg_color_pos = avg_op_pos(self.args.policy_path, 'Color')
                test_model = self.model.model
            else:
                test_model = self.model
                avg_color_pos = None

            test_loss, test_acc, avg_class_acc = test(self.args, self.test_loader, test_model, epoch)
            self.test_accs.append(test_acc)
            self.avg_class_accs.append(avg_class_acc)

            self.update_writer(test_acc, test_loss, avg_class_acc, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            best_avg_class_acc = max(avg_class_acc, best_avg_class_acc)

            self.save_model(test_acc, is_best, epoch)

            self.log_epoch_accuracies(best_acc, best_avg_class_acc)

            if self.args.neptune:
                self.upload_epoch_neptune(test_acc, test_loss, avg_class_acc, avg_color_pos)

            self.reset_average_meters()
            # END OF EPOCH

        # Following line is only run once after training is over
        self.close_writer()

    def estimate_mean_std(self):
        from dataset.utils import mean_std_comp
        means = AverageMeter()
        for batch_idx, unlabeled_batch in enumerate(self.train_unlabeled_loader, 0):
            (inputs_u, _), _ = unlabeled_batch

            batch_mean = torch.mean(inputs_u, dim=(0, 2, 3))
            means.update(batch_mean, len(inputs_u))

        mean = means.avg

        mean_image = torch.cat((torch.full((1, self.args.cropsize, self.args.cropsize), mean[0]),
                                torch.full((1, self.args.cropsize, self.args.cropsize), mean[1]),
                                torch.full((1, self.args.cropsize, self.args.cropsize), mean[2])))


        stds = AverageMeter()
        for batch_idx, unlabeled_batch in enumerate(self.train_unlabeled_loader, 0):
            (inputs_u, _), _ = unlabeled_batch
            deviations = torch.absolute(inputs_u - mean_image)
            batch_std = torch.mean(deviations, dim=(0, 2, 3))
            stds.update(batch_std, len(inputs_u))

        std = stds.avg

        print(f"{self.args.dataset}: Mean = {mean}, Std = {std}")



def main():
    trainer = Trainer()  # inits trainer object (args, device, seed, neptune, logging)



    trainer.init_model()  # creates model (and architect) based on args

    trainer.init_augmentation()  # creates augmentor object (with applied augmentation)

    trainer.init_data()  # loads datasets and creates dataloaders

    trainer.init_optimizer_and_scheduler()  # includes resume option

    if trainer.args.estimate_mean_std:
        trainer.estimate_mean_std()
        return

    trainer.log_intro_training()

    mode = "supervised" if trainer.args.nofixmatch else "fixmatch"

    trainer.init_neptune(mode)  # setting up neptune to upload results

    trainer.train(mode)

    if trainer.args.augment == "dada":
        visualize_dada_process(trainer.policy_folder)


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    avg_class_acc = [AverageMeter() for i in range(args.num_classes)]
    # class_count = np.zeros(args.num_classes)
    pred_count = np.zeros(args.num_classes)
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    # if args.augment == "dada":
    #    model.set_augmenting(False)

    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(inputs)

            pred_labels = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(pred_labels)
            y_true.extend(targets.cpu().numpy())

            if torch.any(targets >= args.num_classes):
                print("numclasses:", args.num_classes)
                print("DANGER: One of the targets is too high: ", targets)

            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, min(args.num_classes, 5)))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            curr_aca = 0  # current average class accuracy
            for i in range(args.num_classes):
                if (targets == i).any():
                    # calculating classwise accuracy
                    mask = targets == i  # mask for class i in batch
                    prec1, prec5 = accuracy(outputs[mask], targets[mask], topk=(1, min(args.num_classes, 5)))
                    curr_count = len(torch.where(mask))  # amount of images of class i in batch
                    avg_class_acc[i].update(prec1, curr_count)
                    # class_count[i] += curr_count
                    # counting how often the model predicts each class
                    # pred_count[i] += accuracy(outputs, torch.full((args.batch_size,), i), topk=(1,))

                curr_aca += avg_class_acc[i].avg / args.num_classes
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}.  aca: {aca:.2f}".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        aca=curr_aca
                    ))
        if not args.no_progress:
            test_loader.close()


    # creating confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    import pandas as pd
    import matplotlib as plt

    classes = test_loader.dataset.classes
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                         columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.png')
    plt.show()


    class_count_output = [(args.classes[i], int(avg_class_acc[i].avg)) for i in range(args.num_classes)]

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    logger.info("avg class acc: {:.2f}".format(curr_aca))
    logger.info(f'Class accuracies: {class_count_output}')
    # logger.info(f'Prediction counter: {pred_count}')
    return losses.avg, top1.avg, curr_aca


if __name__ == '__main__':
    main()
