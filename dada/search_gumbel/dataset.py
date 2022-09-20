import torch
import torchvision
from dada.search_gumbel.operation import apply_augment, Lighting
from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import LambdaLR
from dada.search_gumbel.primitives import sub_policies
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torchvision import transforms
from PIL import Image
import os
import math
import numpy as np


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

class AugmentDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, pre_transforms, after_transforms, valid_transforms, ops_names, search, magnitudes):
        super(AugmentDataset, self).__init__()
        self.dataset = dataset
        self.pre_transforms = pre_transforms
        self.after_transforms = after_transforms
        self.valid_transforms = valid_transforms
        self.ops_names = ops_names
        self.search = search
        self.magnitudes = magnitudes

    def __getitem__(self, index):
        if self.search:
            # start_time = time.time()
            img, target = self.dataset.__getitem__(index)
            img = self.pre_transforms(img)
            magnitude = self.magnitudes.clamp(0, 1)[self.weights_index.item()]
            sub_policy = self.ops_names[self.weights_index.item()]
            probability_index = self.probabilities_index[self.weights_index.item()]
            image = img
            for i, ops_name in enumerate(sub_policy):
                if probability_index[i].item() == 1:
                    image = apply_augment(image, ops_name, magnitude[i])
            image = self.after_transforms(image)
            # print(self.magnitudes)
            # print(self.weights_index)
            # end_time = time.time()
            # print("%f" % (end_time - start_time))
            return image, target
        else:
            img, target = self.dataset.__getitem__(index)
            if self.valid_transforms is not None:
                img = self.valid_transforms(img)
            return img, target

    def __len__(self):
        return self.dataset.__len__()

_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

def num_class(dataset):
    return {
        'cifar10': 10,
        'reduced_cifar10': 10,
        'cifar10.1': 10,
        'cifar100': 100,
        'reduced_cifar100': 100,
        'svhn': 10,
        'reduced_svhn': 10,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]

def get_dataloaders(args, labeled_train_set, labeled_val_set, train_sampler, val_sampler, ops_names, magnitudes):

    transform_train_pre = transforms.Compose([
        transforms.RandomResizedCrop(size=args.cropsize, scale=args.scale),
        transforms.RandomHorizontalFlip(),
    ])
    transform_train_after = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.mean, args.std),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=args.img_size[0]),
        transforms.CenterCrop(size=args.img_size[:2]),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ])

    if args.cutout and args.cutout_length != 0:
        transform_train_after.transforms.append(CutoutDefault(args.cutout_length))


    train_data = AugmentDataset(labeled_train_set, transform_train_pre, transform_train_after, transform_test, ops_names, True, magnitudes)
    valid_data = AugmentDataset(labeled_val_set, transform_train_pre, transform_train_after, transform_test, ops_names, False, magnitudes)

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False,
        sampler=train_sampler, drop_last=False,
        pin_memory=True, num_workers=args.num_workers)

    validloader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        sampler=val_sampler, drop_last=False,
        pin_memory=True, num_workers=args.num_workers)

    return trainloader, validloader
