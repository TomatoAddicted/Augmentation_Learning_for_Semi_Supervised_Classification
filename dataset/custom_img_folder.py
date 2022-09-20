import numpy as np
from PIL import Image
import os
import time
import pickle
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from dataset.utils import * # x_u_split, TransformFixMatch, mean_std_comp, create_pickle, contains_pickle
from qfan.utils.utils import get_augmentor, safe_load_image
from dataset.custom_dataset import CUSTOMSSL
from config import dataset_path, ssd_path

from dada.search_relax.dataset import AugmentDataset
"""
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
"""
# This function is replaced by dataset.datasets.get_dataloaders
def get_img_folder(args, root, augmentor):
    #args = get_dataset_params(args, root)

    #args.classes = base_dataset.classes
    if args.augment == "dada":
        if not args.nofixmatch:
            # for FixMatch and DADA use weak augment for labeled images
            transform_labeled = augmentor.weak
            transform_unlabeled = None  # unlabeled augment is managed by DADA
        else:
            # for Supervised training and DADA augmentation will be managed by DADA
            transform_labeled = None
            transform_unlabeled = None  # irrelevant as unlabeled data is not used

    else:
        if not args.nofixmatch:
            # for FixMatch use weak augment for labeled iamges
            transform_labeled = augmentor.weak
            transform_unlabeled = augmentor # regular FixMatch transform (Strong + weak)
        else:
            # for Supervised training use chosen augmentation for labeled data
            transform_labeled = augmentor.strong
            transform_unlabeled = None  # irrelevant as unlabeled data is not used


    transform_test = augmentor.test

    #transform_val_aug = transform_labeled

    base_dataset = ImageFolder(os.path.join(args.img_dir, "train"), transform=None, loader=safe_load_image)

    args.classes = base_dataset.classes

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    if args.augment == "dada":
        # to learn augmentation we split the train set again in train and val set
        sss = StratifiedShuffleSplit(n_splits=5, test_size=args.val_portion, random_state=0)
        sss = sss.split(list(range(len(train_labeled_idxs))), torch.Tensor(base_dataset.targets)[train_labeled_idxs])
        train_labeled_idxs_idxs, val_labeled_idxs_idxs = next(sss)  # creating indicies for the index list (meta :D)
        val_labeled_idxs = train_labeled_idxs[val_labeled_idxs_idxs]
        train_labeled_idxs = train_labeled_idxs[train_labeled_idxs_idxs]

        val_labeled_dataset = FOLDERSSL(args, val_labeled_idxs, train=True, transform=None,
                                          path=os.path.join(args.img_dir, "train"), loader=safe_load_image)
        val_labeled_dataset = AugmentDataset(val_labeled_dataset, augmentor.pre, augmentor.after, transform_test,
                                                args.sub_policies, False, args.magnitudes)


    train_labeled_dataset = FOLDERSSL(args, train_labeled_idxs, train=True, transform=transform_labeled,
                                    path=os.path.join(args.img_dir, "train"), loader=safe_load_image)


    train_unlabeled_dataset = FOLDERSSL(args, train_unlabeled_idxs, train=True,
                                      transform=transform_unlabeled,
                                      path=os.path.join(args.img_dir, "train"), loader=safe_load_image)

    # TODO: Cutout einfügen, falls aktiviert
    if args.augment == "dada" and args.nofixmatch:
        train_labeled_dataset = AugmentDataset(train_labeled_dataset, augmentor.pre, augmentor.after, transform_test,
                                               args.sub_policies, True, args.magnitudes)
    if args.augment == "dada" and not args.nofixmatch:
        train_unlabeled_dataset = AugmentDataset(train_unlabeled_dataset, augmentor.pre, augmentor.after, transform_test,
                                               args.sub_policies, True, args.magnitudes, ssl=True)


    test_dataset = ImageFolder(os.path.join(args.img_dir, "val"), transform=transform_test, loader=safe_load_image)
    #test_dataset_aug = ImageFolder(os.path.join(args.img_dir, "val"), transform=transform_val_aug, loader=safe_load_image)

    print(f"Labeled train set contains {len(train_labeled_dataset.data)} images.")
    if args.augment == "dada":
        print(f"Labeled validation set contains {len(val_labeled_dataset.data)} images.")
    print(f"Unlabeled train set contains {len(train_unlabeled_dataset.data)} images.")
    print(f"test set contains {len(test_dataset.imgs)} images.")

    # creating samplers
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    if args.balance:
        train_labeled_sampler = get_balanced_class_sampler(args, train_labeled_dataset.targets)
    else:
        train_labeled_sampler = train_sampler(train_labeled_dataset.targets)
    if args.augment == "dada":
        val_labeled_sampler = train_sampler(val_labeled_dataset.targets)

    # creating dataloaders
    if args.augment == "dada":
        val_labeled_loader = torch.utils.data.DataLoader(
            val_labeled_dataset, batch_size=args.batch_size,
            # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            sampler=val_labeled_sampler, drop_last=False,
            pin_memory=True, num_workers=args.num_workers)
        if args.nofixmatch:
            # For supervised training with DADA (DADA labeled loader, regular labeled loader):
            train_labeled_loader = torch.utils.data.DataLoader(
                train_labeled_dataset, batch_size=args.batch_size, shuffle=False,
                sampler=train_labeled_sampler, drop_last=False,
                pin_memory=True, num_workers=args.num_workers)

            train_unlabeled_loader = DataLoader(
                train_unlabeled_dataset,
                sampler=train_sampler(train_unlabeled_dataset),
                batch_size=args.batch_size * args.mu,
                num_workers=args.num_workers,
                drop_last=True)
        else:
            # For FixMatch training with DADA (regulara labeled loader DADA unlabeled loader):
            train_labeled_loader = DataLoader(
                train_labeled_dataset,
                sampler=train_labeled_sampler,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                drop_last=True)
            train_unlabeled_loader = torch.utils.data.DataLoader(
                train_unlabeled_dataset, batch_size=args.batch_size * args.mu, shuffle=False,
                sampler=train_sampler(train_unlabeled_dataset.data), drop_last=False,
                pin_memory=True, num_workers=args.num_workers)
    else:
        # Regular dataloaders for FixMatch and Supervised (without DADA):
        train_labeled_loader = DataLoader(
            train_labeled_dataset,
            sampler=train_labeled_sampler,  # train_sampler(labeled_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True)

        val_labeled_loader = None # We don't need val dataset whithout DADA

        train_unlabeled_loader = DataLoader(
            train_unlabeled_dataset,
            sampler=train_sampler(train_unlabeled_dataset),
            batch_size=args.batch_size * args.mu,
            num_workers=args.num_workers,
            drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size= args.batch_size * (args.mu + 1),  # int(args.batch_size / 2),
        num_workers=args.num_workers)

    data_loaders = {"labeled": train_labeled_loader, "unlabeled": train_unlabeled_loader,
                    "val": val_labeled_loader, "test": test_loader}
    return data_loaders

"""

class FOLDERSSL(ImageFolder):
    def __init__(self, args, indexs, train=True, transform=None, target_transform=None,
                 download=False, path="", loader=None):
        """
        super().__init__(args,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download,
                         images=images,
                         targets=targets,
                         classes=classes)
        """
        self.sum1 = 0
        self.count1 = 0
        self.sum2 = 0
        self.count2 = 0

        self.preload_data = args.preload_data

        super().__init__(path, transform=transform,
                         loader=loader,
                         target_transform=target_transform)
        if True:# indexs is not None:
            if self.preload_data:
                self.data = self.imgs[indexs]
                new_samples = []
                for idx in indexs:
                    new_samples.append(self.samples[idx])
                self.samples = new_samples
            else:
                new_data = []
                new_samples = []
                for idx in indexs:
                    new_data.append(self.imgs[idx])
                    new_samples.append(self.samples[idx])
                self.data = new_data
                self.samples = new_samples
            self.targets = np.array(self.targets)[indexs]
            self.imgs = None

    def __getitem__(self, index):
        if self.preload_data:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
        else:
            try:
                img = self.loader(self.data[index][0])#Image.open(self.data[index][0])
            except:
                img = self.loader(self.data[index][0])
            target = self.targets[index]

        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                print("transform failed", img)
                img = self.transform(img)  # only for debugging

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target







    #labeled_dataset = ImageFolder(os.path.join(args.img_dir, "/train"), transform=transform_labeled, loader=safe_load_image)
    #unlabeled_dataset = ImageFolder(os.path.join(args.img_dir, "/train"), transform=transform_unlabeled, loader=safe_load_image)

