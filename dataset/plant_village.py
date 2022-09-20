import numpy as np
from PIL import Image
import time
import os
import pickle
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchvision.io import read_image

from dataset.utils import * #x_u_split, TransformFixMatch, mean_std_comp, create_pickle, contains_pickle

# These are the imagenet values so far TODO: search for actual values
plant_village_mean = [0.485, 0.456, 0.406]
plant_village_std = [0.229, 0.224, 0.225]

class PlantVillage(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=None,
                 store_all_data=False, images=None, targets=None):
        if (images is not None) and (targets is not None):
            print("skipped additional unpacking of pkl files")
            total_images = images
            total_targets = targets
        else:
            img_dir = root + '/plant_village/color'
            pkl_dir = root + '/plant_village/' #plant_village.pkl'
            if download:
                raise RuntimeError("Downloading PlantVillage is not coded... Sorry!")
            elif not contains_pickle(pkl_dir):#os.path.isfile(pkl_dir):
                print(f"Pickle files not found in '{pkl_dir}'. Creating...")
                total_images, total_targets, self.classes = create_pickle(img_dir, pkl_dir, 'JPG', classwise=True)
            else:
                print("Pickle files found! Unpacking...")
                start = time.time()
                total_images, total_targets, self.classes = open_classwise_pickles(pkl_dir)
                end = time.time()
                print(f"unpacking took {end-start} seconds")

        total_images = total_images.astype(np.uint8)
        total_targets = total_targets.type(torch.LongTensor)

        self.mean, self.std = mean_std_comp(total_images)

        if train:
            self.data, _, self.targets, _ = train_test_split(total_images, total_targets, test_size=0.2,
                                                             stratify=total_targets, random_state=666)
        else:
            _, self.data, _, self.targets = train_test_split(total_images, total_targets, test_size=0.2,
                                                             stratify=total_targets, random_state=666)

        self.transform = transform
        self.target_transform = target_transform

        if store_all_data:
            self.total_images = total_images
            self.total_targets = total_targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def get_plant_village(args, root, augment, augment_labeled=False):
    base_dataset = PlantVillage(root, train=True, store_all_data=True)

    args.classes = base_dataset.classes

    transform_labeled = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(size=32,
                             padding=int(32 * 0.125),
                             padding_mode='reflect')]
        + ([augment(n=2, m=10)] if augment_labeled else [])
        + [transforms.ToTensor(),
        transforms.Normalize(mean=base_dataset.mean, std=base_dataset.std)])
    """
    transform_labeled = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=base_dataset.mean, std=base_dataset.std)])
    """
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=base_dataset.mean, std=base_dataset.std)])

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = PLANTVILLAGESSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled,
        images=base_dataset.total_images, targets=base_dataset.total_targets)

    train_unlabeled_dataset = PLANTVILLAGESSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=base_dataset.mean, std=base_dataset.std,
                                    augment=augment),
        images=base_dataset.total_images, targets=base_dataset.total_targets)

    test_dataset = PlantVillage(root, train=False, transform=transform_val,
                                images=base_dataset.total_images, targets=base_dataset.total_targets)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class PLANTVILLAGESSL(PlantVillage):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, images=None, targets=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download,
                         images=images,
                         targets=targets)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

