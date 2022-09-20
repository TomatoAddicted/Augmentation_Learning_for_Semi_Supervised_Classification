import numpy as np
from PIL import Image
import os
import pickle
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchvision.io import read_image

from dataset.utils import x_u_split, TransformFixMatch, mean_std_comp, create_pickle


class ChestX(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=None):
        img_dir = root + '/chestx'
        pkl_dir = root + '/chestx/chestx.pkl'
        if download:
            raise RuntimeError("Downloading ISIC is not coded... Sorry!")
        elif not os.path.isfile(pkl_dir):
            print("Pickle file does not exist yet. Creating...")
            data = create_pickle(img_dir, pkl_dir, 'JPG')
        else:
            print("Pickle file found! Unpacking...")
            with open(pkl_dir, 'rb') as f:
                data = pickle.load(f)


        total_images = (data['image_data'])

        label_dict = data['class_dict']

        classes = list(label_dict.keys())

        total_targets = torch.zeros(len(total_images))

        for i in range(len(classes)):
            for index in label_dict[classes[i]]:
                total_targets[index] = i

        total_images = total_images.astype(np.uint8)
        total_targets = total_targets.type(torch.LongTensor)

        self.chestx_mean, self.chestx_std = mean_std_comp(total_images)

        if train:
            self.data, _, self.targets, _ = train_test_split(total_images, total_targets, test_size=0.2,
                                                             stratify=total_targets, random_state=666)
        else:
            _, self.data, _, self.targets = train_test_split(total_images, total_targets, test_size=0.2,
                                                             stratify=total_targets, random_state=666)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_chestx(args, root, augment):
    base_dataset = ChestX(root, train=True)

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=base_dataset.chestx_mean, std=base_dataset.chestx_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=base_dataset.chestx_mean, std=base_dataset.chestx_std)])

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CHESTXSSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CHESTXSSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=base_dataset.chestx_mean, std=base_dataset.chestx_std, augment=augment))

    test_dataset = ChestX(root, train=False, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class CHESTXSSL(ChestX):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
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

