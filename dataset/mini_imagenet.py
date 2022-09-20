
import numpy as np
from PIL import Image
import pickle
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


from dataset.utils import x_u_split, TransformFixMatch

# I found those on Github: https://github.com/DeepVoltaire/AutoAugment/issues/4
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
# some folks on stack overflow said its okay to use Imagenet values for similar datasets: https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
mini_imagenet_mean = imagenet_mean
mini_imagenet_std = imagenet_std



class MiniImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, store_all_data=False,
                 images=None, targets=None):
        if (images is not None) and (targets is not None):
            print("skipped additional unpacking of pkl file")
            total_images = images
            total_targets = targets
        else:
            if download:
                raise RuntimeError("Downloading miniImagenet is not coded... Sorry!")
            with open(root + '/mini-imagenet/mini-imagenet-cache-train.pkl', 'rb') as f:
                data0 = pickle.load(f)
            with open(root + '/mini-imagenet/mini-imagenet-cache-test.pkl', 'rb') as f:
                data1 = pickle.load(f)
            with open(root + '/mini-imagenet/mini-imagenet-cache-val.pkl', 'rb') as f:
                data2 = pickle.load(f)

            total_images = np.concatenate((data0['image_data'],
                                           data1['image_data'],
                                           data2['image_data']), axis=0)

            len0 = len(data0['image_data'])
            len1 = len(data1['image_data'])
            len2 = len(data2['image_data'])

            label_dict0 = data0['class_dict']
            label_dict1 = data1['class_dict']
            label_dict2 = data2['class_dict']

            classes0 = list(label_dict0.keys())
            classes1 = list(label_dict1.keys())
            classes2 = list(label_dict2.keys())

            num_classes0 = len(classes0)
            num_classes1 = len(classes1)
            num_classes2 = len(classes2)


            total_targets = torch.zeros(len(total_images))

            self.classes = classes0 + classes1 + classes2

            for i in range(len(classes0)):
                for index in label_dict0[classes0[i]]:
                    total_targets[index] = i

            for i in range(len(classes1)):
                for index in label_dict1[classes1[i]]:
                    total_targets[index + len0] = i + num_classes0

            for i in range(len(classes2)):
                for index in label_dict2[classes2[i]]:
                    total_targets[index + len0 + len1] = i + num_classes0 + num_classes1

        total_targets = total_targets.type(torch.LongTensor)
        if store_all_data:
            self.total_images = total_images
            self.total_targets = total_targets

        if train:
            self.data, _, self.targets, _ = train_test_split(total_images, total_targets, test_size=0.2, stratify=total_targets, random_state=666)
        else:
            _, self.data, _, self.targets = train_test_split(total_images, total_targets, test_size=0.2, stratify=total_targets, random_state=666)

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


def get_mini_imagenet(args, root, augment, augment_labeled=False):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect')]
        + ([augment(n=2, m=10)] if augment_labeled else [])
        + [transforms.ToTensor(),
        transforms.Normalize(mean=mini_imagenet_mean, std=mini_imagenet_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mini_imagenet_mean, std=mini_imagenet_std)])

    base_dataset = MiniImageNet(root, train=True, store_all_data=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = MINIIMAGENETSSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled,
        images=base_dataset.total_images, targets=base_dataset.total_targets)

    train_unlabeled_dataset = MINIIMAGENETSSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=imagenet_mean, std=imagenet_std, augment=augment),
        images=base_dataset.total_images, targets=base_dataset.total_targets)

    test_dataset = MiniImageNet(root, train=False, transform=transform_val,
                                images=base_dataset.total_images, targets=base_dataset.total_targets)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_imagenet(args, root, augment):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])


    # TODO: replace by handcrafted dataloader
    base_dataset = datasets.ImageNet(
        root, train=True, download=None)



    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = IMAGENETSSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = IMAGENETSSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=imagenet_mean, std=imagenet_std, augment=augment))

    test_dataset = datasets.ImageNet(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset



class MINIIMAGENETSSL(MiniImageNet):
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

class IMAGENETSSL(datasets.ImageNet):
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

