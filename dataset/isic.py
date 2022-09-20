import numpy as np
from PIL import Image
import os
import time
import pickle
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchvision.io import read_image
from torchvision.datasets import ImageFolder


from dataset.utils import * # x_u_split, TransformFixMatch, mean_std_comp, create_pickle, contains_pickle
from qfan.utils.utils import get_augmentor, safe_load_image
from dataset.custom_dataset import  CUSTOMSSL
from config import dataset_path, ssd_path


def get_isic(args, root, augment):
    #args = get_dataset_params(args, root)

    #args.classes = base_dataset.classes

    if args.augment == "none":
        transform_labeled = transforms.Compose([
            transforms.Resize(size=args.img_size[0]),
            transforms.CenterCrop(size=args.img_size[:2]),
            CustomToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std),
            Make3D()])
    elif args.augment in ["weak", "strong"]:
        transform_labeled = transforms.Compose([transforms.Resize(size=args.img_size[0]),
                                                transforms.CenterCrop(size=args.img_size[:2]),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomCrop(size=args.cropsize,  # 32,  #changed this for chestx
                                                                    padding=int(args.cropsize * 0.125),  # int(32 * 0.125),
                                                                    padding_mode='reflect')]
                                               + ([augment(n=2, m=10, size=args.cropsize)] if (args.augment == "strong"
                                                                                               and args.nofixmatch) else [])
                                               + [CustomToTensor(),
                                                  transforms.Normalize(mean=args.mean, std=args.std)])
    else:
        raise RuntimeError(f"Invalid value for augment: {args.augment}")

    transform_unlabeled = transforms.Compose([transforms.Resize(size=args.img_size[0]),
                                              transforms.CenterCrop(size=args.img_size[:2]),
                                              TransformFixMatch(mean=args.mean, std=args.std,
                                                                augment=augment, cropsize=args.cropsize,
                                                                mode=args.augment)])

    transform_val = transforms.Compose([
        transforms.Resize(size=args.img_size[0]),
        transforms.CenterCrop(size=args.img_size[:2]),
        CustomToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
        Make3D()])

    base_dataset = ImageFolder(os.path.join(args.img_dir, "train"), transform=None, loader=safe_load_image)
    args.classes = base_dataset.classes

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = ISICSSL(args, train_labeled_idxs, train=True, transform=transform_labeled,
                                      images=base_dataset.imgs, targets=base_dataset.targets,
                                      classes=base_dataset.classes,
                                    path=os.path.join(args.img_dir, "train"), loader=safe_load_image)


    train_unlabeled_dataset = ISICSSL(args, train_unlabeled_idxs, train=True,
                                        transform=transform_unlabeled, images=base_dataset.imgs,
                                        targets=base_dataset.targets, classes=base_dataset.classes,
                                        path=os.path.join(args.img_dir, "train"), loader=safe_load_image)

    test_dataset = ImageFolder(os.path.join(args.img_dir, "val"), transform=transform_val, loader=safe_load_image)

    print(f"Labeled train set contains {len(train_labeled_dataset.data)} images.")
    print(f"Unlabeled train set contains {len(train_unlabeled_dataset.data)} images.")
    print(f"test set contains {len(test_dataset.imgs)} images.")

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

class ISICSSL(ImageFolder):
    def __init__(self, args, indexs, train=True, transform=None, target_transform=None,
                 download=False, images=None, targets=None, classes=None, path="", loader=None):
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
        self.preload_data = args.preload_data
        #self.transform = transform
        #self.target_transform = target_transform
        #self.classes = classes
        super().__init__(path, transform=transform, loader=loader, target_transform=target_transform)
        if True:# indexs is not None:
            if self.preload_data:
                self.data = self.imgs[indexs]
            else:
                new_data = []
                for idx in indexs:
                    new_data.append(self.imgs[idx])
                self.data = new_data

            self.targets = np.array(self.targets)[indexs]
            self.imgs=None

    def __getitem__(self, index):
        if self.preload_data:
            img, target = self.data[index], self.targets[index]
        else:
            #img = crop_image(np.array(Image.open(self.data[index])), self.args.img_size)  # image should now have 3 channels (also if it's grayscale)
            img = np.array(self.loader(self.data[index][0]))
            target = self.targets[index]

        img = Image.fromarray(img) #if img.shape[2] > 1 else Image.fromarray(img[:, :, 0])

        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                print("transform failed")
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target







    #labeled_dataset = ImageFolder(os.path.join(args.img_dir, "/train"), transform=transform_labeled, loader=safe_load_image)
    #unlabeled_dataset = ImageFolder(os.path.join(args.img_dir, "/train"), transform=transform_unlabeled, loader=safe_load_image)










"""
class Isic(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=None,
                 store_all_data=False, images=None, targets=None, classes=None):
        if (images is not None) and (targets is not None):
            print("skipped additional unpacking of pkl files")
            total_images = images
            total_targets = targets
            self.classes = classes
        else:
            img_dir = root + '/isic/ISIC-images'
            pkl_dir = root + '/isic/'
            csv_dir = root + '/isic/ISIC-images/metadata.csv'
            if download:
                raise RuntimeError("Downloading ISIC is not coded... Sorry!")
            elif not contains_pickle(pkl_dir):
                print("Pickle file does not exist yet. Creating...")
                total_images, total_targets, self.classes = create_pickle(img_dir, pkl_dir, classwise=True,
                                                                          labels_from=csv_dir,
                                                                          img_size=(512, 512))
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
            # drop classes of the instance nan:
            if "nan" in self.classes:
                self.data = self.data[self.targets != self.classes.index("nan")]
                self.targets = self.targets[self.targets != self.classes.index("nan")]


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


def get_isic(args, root, augment, augment_labeled = False):
    base_dataset = Isic(root, train=True, store_all_data=True)

    args.classes = base_dataset.classes

    transform_labeled = transforms.Compose([
           transforms.RandomHorizontalFlip(),
           transforms.RandomCrop(size=32,
                                 padding=int(32 * 0.125),
                                 padding_mode='reflect')]
       + ([augment(n=2, m=10)] if augment_labeled else [])
       + [transforms.ToTensor(),
          transforms.Normalize(mean=base_dataset.mean, std=base_dataset.std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=base_dataset.mean, std=base_dataset.std)])

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = ISICSSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled,
        images=base_dataset.total_images, targets=base_dataset.total_targets, classes=base_dataset.classes)


    train_unlabeled_dataset = ISICSSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=base_dataset.mean, std=base_dataset.std,
                                    augment=augment),
        images=base_dataset.total_images, targets=base_dataset.total_targets, classes=base_dataset.classes)


    test_dataset = Isic(root, train=False, transform=transform_val,
                        images=base_dataset.total_images, targets=base_dataset.total_targets,
                        classes=base_dataset.classes)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class ISICSSL(Isic):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, images=None, targets=None, classes=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download,
                         images=images,
                         targets=targets,
                         classes=classes)
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

"""