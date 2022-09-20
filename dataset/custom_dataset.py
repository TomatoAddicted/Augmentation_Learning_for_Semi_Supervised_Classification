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
from config import ssd_path, machine

from dataset.utils import * # x_u_split, TransformFixMatch, mean_std_comp, create_pickle, contains_pickle


class CustomDataset(Dataset):
    def __init__(self, args, train=True, transform=None, target_transform=None, download=None,
                 store_all_data=False, images=None, targets=None, classes=None):
        self.preload_data = args.preload_data
        self.args = args

        if (images is not None) and (targets is not None):
            print("skipped additional unpacking of pkl files")
            total_images = images
            total_targets = targets
            self.classes = classes
        else:
            if self.preload_data:
                if download:
                    raise RuntimeError("Downloading custom dataset is not implemented... Sorry!")
                elif not (contains_pickle(args.pkl_dir) or os.path.isfile(args.pkl_dir)):
                    print("Pickle file does not exist yet. Creating...")
                    total_images, total_targets, self.classes = create_pickle(args)
                else:

                    start = time.time()
                    if args.classwise_pkl:
                        print("Pickle files found! Unpacking...")
                        total_images, total_targets, self.classes = open_classwise_pickles(args)
                    else:
                        print("Pickle file found! Unpacking...")
                        with open(args.pkl_dir, 'rb') as f:
                            data = pickle.load(f)
                            total_images, total_targets, self.classes = data['total_images'], data['total_targets'], data["classes"]

                    end = time.time()
                    print(f"unpacking took {end - start} seconds")



                total_images = total_images.astype(np.uint8)
                #total_targets = torch.Tensor(total_targets).type(torch.LongTensor)

                self.mean, self.std = mean_std_comp(total_images)
                print(f"Mean: {self.mean}, Standard deviation: {self.std}")

            else:
                print("Images are loaded during training. Loading filenames and labels...")
                total_images, total_targets, self.classes = collect_filenames_and_targets(args, args.img_dir)


                #total_targets = total_targets.type(torch.LongTensor)
                self.mean, self.std = args.mean, args.std


        if train:
            self.data, _, self.targets, _ = train_test_split(total_images, total_targets, test_size=0.2,
                                                             stratify=total_targets, random_state=666)
        else:
            _, self.data, _, self.targets = train_test_split(total_images, total_targets, test_size=0.2,
                                                             stratify=total_targets, random_state=666)
            # drop classes of the instance nan (only for testset):
            if "nan" in self.classes:
                mask = self.targets != self.classes.index("nan")
                if not args.preload_data:
                    idxs = np.where(mask)[0]
                    new_data = []
                    for i in range(len(idxs)):
                        new_data.append(self.data[i])
                    self.data = new_data
                else:
                    self.data = self.data[mask]
                self.targets = self.targets[self.targets != self.classes.index("nan")]


        self.transform = transform
        self.target_transform = target_transform

        if store_all_data:
            self.total_images = total_images
            self.total_targets = total_targets
        """
        new_data = np.zeros((len(self.data), *args.img_size))
        for i, img in enumerate(tqdm(self.data)):
            new_data[i] = np.array(Image.open(self.data[i]))
        self.data = new_data"""

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if self.preload_data:
            img, target = self.data[index], self.targets[index]
        else:
            img = np.array(Image.open(self.data[index]))  # image should now have 3 channels (also if it's grayscale)
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



# This function is replaced by dataset.datasets.get_dataloaders
def get_custom_dataset(args, root, augmentor):
    #args = get_dataset_params(args, root)
    base_dataset = CustomDataset(args, train=True, store_all_data=True)

    args.classes = base_dataset.classes

    if not args.nofixmatch:
        # for FixMatch use weak augment for labeled iamges
        transform_labeled = augmentor.weak
    else:
        # for Supervised training use chosen augmentation for labeled data
        transform_labeled = augmentor.strong


    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=base_dataset.mean, std=base_dataset.std),
        Make3D()])

    #transform_val_aug = transform_labeled

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CUSTOMSSL(args, train_labeled_idxs, train=True, transform=transform_labeled,
                                      images=base_dataset.total_images, targets=base_dataset.total_targets,
                                      classes=base_dataset.classes)


    train_unlabeled_dataset = CUSTOMSSL(args, train_unlabeled_idxs, train=True, transform=augmentor,
                                        images=base_dataset.total_images, targets=base_dataset.total_targets,
                                        classes=base_dataset.classes)


    test_dataset = CustomDataset(args, train=False, transform=transform_val,
                                 images=base_dataset.total_images, targets=base_dataset.total_targets,
                                 classes=base_dataset.classes)

    #test_dataset_aug = CustomDataset(args, train=False, transform=transform_val_aug,
    #                             images=base_dataset.total_images, targets=base_dataset.total_targets,
    #                             classes=base_dataset.classes)

    datasets = {"labeled": train_labeled_dataset, "unlabeled": train_unlabeled_dataset, "test": test_dataset}
    return datasets


class CUSTOMSSL(CustomDataset):
    def __init__(self, args, indexs, train=True, transform=None, target_transform=None,
                 download=False, images=None, targets=None, classes=None):
        super().__init__(args,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download,
                         images=images,
                         targets=targets,
                         classes=classes)
        self.sum1 = 0
        self.count1 = 0
        self.sum2 = 0
        self.count2 = 0

        if True:# indexs is not None:
            if self.preload_data:
                self.data = self.data[indexs]
            else:
                new_data = []
                for idx in indexs:
                    new_data.append(self.data[idx])
                self.data = new_data

            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        if self.preload_data:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
        else:
            #img = crop_image(np.array(Image.open(self.data[index])), self.args.img_size)  # image should now have 3 channels (also if it's grayscale)
            img = Image.open(self.data[index])
            target = self.targets[index]

        #img = Image.fromarray(img) #if img.shape[2] > 1 else Image.fromarray(img[:, :, 0])

        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                print("transform failed")
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target




