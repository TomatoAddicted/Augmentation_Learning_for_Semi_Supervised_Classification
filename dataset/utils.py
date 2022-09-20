import math

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import transforms
import torch
import glob
import os
from PIL import Image, ImageFilter
import pickle
from tqdm import tqdm
import cv2 as cv
import torch.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from dada.search_relax.operation import SingleAugment
from .randaugment import RandAugmentMC
from .load_augment import LoadAugment

"""
class BaseDataset(Dataset):
    def __init__(self):
        pass

    def init_dataset(self, total_images, label_dict, train, transform, target_transform,
                     store_all_data):

        classes = list(label_dict.keys())

        total_targets = torch.zeros(len(total_images))

        for i in range(len(classes)):
            for index in label_dict[classes[i]]:
                total_targets[index] = i

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
        return image, label"""


def x_u_split(args, labels):
    label_per_class_max = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    min_label = label_per_class_max
    max_label = 0
    if "nan" in args.classes:
        num_iter = args.num_classes + 1  # if data contains nan labels, there is one label more than classes -> one mo
    else:
        num_iter = args.num_classes
    for i in range(num_iter):
        idx = np.where(labels == i)[0]
        if "nan" in args.classes and i == args.classes.index(
                "nan"):  # skip nan class. nan instances are only used for unlabeled data
            print(f"excluded {len(idx)} nan instances from labeled training set.")
            continue
        label_per_class = min(label_per_class_max, len(idx))  # relevant if classes have different amounts of images
        print(f"class {args.classes[i]} has {label_per_class} labels")
        assert label_per_class >= 0, f"Class {args.classes[i]} has no labels: n = {label_per_class}"

        min_label = min(min_label, label_per_class)
        max_label = max(max_label, label_per_class)

        args.num_labeled -= label_per_class_max - label_per_class  # update num_labeled if some images were missing
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    print(f"loaded a total of {len(labeled_idx)} images. Should be: {args.num_labeled}")
    # assert len(labeled_idx) == args.num_labeled, f"{len(labeled_idx)} /= {args.num_labeled}"

    print(f"Data splitted, labeled images per class range from {min_label} to {max_label}")

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class Make3D(object):
    """Adds channel dimensions if it does not exist (required to make grayscale images 3D."""

    def __init__(self):
        pass

    def __call__(self, tensor):
        if len(tensor.shape) == 2:
            #print("orig", tensor)
            #print("1", np.tile(tensor, (1, 1, 3)))
            return tensor[None, :, :]
        else:
            return tensor

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:

        if len(tensor.shape) == 2:
            #print("orig", tensor)
            #print("1", np.tile(tensor, (1, 1, 3)))
            # new_tensor = np.zeros((3, *tensor.shape))
            # return new_tensor
            return tensor[None, :, :]
        else:
            return tensor

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort

class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img
        #print("PAWS")
        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class ParallelTransform(object):
    # used for augmentation 'learn1', applies two augmentations on the same image (one is to be selected later)
    def __init__(self, transform1, transform2):
        self.t1 = transform1
        self.t2 = transform2

    def __call__(self, x):
        return torch.stack((self.t1(x), self.t2(x)))


class TransformFixMatch(object):
    def __init__(self, args): #mean, std, cropsize, mode="strong", scale=(0.5, 1.), policy_path=''):
        self.pre = transforms.Compose([
            transforms.RandomResizedCrop(size=args.cropsize, scale=args.scale),
            transforms.RandomHorizontalFlip(),
        ])

        self.after = transforms.Compose([
            transforms.ToTensor(), #CustomToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std),
        ])
        self.test = transforms.Compose([
            transforms.Resize(size=args.cropsize),
            transforms.CenterCrop(size=(args.cropsize, args.cropsize)),
            self.after
        ])

        self.weak = transforms.Compose([
            self.pre,
            self.after
        ])
        self.fixmatch_strong = transforms.Compose([
            self.pre,
            RandAugmentMC(n=args.n_aug, m=10, size=args.cropsize),
            self.after
        ])
        self.paws = transforms.Compose([
             self.pre,
             get_color_distortion(s=1),
             GaussianBlur(p=0.5),
             self.after
        ])
        if args.augment == "strong":
            self.strong = self.fixmatch_strong
        elif args.augment == "paws":
            self.strong = self.paws
        elif args.augment == "learn":
            self.strong = ParallelTransform(self.fixmatch_strong, self.paws)
        elif args.augment == "dada":
            # dada has no augmentaion here, as augment is managed by AugmentDataset
            self.strong = transforms.Compose([])
        elif args.augment == "load":
            # use this mode to load a learned augmentation (e.g. by dada)
            self.strong = transforms.Compose([self.pre,
                                              LoadAugment(args.policy_path,
                                                          n=args.n_aug,
                                                          k=args.k_aug,
                                                          m_mode=args.m_mode,
                                                          weight_policies=args.weight_policies,
                                                          sharpen=args.sharpen_weights,
                                                          T=args.T_weights),
                                              self.after
                                              ])
        elif args.augment == "randload": # Loads random policies (baseline for load)
            self.strong = transforms.Compose([self.pre,
                                              LoadAugment(args.policy_path,
                                                          n=args.n_aug,
                                                          k=args.k_aug,
                                                          rand=True,
                                                          m_mode=args.m_mode),
                                              self.after
                                              ])
        elif args.augment == "weak":
            self.strong = self.weak

            #list(fixmatch_augment_dict().keys()):  # apply single augmentation
        elif args.augment in ['AutoContrast', 'Brightness', 'Color', 'Contrast', 'Cutout',
                          'Equalize', 'Invert', 'Posterize', 'Rotate', 'Sharpness', 'ShearX',
                          'ShearY', 'Solarize', 'SolarizeAdd', 'TranslateX', 'TranslateY', 'Black']:
            self.strong = transforms.Compose([self.pre,
                                              SingleAugment(args.augment),
                                              self.after
                                              ])

        elif args.augment == "none":
            # This mode makes no sense and is only implemented to test run times
            self.weak = transforms.ToTensor()
            self.strong = transforms.ToTensor()

        elif args.augment == "mean_std":
            self.weak = transforms.Compose([transforms.RandomResizedCrop(size=args.cropsize, scale=(1., 1.)),
                                            transforms.ToTensor()])
            self.strong = transforms.Compose([transforms.RandomResizedCrop(size=args.cropsize, scale=(1., 1.)),
                                              transforms.ToTensor()])



    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        #try:
        return weak, strong
        #return self.normalize(weak), self.normalize(strong)
        #except:
        #    print("x: ", np.array(x).shape)
        #    print("weak: ", np.array(weak).shape)
        #    print("strong: ", np.array(strong).shape)
        #    return self.normalize(weak), self.normalize(strong)



def create_pickle(args):
    image_types = ['JPG', 'jpg', 'JPEG', 'jpeg', 'png', 'PNG']
    if args.csv_dir is None:
        return create_pickle_label_by_dir(args, image_types=image_types)
    if isinstance(args.csv_dir, str) and args.csv_dir[-4:] == ".csv":
        return create_pickle_label_by_csv(args, image_types=image_types)
    else:
        raise RuntimeError("Invalid source of labels: " + args.csv_dir + "should be .csv or None (for classes by dir)")


def create_pickle_label_by_dir(args, image_types=['JPG', 'jpg', 'JPEG', 'jpeg', 'png', 'PNG']):
    """
    reads data from img dir and creates a pkl file at pkl_dir
    images are expected to be stored in separate subdirectories for each class
    param: img_dir: mother directory of all class directories
    param: pkl_dir: directory of the pkl file to be created. If classwise=False, path should contain name of
    .pkl file. If classwise=True path should be a diretory.
    param: img_size: set None to keep original image size
    param: classswise: set False to store all data in to one pkl file, set true to create one for each class
    """
    image_types_jpg = ['JPG', 'jpg', 'JPEG', 'jpeg']
    image_types_png = ['png', 'PNG']

    if args.classwise_pkl:
        assert args.pkl_dir[-1] == "/", "When pickling classwise pkl dir should be a directory: " + args.pkl_dir
    else:
        assert args.pkl_dir[
               -4:] == ".pkl", f"pkl_dir should be the path to the pkl file to be created, when creating a " \
                               f"single pkl file: {args.pkl_dir}. To create multiple pkl files set classwise_pkl=True"
    path_len = len(args.img_dir)
    num_images = 0
    labels = []
    num_classes = 0
    num_images_class = []
    for subdir, dirs, files in os.walk(args.img_dir):
        if len(files) == 0:
            continue
        num_images += len(files)
        num_images_class.append(len(files))
        labels.append(subdir[path_len:])
        # dims_img = np.array(Image.open(subdir + '/' + files[0])).shape

        dims_img = np.array(Image.fromarray(cv.imread(os.path.join(subdir, files[0])))).shape
        num_classes += 1

    if args.img_size is not None:
        # store images with new image size instead of old one
        print(f"Scaling down images from e.g. {dims_img[0, 1]} to desired size {args.img_size}.")
        dims_img[0, 1] = args.img_size

    assert num_classes >= 1, "No files in given directory: " + args.img_dir

    dims_images = (num_images, dims_img[0], dims_img[1], dims_img[2])
    total_images = np.zeros(dims_images)
    total_targets_str = []  # actual labels (string)
    total_targets = []  # integer representation of labels
    num_loaded = 0
    i = 0
    classes = []
    for subdir, dirs, files in os.walk(args.img_dir):
        if len(files) == 0:
            continue

        print(f"loading files from directory {i + 1}/{num_classes}")

        class_label = subdir[path_len + 1:]
        """
        filelist_jpg = []
        for ftype in image_types_jpg:
            filelist_jpg = filelist_jpg + glob.glob(subdir + '/*.' + ftype)

        filelist_png = []
        for ftype in image_types_png:
            filelist_png = filelist_png + glob.glob(subdir + '/*.' + ftype)
        n_new = len(filelist_jpg) + len(filelist_png)

        print(f"reading {n_new} images. for class {class_label}. class folder has a total of \
                {num_images_class[i]} files.")
        new_images = torch.Tensor([np.array(Image.open(fname)) for fname in filelist_jpg] +
                                  [np.array(Image.open(fname))[:, :, :3] for fname in filelist_png])  # png[3]: alpha
        """
        # The following two lines are not tested yet, If they don't work, replace with commented block above
        new_images = load_images_from_dir(args, subdir, types=image_types)
        n_new = new_images.shape[0]
        if not ((hasattr(args, "min_class_labels")) and len(new_images) < args.min_class_labels):

            if args.classwise_pkl: #and not ((hasattr(args, "min_class_labels")) and len(new_images) < args.min_class_labels):
                class_data = {'target': class_label,
                              'images': new_images}
                with open(args.pkl_dir + '/' + class_label + ".pkl",
                          'wb') as f:  # If a PathNotFound error occurs here, try: open(root + '/' + dataset + '.pkl' ...
                    pickle.dump(class_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            total_images[num_loaded: num_loaded + n_new] = new_images
            # total_targets_str = total_targets_str + [class_label for j in range(len(filelist))]
            classes.append(class_label)
            total_targets = total_targets + [i for j in range(n_new)]
            num_loaded += n_new
            i += 1

    """
    # creating dictionary to store all indices for each class
    class_dict = {}
    for classx in labels:
        indices = np.where(np.array(total_targets) == classx)
        class_dict[classx] = indices

    data = {'class_dict': class_dict,
            'image_data': total_images}
    """

    if not args.classwise_pkl:
        print("WARNING: you selected non classwise pickling. Dropping classes with to few instances is not implemented")
        data = {'total_images': total_images,
                'total_targets': torch.tensor(total_targets),
                'classes': classes}
        with open(args.pkl_dir,
                  'wb') as f:  # If a PathNotFound error occurs here, try: open(root + '/' + dataset + '.pkl' ...
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_images, total_targets, classes = filter_images(args, total_images, total_targets, classes)

    return np.array(total_images), torch.Tensor(total_targets), classes


def create_pickle_label_by_csv(args, image_types=['JPG', 'jpg', 'JPEG', 'jpeg', 'png', 'PNG'], drop_nan=False):
    """
    reads data from img dir and creates a pkl file at pkl_dir
    images are expected to be stored in separate subdirectories for each class
    param: img_dir: mother directory of all class directories
    param: pkl_dir: directory of the pkl file to be created. If classwise=False, path should contain name of
    .pkl file. If classwise=True path should be a diretory.
    param: img_size: set None to keep original image size
    param: classswise: set False to store all data in to one pkl file, set true to create one for each class
    """
    total_images, ids = load_images_from_dir(args, dir=args.img_dir, save_ids=True, load_subdirs=True,
                                             types=image_types)

    d = pd.read_csv(args.csv_dir)

    classes = list(set(d[args.target_attr].tolist()))
    if "nan" in classes:
        print(f"Data contains {len(classes)} different classes including nan")
    else:
        print(f"Data contains {len(classes)} different classes, all images are labeled")
    # storing classes in a txt file for monitoring
    file = open(args.pkl_dir + 'classes.txt', 'w+')
    file.writelines('\n'.join(classes))

    # dictionary to transform labels to indices:
    labels_to_idx = {}
    for i in range(len(classes)):
        if (not isinstance(classes[i], str)) and math.isnan(classes[i]):
            classes[i] = 'nan'
        labels_to_idx[classes[i]] = i

    # make "nan" the label of the highest index (this is important for the classification later)
    if "nan" in classes:
        swap_class = classes[-1]
        classes[-1] = "nan"
        classes[labels_to_idx["nan"]] = swap_class
        labels_to_idx[swap_class] = labels_to_idx['nan']
        labels_to_idx['nan'] = len(classes) - 1

    # reading targets from .csv file
    total_targets = np.zeros(len(ids))
    for i in range(len(ids)):
        classx = d.query(args.index_attr + ' == ' + '"' + ids[i] + '"')[args.target_attr].values[0]
        if not isinstance(classx, str) and math.isnan(classx):
            classx = 'nan'
        total_targets[i] = labels_to_idx[classx]
        # print(f"image {ids[i]} has the target {classx}")

    # dropping all images of class nan:
    if drop_nan:
        drop_idx = total_targets != labels_to_idx['nan']
        # print("drop_idx: ", drop_idx[:10])
        print(f"dropping a total of {len(total_targets) - sum(drop_idx)}/{len(total_targets)} images of type 'nan'")
        total_targets = total_targets[drop_idx]
        total_images = total_images[drop_idx]
        classes = classes.remove('nan')

    if args.classwise_pkl:
        print(f"saving images in {len(classes)} .pkl files. (One for each class)")

        n_images = []
        selected_idxs = np.zeros(len(total_images), dtype=bool)
        # new_images = np.zeros((0, *args.img_size))
        new_targets = []
        new_index = 0
        new_classes = []
        for i, classx in enumerate(classes):
            class_idx = total_targets == i  # classx
            class_images = total_images[class_idx]
            if hasattr(args, "min_class_labels") and len(class_images) < args.min_class_labels:
                continue
            new_classes.append(classx)
            selected_idxs = selected_idxs + class_idx
            # new_images = torch.cat((new_images, class_images))
            new_targets = new_targets + [new_index for j in range(len(class_images))]
            new_index += 1
            print(f"pickling class {i + 1}/{len(classes)}: {classx}. Class contains {len(class_images)} images.")
            n_images.append(classx + ": " + str(len(class_images)))
            data = {'target': classx,
                    'images': class_images}
            with open(args.pkl_dir + classx + '.pkl',
                      'wb') as f:  # If a PathNotFound error occurs here, try: open(root + '/' + dataset + '.pkl' ...
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        new_images = total_images[selected_idxs]
        file2 = open(args.pkl_dir + "labels_per_class.txt", "w+")
        file2.writelines('\n'.join(n_images))

        new_images, new_targets, new_classes = filter_images(args, new_images, new_targets, new_classes)

        return np.array(new_images), torch.Tensor(new_targets), new_classes


    else:
        print("WARNING: you selected non classwise pickling. Dropping classes with to few instances is not implemented")
        data = {'total_images': total_images,
                'total_targets': total_targets,
                'classes': classes}
        with open(args.pkl_dir,
                  'wb') as f:  # If a PathNotFound error occurs here, try: open(root + '/' + dataset + '.pkl' ...
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        total_images, total_targets, classes = filter_images(args, total_images, total_targets, classes)
        return np.array(total_images), torch.Tensor(total_targets), classes


def load_images_from_dir(args, dir, types=['JPG', 'jpg', 'JPEG', 'jpeg', 'png', 'PNG'], save_ids=False,
                         load_subdirs=True, only_filenames=False, crop=True):
    """
    loads all images from a given folder.
    param: img_dir: directory to load images from
    param: types: file endings to load
    param: load_subdirs: set True to also load images in subdirectories
    param: only_filenames: set True to load a list of all filenames instead of loading images
    """
    # print(f"loading files from directory {img_dir[-img_dir[::-1].find('/'):]}")
    # load folders from directory itself
    filelist = []
    for ftype in types:
        filelist = filelist + glob.glob(dir + '/*.' + ftype)

    n_images = len(filelist)

    if not only_filenames:
        print(f"reading {n_images} images. from directory {dir[-dir[::-1].find('/'):]}. class folder has a total of \
                    {len(glob.glob(dir + '/*'))} files.")
        if crop:
            images = torch.Tensor([crop_image(np.array(Image.fromarray(cv.imread(fname))), size=args.img_size) for fname in
                                   tqdm(filelist)])  # img[:3]: cutting off alpha, if exists
        else:
            images = torch.Tensor([np.array(Image.fromarray(cv.imread(fname))) for fname in
                                   tqdm(filelist)])  # img[:3]: cutting off alpha, if exists
    else:
        images = filelist
    if save_ids:
        #print("saving image id's")
        if args.id_with_type:  # keep file ending in id and only drop path
            ids = [fname[-fname[::-1].find('/'):] for fname in filelist]  # collect all filenames with ending
        else:  # drop filename and path
            ids = [fname[-fname[::-1].find('/'): -fname[::-1].find('.') - 1] for fname in
                   filelist]  # collect all filenames without file ending (e.g. .jpg)
    # Load files from subdirectories
    for subdir, dirs, files in os.walk(dir):
        for subsubdir in dirs:
            if save_ids:
                new_images, new_ids = load_images_from_dir(args, os.path.join(subdir, subsubdir), types=types,
                                                           save_ids=save_ids, load_subdirs=load_subdirs,
                                                           only_filenames=only_filenames, crop=crop)
                ids = ids + new_ids
            else:
                new_images = load_images_from_dir(args, os.path.join(subdir, subsubdir), types=types,
                                                  save_ids=save_ids, load_subdirs=load_subdirs,
                                                  only_filenames=only_filenames, crop=crop)
            if only_filenames: # images is list of strings
                images = images + new_images
            else: # images is torch tensor of images
                images = torch.cat((images, new_images))

    if save_ids:
        return images, ids
    else:
        return images

def get_targets(args, ids):
    d = pd.read_csv(args.csv_dir)

    classes = list(set(d[args.target_attr].tolist()))
    for i in range(len(classes)):
        if (not isinstance(classes[i], str)) and math.isnan(classes[i]):
            classes[i] = 'nan'

    if "nan" in classes:
        print(f"Data contains {len(classes)} different classes including nan")
    else:
        print(f"Data contains {len(classes)} different classes, all images are labeled")
    # storing classes in a txt file for monitoring
    file = open(args.pkl_dir + 'classes.txt', 'w+')
    file.writelines('\n'.join(classes))

    # dictionary to transform labels to indices:
    labels_to_idx = {}
    for i in range(len(classes)):
        if (not isinstance(classes[i], str)) and math.isnan(classes[i]):
            classes[i] = 'nan'
        labels_to_idx[classes[i]] = i

    # make "nan" the label of the highest index (this is important for the classification later)
    if "nan" in classes:
        swap_class = classes[-1]
        classes[-1] = "nan"
        classes[labels_to_idx["nan"]] = swap_class
        labels_to_idx[swap_class] = labels_to_idx['nan']
        labels_to_idx['nan'] = len(classes) - 1

    # get dictionary to map the given classes to the desired isic classes
    if args.dataset == "isic":
        isic_class_dict, classes, labels_to_idx = get_isic_class_dict()
    # reading targets from .csv file
    total_targets = np.zeros(len(ids))
    #print("WARNING only a fraction of the data is used for testing causes")
    for i in tqdm(range(len(ids))):
        classx = d.query(args.index_attr + ' == ' + '"' + ids[i].replace(" ", "_") + '"')[args.target_attr].values[0]
        if not isinstance(classx, str) and math.isnan(classx):
            classx = 'nan'
        if args.dataset == "isic":
            try:
                classx = isic_class_dict[classx]
            except:
                print(f"Class {classx} is missing in the isic class dict")
        try:
            total_targets[i] = labels_to_idx[classx]
        except:
            total_targets[i] = labels_to_idx[classx]
        # print(f"image {ids[i]} has the target {classx}")

    """
    # dropping all images of class nan:
    if drop_nan:
        drop_idx = total_targets != labels_to_idx['nan']
        # print("drop_idx: ", drop_idx[:10])
        print(f"dropping a total of {len(total_targets) - sum(drop_idx)}/{len(total_targets)} images of type 'nan'")
        total_targets = total_targets[drop_idx]
        total_images = total_images[drop_idx]
        classes = classes.remove('nan')
    """

    return total_targets, classes

def filter_images(args, images, targets, classes):
    if args.other_class:
        print(f"collecting all classes with less than {args.min_class_labels} Samples in new class called 'other'.")
        #other_targets = []
        #other_images = torch.zeros((0, *args.img_size)) if args.preload_data else []
        other_idxs = np.array([]).astype(int)
    else:
        print(f"dropping all classes with less than {args.min_class_labels} Samples.")

    new_classes = []
    new_images = torch.zeros((0, *args.img_size)) if args.preload_data else []
    new_targets = []

    next_target = 1 if args.other_class else 0  # index to assign to next target, starting at 1 if 'other' class is inserted later
    for i, classx in enumerate(classes):
        mask = targets == i
        idxs = np.where(mask)[0]
        if len(idxs) >= args.min_class_labels:
            if "chestx" in args.dataset and "|" in classx:
                continue
            if args.preload_data:
                new_images = torch.cat((new_images, images[mask]))
            else:
                class_images = []
                for idx in idxs:
                    class_images.append(images[idx])
                new_images = new_images + class_images

            new_targets = new_targets + [next_target for j in range(len(idxs))]
            next_target += 1
            new_classes.append(classx)
        elif args.other_class:
            other_idxs = np.concatenate((other_idxs, idxs))

    if args.other_class and len(other_idxs) >= args.min_class_labels:
        classes = ["other"] + classes

        if args.preload_data:
            new_images = torch.cat((new_images, images[other_idxs]))
        else:
            other_images = []
            for idx in other_idxs:
                other_images.append(images[idx])
            new_images = new_images + other_images
        new_targets = new_targets + [0 for j in range(len(other_idxs))]  # np.concatenate((np.array(new_targets), targets[other_idxs]))
    elif args.other_class and not len(other_idxs) >= args.min_class_labels:
        print(f"There were only {len(other_idxs)} samples of classes not used. no 'other' class is formed")
        # shifting all images down by one, because space left for 'other' with index 0 is not needed now:
        classes = classes[1:]
        new_targets = np.array(new_targets) - 1

    if args.preload_data:
        new_images = np.array(new_images)

    print("classes after filtering: ", new_classes)

    return new_images, np.array(new_targets), new_classes


def get_isic_class_dict():
    """
    we want to use the following classes for isic:
    melanoma: 5598
    nevus: 27878
    basal cell carcinoma: 3396
    actinic keratosis: 869
    benign keratosis:
        pigmented benign keratosis: 1099
        seborrheic keratosis: 1464
        lichenoid keratosis: 32
    dermatofibroma: 246
    vascular lesion: 253
    squamos cell carcinoma: 656
    other:
        small classes: 69
        solar lentigo: 270
    """
    dictionary = {
        "melanoma": "melanoma",#
        "nevus": "nevus",#
        "basal cell carcinoma": "basal cell carcinoma",#
        "actinic keratosis": "actinic keratosis",#
        "pigmented benign keratosis": "benign keratosis",#
        "seborrheic keratosis": "benign keratosis",#
        "lichenoid keratosis": "benign keratosis",#
        "dermatofibroma": "dermatofibroma",#
        "vascular lesion": "vascular lesion",#
        "squamous cell carcinoma": "squamous cell carcinoma",#
        "other": "other",#
        "solar lentigo": "other",#
        "lentigo NOS": "other",#
        "cafe-au-lait macule": "other",#
        "atypical melanocytic proliferation": "other",#
        "lentigo simplex": "other",#
        "angioma": "other",
        "angiofibroma or fibrous papule": "other",#
        "scar": "other",#
        "nan": "nan",#
    }

    classes = ["melanoma", "nevus", "basal cell carcinoma", "actinic keratosis", "benign keratosis", "dermatofibroma",
               "vascular lesion", "squamous cell carcinoma", "other", "nan"]
    labels_to_idx = {}
    for i, classx in enumerate(classes):
        labels_to_idx[classx] = i
    return dictionary, classes, labels_to_idx



def collect_filenames_and_targets(args, dir):
    image_paths, ids = load_images_from_dir(args, dir, save_ids=True, load_subdirs=True, only_filenames=True)
    targets, classes = get_targets(args, ids)
    image_paths, targets, classes = filter_images(args, image_paths, targets, classes)


    if "nan" in classes:
        print(f"collected {len(targets)} Images of {len(classes)} classes including nan. ")
        args.num_classes = len(classes) - 1
    else:
        print(f"collected {len(targets)} Images of {len(classes)} classes. ")
        args.num_classes = len(classes)
    return image_paths, torch.tensor(targets), classes

def crop_image(image, size=None):
    """
    crops image to squared image, if size is given. Otherwise the original image is returned
    """
    if size == None:
        return image
    assert size[0] == size[1], f"Cropping size {size} needs to be square. (other ratios are not implemented)"

    # images with only two dimesions are casted into arrays with one channel (required by torch)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    # addiional channels are cut of (like alpha channel in png's)
    image = image[:, :, :size[2]]

    half_side_length = int(min(image.shape[0], image.shape[1]) // 2)
    mid_height = int(image.shape[0] // 2)
    mid_width = int(image.shape[1] // 2)
    square_image = image[mid_height - half_side_length: mid_height + half_side_length,
                   mid_width - half_side_length: mid_width + half_side_length]

    cropped_image = cv.resize(square_image, size[:2])
    if len(cropped_image.shape) == 2:
        cropped_image = np.expand_dims(cropped_image, axis=2)
        tripple_image = np.tile(cropped_image, (1, 1, 3)) #cropped_image.repeat((1, 1, 3))
        return tripple_image
        #return np.expand_dims(cropped_image, axis=2)
    else:
        return cropped_image



def open_classwise_pickles(args):
    """
    opens all pickle files in a given directory and merges the data
    """
    filelist = glob.glob(args.pkl_dir + '/*.' + "pkl")
    total_targets = []
    classes = []
    # for i, file in tqdm(enumerate(filelist)):
    nan_occurred = False
    delta_idx = 0
    for i in tqdm(range(len(filelist))):
        # print("unpacking:", file)
        with open(filelist[i], 'rb') as f:
            new_data = pickle.load(f)
            if hasattr(args, "min_class_labels") and len(new_data['images']) < args.min_class_labels:
                delta_idx += 1
                continue
            if new_data['target'] == "nan":
                nan_occurred = True
                delta_idx = 1
            else:
                classes.append(new_data['target'])
            if not ('total_images' in locals()):
                total_images = new_data['images']
            else:
                total_images = torch.cat((total_images, new_data['images']))
            # print(f"Added {len(new_data['images'])} images of class {new_data['target']}. Total images: {len(total_images)}")
            if new_data['target'] == "nan":  # set nan as the highest index to easily drop it for classification
                total_targets = total_targets + [len(filelist) - 1 for j in range(len(new_data['images']))]
            else:
                total_targets = total_targets + [i - delta_idx for j in range(len(new_data['images']))]
    if nan_occurred:
        classes.append("nan")  # append nan at the end of classlist to maintain the right order

    total_targets = torch.Tensor(total_targets)

    print("total images loaded: ", len(total_targets))

    return filter_images(args, torch.tensor(total_images), total_targets, classes)


def contains_pickle(pkl_dir):
    """
    checks if a pkl file exists in the given directory
    """
    filelist = glob.glob(pkl_dir + '/*.' + "pkl")
    length = len(filelist)
    if length >= 1:
        return True
    else:
        return False

class CustomToTensor(transforms.ToTensor):
    def __call__(self, pic):
        """
        Manipulated Version of ToTensor, to be able to deal with int32 (ToTensor only accepts uint8 and float)
        """
        #print("0", pic.mode)
        #if pic.mode == "I":  # pic.dtype == "int32":
        #    pic = np.array(pic).astype(float) / 200000  # pic.astype(float) / 200000  # (2**31 - 1)
        return super(CustomToTensor, self).__call__(pic)



def mean_std_comp(total_images):
    try:
        mean = np.mean(total_images, axis=(0, 1, 2))
        std = np.std(total_images, axis=(0, 1, 2))
    except:
        mean = torch.mean(total_images, axis=(0, 1, 2))
        std = torch.std(total_images, axis=(0, 1, 2))
    if total_images.dtype == "uint8":
        max_val = 255
    elif total_images.dtype == "int32" or total_images.dtype == torch.float32:
        max_val = 200000 #2**31 - 1
    else:
        print(f"Warning: max value for dtype {total_images.dtype} not implemented yet! 1 is used")
        max_val = 1
    print(f"dtype {total_images.dtype} detected. Maximum value of {max_val} is assumed.")
    norm_mean = mean / max_val
    norm_std = std / max_val

    return norm_mean, norm_std

def save_create_dir(file_path):
    #directory = os.path.dirname(file_path)
    if not os.path.exists(file_path): #(directory):
        os.makedirs(file_path)

def create_train_test_split(train_dir, val_dir, ratio=0.8):

    assert ratio > 0 and ratio < 1, "Ratio needs to be between zero and one"
    import os
    import shutil
    os.mkdir(val_dir)  # This line throws an error if dir already exists (on purpose, if split already exists)
    classes = os.listdir(train_dir)
    print(f"Found {len(classes)} classes.")
    print("Transfering files to val set classwise, DO NOT INTERUPT!!!! (plis)")
    for classx in tqdm(classes):
        class_train_path = os.path.join(train_dir, classx)
        class_val_path = os.path.join(val_dir, classx)
        os.mkdir(class_val_path)  # This line throws an error if dir already exists (on purpose)
        files = os.listdir(class_train_path)
        n = len(files)  # Total amount of images in classx
        m = int(n * (1 - ratio))  # Amount of images supposed to be val
        # Creating a mask to decide which elements are moved to val set
        mask = np.full(n, False) # init all are false
        mask[:m] = True  # set first m elements to True
        np.random.shuffle(mask) # Shuffle array to distribute True values randomly
        for i, f in enumerate(files):
            if mask[i]:
                """ # This part is only to observe images with weird shapes 
                img = PIL.Image.open(class_train_path + '/'+ f)
                img = np.array(img)
                print(img.shape)
                if img.shape[0] >= 8000:
                    plt.imshow(img)
                    plt.title(classx)
                    plt.show()
                """
                shutil.move(class_train_path + '/'+ f, class_val_path + '/'+ f)
