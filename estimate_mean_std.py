from dataset.utils import load_images_from_dir, mean_std_comp, CustomToTensor
from config import get_dataset_params
from config import dataset_path
import argparse
import matplotlib.pyplot as plt
import numpy as np

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main():
    parser = argparse.ArgumentParser(description='Precropping data for fixmatch')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'mini_imagenet', 'imagenet', 'euro_sat', 'plant_village',
                                 'isic', 'mini_plant_village', 'mini_isic', 'chestx', 'mini_chestx', 'cem_holes',
                                 'cem_squares', 'toy', 'clipart', 'infograph', 'painting', 'quickdraw', 'real',
                                 'sketch', 'sketch10'],
                        help='dataset name')
    parser.add_argument('--image-size', default='256', type=int,
                        help='height of images (width will be the same)')
    parser.add_argument('--data-type', default='regular', type=str, choices=['regular', 'folder'],
                        help='How the data is stored (e.g. Image_folder)')
    args = parser.parse_args()
    args.cropratio = 0.8
    args.arch = 'resnet'

    root = dataset_path

    args = get_dataset_params(args, root)


    if args.data_type == "regular":
        images = load_images_from_dir(args, args.img_dir, load_subdirs=True, crop=False)
    elif args.data_type == "folder":
        images = load_images_from_dir(args, args.img_dir + "/train/", load_subdirs=True, crop=True)
    print(images.shape)

    print()

    mean, std = mean_std_comp(images)

    print(f"{args.dataset}: Mean: {mean}, Standard deviation: {std}")

if __name__ == "__main__":
    main()
