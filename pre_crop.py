import argparse
from config import dataset_path, get_dataset_params
from tqdm import tqdm
import numpy as np
import torch
import os

import glob

from dataset.utils import load_images_from_dir, crop_image, save_create_dir
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description='Precropping data for fixmatch')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'mini_imagenet', 'imagenet', 'euro_sat', 'plant_village',
                                 'isic', 'mini_plant_village', 'mini_isic', 'chestx', 'mini_chestx', 'clipart',
                                'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
                        help='dataset name')
    parser.add_argument('--csv', action='store_true', default=False,
                        help='Set to also create a corresponding csv file')
    parser.add_argument('--imgfolder', action='store_true', default=False,
                        help='Set if data is stored in img folder structure. Will keep that structure for cropped images')


    args = parser.parse_args()
    args.cropsize = None #required to deny not defined error

    root = dataset_path

    args = get_dataset_params(args, root,no_model=True)

    print(f"preprocessing data from {args.orig_img_dir}. Storing Images to {args.img_dir}")

    if not args.imgfolder:
        crop_and_save_images(args, args.orig_img_dir, args, plot_bar=True)
    else:
        ### Crop train images ###
        print("Cropping Training images")
        orig_train_dir = os.path.join(args.orig_img_dir, 'train')
        cropped_train_dir = os.path.join(args.img_dir, 'train')
        # create train dir (for cropped images)
        save_create_dir(cropped_train_dir)
        # get classes from orig image dir (classes are directory names)
        classes = os.listdir(orig_train_dir)
        for classx in tqdm(classes):
            orig_classx_dir = os.path.join(orig_train_dir, classx)
            cropped_classx_dir = os.path.join(cropped_train_dir, classx)
            save_create_dir(cropped_classx_dir)
            print("created ", cropped_classx_dir)
            crop_and_save_images(args, orig_classx_dir, cropped_classx_dir, load_subdirs=False)


        ### Crop val images ###  (repeat same steps for val images)
        print("Cropping Validation images")
        orig_val_dir = os.path.join(args.orig_img_dir, 'val')
        cropped_val_dir = os.path.join(args.img_dir, 'val')
        # create train dir (for cropped images)
        save_create_dir(cropped_val_dir)
        # get classes from orig image dir (classes are directory names)
        classes = os.listdir(orig_val_dir)
        for classx in tqdm(classes):
            orig_classx_dir = os.path.join(orig_val_dir, classx)
            cropped_classx_dir = os.path.join(cropped_val_dir, classx)
            save_create_dir(cropped_classx_dir)
            crop_and_save_images(args, orig_classx_dir, cropped_classx_dir, load_subdirs=False)



    if args.csv:
        print("Creating CSV file")
        create_csv_by_dirs(args.orig_img_dir, args.csv_dir)
    """

    images = np.array(images).astype(np.uint8)

    print("storing images...")

    for i, id in enumerate(tqdm(ids)):
        im = Image.fromarray(images[i])
        im.save(args.img_dir + '/' + id + ('.png' if not args.id_with_type else ''))
    """




def crop_and_save_images(args, dir, crop_dir, types=['JPG', 'jpg', 'JPEG', 'jpeg', 'png', 'PNG'], load_subdirs=True,
                         plot_bar=False):
    """
    loads all images from a given folder, crop them and savae them in crop_dir.
    param: img_dir: directory to load images from
    param: types: file endings to load
    param: load_subdirs: set True to also load images in subdirectories
    param: only_filenames: set True to load a list of all filenames instead of loading images
    """
    # print(f"loading files from directory {img_dir[-img_dir[::-1].find('/'):]}")
    # load folders from directory itself
    print("Collecting filelist for directory: " + dir)
    filelist = []
    for ftype in types:
        filelist = filelist + glob.glob(dir + '/*.' + ftype)

    n_images = len(filelist)


    print(f"cropping {n_images} images. from directory {dir[-dir[::-1].find('/'):]}. class folder has a total of \
                    {len(glob.glob(dir + '/*'))} files.")


    for fname in (tqdm(filelist) if plot_bar else filelist):
        img = Image.fromarray(crop_image(np.array(Image.open(fname)), size=args.img_size))
        img_id = fname[-fname[::-1].find('/'):]
        #print("Test Mode: Would save image to", crop_dir + '/' + img_id)
        img.save(crop_dir + '/' + img_id)

    # Load files from subdirectories
    for subdir, dirs, files in os.walk(dir):
        for subsubdir in dirs:
            crop_and_save_images(args, dir=os.path.join(subdir, subsubdir), crop_dir=crop_dir, types=types, load_subdirs=load_subdirs)




def create_csv_by_dirs(img_dir, csv_dir, write_head=True):
    print("Collecting filelist for directory: " + img_dir)

    types=['JPG', 'jpg', 'JPEG', 'jpeg', 'png', 'PNG']
    filelist = []
    for ftype in types:
        filelist = filelist + glob.glob(img_dir + '/*.' + ftype)

    n_images = len(filelist)


    print(f"collecting labels for {n_images} images. from directory {img_dir[-img_dir[::-1].find('/'):]}. class folder has a total of \
                    {len(glob.glob(img_dir + '/*'))} files.")

    with open(csv_dir, 'a') as f:
        if write_head:
            f.write("name" + "," + "target" + "\n")
        for fname in tqdm(filelist):
            img_target = fname.split("/")[-2]
            img_id = fname[-fname[::-1].find('/'):].replace(" ", "_")
            f.write(img_id + "," + img_target + "\n")
    # Load files from subdirectories
    for subdir, dirs, files in os.walk(img_dir):
        for subsubdir in dirs:
            create_csv_by_dirs(img_dir + "/" + subsubdir, csv_dir, write_head=False)


if __name__ == "__main__":
    #create_csv_by_dirs('/mnt/CVAI/data/fixmatch/plant_village/color', '/data/fixmatch/plant_village/meta_data.csv')
    main()



