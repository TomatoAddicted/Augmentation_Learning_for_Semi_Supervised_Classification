import pickle

import PIL.Image
import numpy
import torchvision.transforms

np = numpy
"""
#import matplotlib.pyplot as plt
from torchvision import datasets
from pathlib import Path
from torchvision.io import read_image
#import cv2 as cv
import pandas as pd
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from torchvision.models import resnet18
#from dataset.mini_imagenet import MiniImageNet
import matplotlib.pyplot as plt
from time import sleep
from torchvision import transforms
from tqdm import tqdm
from dataset.randaugment import *
from dataset.utils import load_images_from_dir, create_pickle_label_by_csv, open_classwise_pickles, CustomToTensor
from utils import accuracy, AverageMeter
from train import interleave
"""
import cv2 as cv
import matplotlib.pyplot as plt
#from dataset.utils import create_train_test_split
#from config import ssd_path
#root = "/mnt/CVAI/data/test_images"
#root = "/mnt/CVAI/data/fixmatch/isic/ISIC-images/2018_JID_Editorial_Images"
#csv_dir = "/mnt/CVAI/data/fixmatch/isic/ISIC-images/metadata.csv"
#csv_dir = "/data/fixmatch/chestx/metadata.csv"
#pkl_dir = "/data/fixmatch/mini_isic/"

#fname = "/data/fixmatch/chestx/images/00023372_000.png"
#fname = "/mnt/CVAI/data/fixmatch/isic/ISIC-images/UDA-1/ISIC_0000468.jpg"
#fname = "/nobackup/users/timfro/fixmatch_data/cem_holes/train/high/21mar02c_grid1c_00036gr_00014sq_v01_00005hl_v01_00025en-a_crop.png"
#fname = "/nobackup/users/timfro/fixmatch_data/"

#create_train_test_split(ssd_path + '/domainnet/clipart/train', ssd_path + '/domainnet/clipart/val')

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
args = Namespace()


from dataset.load_augment import visualize_dada_process

#folders = ["intersave", "intersave2", "intersave3", "lambda_u_0", "lambda_u_0_2", "lambda_u_0_3"]
folders = [
    "one_percent",
    "five_percent",
    "ten_percent",
    #"five_percent_lower_lr/even_lower",
    #"five_percent_lower_lr",
    #"one_percent_lower_lr/even_lower",
    #"one_percent_lower_lr",
]
for folder in folders:
    path = "/nobackup/users/timfro/run_scripts/dada/cryo/" + folder + "/policy_files"
    #path = "/nobackup/users/timfro/run_scripts/domainnet/dada/sketch10/" + folder + "/policy_files"
    #path = "/nobackup/users/timfro/run_scripts/dada/euro_sat/" + folder + "/policy_files"

    visualize_dada_process(path, title=folder, ma=2)

"""
avg_class_acc = [AverageMeter() for i in range(args.num_classes)]

outputs = torch.tensor([[0,1],
                        [0,1],
                        [0,1],
                        [1,0],
                        [1,0],
                        [1,0],
                        [1,0],
                        [1,0],
                        [1,0],
                        [1,0],
                     ])
targets = torch.tensor([1,1,0,0,0,0,0,0,0,0])

curr_aca = 0 # current average class accuracy
for i in range(args.num_classes):
    prec1, prec5 = accuracy(outputs[targets == i], targets[targets == i], topk=(1, min(args.num_classes, 5)))
    avg_class_acc[i].update(prec1, len(np.where(targets == i)))
    curr_aca += avg_class_acc[i].avg/args.num_classes

print(curr_aca)

"""

"""


argsc.pkl_dir = "/data/fixmatch/plant_village/"
imagesc, targetsc, classesc = open_classwise_pickles(argsc)

test_image = imagesc[0, :, :]

print(test_image.shape)
tripple_image = np.tile(test_image, (1, 1, 3))
print(tripple_image.shape)
"""

"""
randaugment = RandAugmentPC(n=2, m=10, gray_scale=True)

ops = randaugment.augment_pool



test_image = np.random.uniform(0, 255, (512, 512, 3))

print(test_image.shape)

test_image = Image.fromarray(test_image[:, :, 0])

pre_trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32,
                          padding=int(32 * 0.125),
                          padding_mode='reflect')])

test_image = pre_trans(test_image)

for op, max_v, bias in ops:
    img = op(test_image, v=10, max_v=max_v, bias=bias)
    try:
        img = op(test_image, v=10, max_v=max_v, bias=bias)
        print(f"operation {op} worked")
    except:
        print(f"operation {op} caused an error")


"""
"""
#print(imagesc)
test = torch.zeros(33)
print("test",test)
print(targetsc)
print(classesc)

argsp = Namespace()
argsp.pkl_dir = "/data/fixmatch/mini_plant_village/"
imagesp, targetsp, classesp = open_classwise_pickles(argsp)

#print(imagesp)
print(targetsp)
print(classesp)



with open(argsc.pkl_dir + "Effusion.pkl", 'rb') as f:
    chestx_data = pickle.load(f)

with open(argsp.pkl_dir + "Peach___healthy.pkl", 'rb') as f:
    plant_data = pickle.load(f)

plt.imshow(imagesp[0]/255)
plt.show()
plt.imshow(imagesc[0]/255)
plt.show()
"""



"""

#images, targets, classes = create_pickle_label_by_csv(img_dir=root, pkl_dir="/mnt/CVAI/data/fixmatch/isic/",
#                                                      csv_dir=csv_dir, img_size=(128, 128), classwise=True)

d = pd.read_csv(csv_dir)
print(list(set(d['Finding_Labels'].tolist())))
print(len(list(set(d['Finding_Labels'].tolist()))))

#x = d.query('Height] == "2749"')['Finding Labels'].values[0]
x = d.query('Image_Index == "00007857_015.png"')['Finding_Labels'].values[0]
print(x, type(x))
#print(images.shape)
#print(len(targets))
#load_images_from_dir(root)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

args = Namespace()
args.pkl_dir = "/data/fixmatch/isic/ignored_classes/"
images, targets, classes = open_classwise_pickles(args)


class_data = {'target': "other",
              'images': images}
with open("/mnt/CVAI/data/fixmatch/chestx" + '/' + "other" + ".pkl",
          'wb') as f:  # If a PathNotFound error occurs here, try: open(root + '/' + dataset + '.pkl' ...
    pickle.dump(class_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
"""

"""png = np.array(Image.open(root + "/test.png"))
jpg = np.array(Image.open(root + "/test.jpg"))

print("png.shape", png.shape)
print("jpg.shape", jpg.shape)
print("PNG", png[0, 0])
print("jpg", jpg[0, 0])"""


"""

x = [1,2,3]
y = [4,5,6]

print(False and x or y)
print(x if False else y)
"""
"""
for i in tqdm(range(100)):
    sleep(3)
    print("hihi")
    """
"""
mini_train = MiniImageNet(root, train=True, store_all_data=True)

for i, class_i in enumerate(mini_train.classes):
    class_images = mini_train.total_images[mini_train.total_targets == i]

    print(i, len(class_images))
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
    fig.suptitle("class number" + str(i) + ", class: " + str(class_i))
    ax1.imshow(class_images[0])
    ax2.imshow(class_images[1])
    ax3.imshow(class_images[2])
    ax4.imshow(class_images[-2])
    ax5.imshow(class_images[-1])
    plt.show()
"""



"""
img_dir = root + '/euro_sat/'
pkl_dir = root + '/euro_sat/euro_sat.pkl'
with open(pkl_dir, 'rb') as f:
    data = pickle.load(f)


classes = list(data["class_dict"].keys())
print(data["class_dict"][classes[0]][0])
for i in range(len(classes)):
    print(i, classes[i], len(data["class_dict"][classes[i]][0]))

print("end")
"""
"""
x = np.array(["apfel", "birne"])


print(np.where(x == "apfel"))


"/home/fixmatch"

targets = np.array(["a", "b", "a", "d", "a"])
print(np.where(targets == "a"))

targets = np.array([1,2,3,4,2])
print(np.where(targets > 2))

with open('test.pkl', 'wb') as f:
    pickle.dump(targets, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('test.pkl', 'rb') as f:
    data0 = pickle.load(f)

print(targets)
print(data0)
print(targets == data0)
"""

"""
import os
root = '/home/data2/fixmatch/PlantVillage-Dataset/raw/color/'
path_len = len(root)
#print(path_len)

import glob

num_images = 0
labels = []
for subdir, dirs, files in os.walk(root):
    if len(files) == 0:
        continue
    num_images += len(subdir)
    dims_img = np.array(Image.open(subdir + '/' + files[0])).shape
    labels.append(subdir[path_len:])

dims_images = (num_images, dims_img[0], dims_img[1], dims_img[2])
total_images = torch.zeros(dims_images)
total_targets = torch.zeros(num_images)
num_loaded = 0
for subdir, dirs, files in os.walk(root):
    if len(files) == 0:
        continue
    class_label = subdir[path_len:]
    filelist = glob.glob(subdir + '/*.JPG')
    total_images[num_loaded : num_loaded + len(filelist)] = torch.Tensor([np.array(Image.open(fname)) for fname in filelist])
    total_targets[num_loaded : num_loaded + len(filelist)] = class_label
    num_loaded += len(filelist)
"""

"""

#image = read_image(rootdir + 'Apple___Apple_scab/ff99efdc-a9f8-4360-9c64-f8274f456be5___FREC_Scab 3161.JPG')
image = read_image('/home/data2/fixmatch/PlantVillage-Dataset/raw/color/Apple___Apple_scab/fdc58f83-8a94-4f0c-aefe-c6f2b5028682___FREC_Scab 3014.JPG')
image = image.T
plt.imshow(image)
plt.show()
"""




