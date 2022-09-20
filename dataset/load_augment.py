import random

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import glob
import os

from dada.search_relax.primitives import sub_policies as sub_policies_raw
from dada.search_relax.train_search_paper import avg_op_pos

import torchvision.transforms as transforms
import torch
from torch import Tensor
import numpy as np
import PIL


random_mirror = True

class LoadAugment(object):
    def __init__(self, policy_path, k=20, n=2, rand=False, m_mode="uniform", weight_policies=False, sharpen=True, T=0.05):
        self.k = k  # amount of subpoliciies loaded (top k policies will be used for training)
        self.n = n  # amount of subpolicies applied to each image
        self.policy_path = policy_path
        self.m_mode = m_mode
        #self.sub_policies = load_policies(policy_path, self.k, rand=rand)
        self.ops, self.w = load_policies(policy_path, self.k, rand=rand)  # each elem in ops is tuple (op, p)
        #plt.bar(range(len(self.w)), self.w)
        #plt.show()
        if sharpen:
            self.w = self.sharpen_weights(self.w, T)
        else:
            # w's might not exactly add up to 1 (sum is like 0.9999999532...) so we add that small mising margin to w0
            # This will have no effect but avoids an error when sampling
            sum_w = sum(self.w)
            delta_w = 1 - sum_w
            self.w[0] += delta_w
        plt.bar(range(len(self.w)), self.w)
        plt.show()
        self.augment_dict = get_augmentDict()
        self.weight_policies = weight_policies


    def sharpen_weights(self, w, T=0.05):
        sharpened_w = np.zeros(len(w))
        exp_sum = sum([np.e**(w[i] / T) for i in range(len(w))])
        for i in range(len(w)):
            sharpened_w[i] = (np.e**(w[i] / T)) / exp_sum

        return sharpened_w

    def sample_magnitude(self, m_base=0.5):
        if self.m_mode == "uniform":
            # sample uniform, ignoring m_base
            return float(np.random.randint(1, 10) / 10)
        if self.m_mode == "normal":
            # sample normal distribution around m_base
            return np.clip(np.random.normal(m_base, 0.1), 0, 1)
        if self.m_mode == "fix":
            # just use m_base
            return m_base

    def __call__(self, img):
        if self.weight_policies:
            t_idxs = np.random.choice(range(len(self.ops)), replace=False, p=self.w, size=(self.n, 1))  # sample subpolicies by weights
            transformations = [self.ops[t_idxs[i, 0]] for i in range(len(t_idxs))]
        else:
            transformations = random.sample(self.ops, self.n)  # sample subpolicies uniformly
        augmentations = [
            (self.augment_dict[op[0]](self.sample_magnitude(float(m[0])), float(p[0])),
             self.augment_dict[op[1]](self.sample_magnitude(float(m[1])), float(p[1]))) for op, p, m in transformations]

        augmentations = list(sum(augmentations, ()))
        trans = transforms.Compose(augmentations)

        return trans(img)


def load_policies(policy_path, k=20, rand=False):
    if not rand:  # Load sub policies from file
        with open(policy_path, 'r') as f:
            lines = f.readlines()
        line_list = [line.split(' ') for line in lines]
        pretty_values = [[value.replace('(', '').replace(')', '').replace(':', '').replace(',', '') for value in value_list]
                         for value_list in line_list]
        ops = [(ops[1], ops[5]) for ops in pretty_values]
        p = [(ops[2], ops[6]) for ops in pretty_values]
        m = [(ops[3], ops[7]) for ops in pretty_values]
        w = [float(ops[4]) for ops in pretty_values]
    else:  # generate random sub policies (for baseline experiments)
        ops = random.sample(sub_policies_raw, k)
        p = [(0.5, 0.5) for _ in range(k)]  # all probabilities and magnitudes are set 0.5
        m = [(0.5, 0.5) for _ in range(k)]
        w = [1 / k for _ in range(k)]
    """
    augment_dict = get_augmentDict()

    transformations = [
        (augment_dict[o[0]](float(m[i][0]), float(p[i][0])), augment_dict[o[1]](float(m[i][1]), float(p[i][1]))) for
        i, o in enumerate(ops)]

    return transformations[:k]
    """
    return [(ops[i], p[i], m[i]) for i in range(k)], w


class MyRandomAffine(transforms.RandomAffine):

    def __init__(self,
                 degrees,
                 translate=None,
                 scale=None,
                 shear=None,
                 p=0.5):
        super().__init__(degrees=degrees, translate=translate, scale=scale, shear=shear)
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            new_img = super().forward(img)
            return new_img
        return img


class MyColorJitter(object):

    def __init__(self,
                 contrast=0,
                 color=0,
                 brightness=0,
                 sharpness=0,
                 p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.color = color
        self.sharpness = sharpness
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            if self.contrast > 0:
                img = PIL.ImageEnhance.Contrast(img).enhance(self.contrast)
            if self.color > 0:
                img = PIL.ImageEnhance.Color(img).enhance(self.color)
            if self.brightness > 0:
                img = PIL.ImageEnhance.Brightness(img).enhance(self.brightness)
            if self.sharpness > 0:
                img = PIL.ImageEnhance.Sharpness(img).enhance(self.sharpness)
            return img

        return img


class Cutout_(object):
    def __init__(self,
                 magnitude=0,
                 p=0.6):

        self.magnitude = magnitude
        self.p = p

    def __call__(self, img):

        if self.magnitude <= 0:
            return img

        length = self.magnitude * img.size[0]
        if torch.rand(1).item() < self.p:
            w, h = img.size
            x0 = np.random.uniform(w)
            y0 = np.random.uniform(h)

            x0 = int(max(0, x0 - length / 2.))
            y0 = int(max(0, y0 - length / 2.))
            x1 = min(w, x0 + length)
            y1 = min(h, y0 + length)

            xy = (x0, y0, x1, y1)
            color = (125, 123, 114)
            # color = (0, 0, 0)
            img = img.copy()
            PIL.ImageDraw.Draw(img).rectangle(xy, color)
            return img
        return img


def ShearX(v, p):  # [-0.3, 0.3]
    m = standardize(v, 0, 0.3)
    m = m * 90

    if random_mirror and random.random() > 0.5:
        m = -m

    return MyRandomAffine(degrees=0, shear=(m, m, 0, 0), p=p)


def ShearY(v, p):  # [-0.3, 0.3]
    m = standardize(v, 0, 0.3)
    m = m * 90
    if random_mirror and random.random() > 0.5:
        m = -m

    return MyRandomAffine(degrees=0, shear=(0, 0, m, m), p=p)


def TranslateX(v, p):  # [-150, 150] => percentage: [-0.45, 0.45]
    m = standardize(v, 0, 0.45)

    return MyRandomAffine(degrees=0, translate=(m, 0), p=p)


def TranslateY(v, p):  # [-150, 150] => percentage: [-0.45, 0.45]
    m = standardize(v, 0, 0.45)

    return MyRandomAffine(degrees=0, translate=(0, m), p=p)


def Rotate(v, p):  # [-30, 30]
    m = standardize(v, 0, 30)
    if random_mirror and random.random() > 0.5:
        m = -m

    return MyRandomAffine(degrees=(m, m), p=p)


def AutoContrast(_, p):
    return transforms.RandomAutocontrast(p)


def Invert(_, p):
    return transforms.RandomInvert(p)


def Equalize(_, p):
    return transforms.RandomEqualize(p)


def Flip(_, p):  # not from the paper
    return transforms.RandomHorizontalFlip(p)


def Solarize(v, p):  # [0, 256]
    m = standardize(v, 0, 256)
    return transforms.RandomSolarize(m, p)


def Posterize(v, p):  # [4, 8]
    m = standardize(v, 4, 8)
    m = int(m)
    return transforms.RandomPosterize(m, p)


def Contrast(v, p):  # [0.1,1.9]
    m = standardize(v, 0.1, 1.9)
    return MyColorJitter(contrast=m, p=p)


def Color(v, p):  # [0.1,1.9]
    m = standardize(v, 0.1, 1.9)

    return MyColorJitter(color=m, p=p)


def Brightness(v, p):  # [0.1,1.9]
    m = standardize(v, 0.1, 1.9)

    return MyColorJitter(brightness=m, p=p)


def Sharpness(v, p):  # [0.1,1.9]
    m = standardize(v, 0.1, 1.9)

    return MyColorJitter(sharpness=m, p=p)


def Cutout(v, p):
    m = standardize(v, 0, 0.2)
    return Cutout_(m, p)


def augment_list():  # 14 Operations
    aug_list = [
        ShearX,  # 0
        ShearY,  # 1
        TranslateX,  # 2
        TranslateY,  # 3
        Rotate,  # 4
        AutoContrast,  # 5
        Invert,  # 6
        Equalize,  # 7
        Solarize,  # 8
        Posterize,  # 9
        Contrast,  # 10
        Color,  # 11
        Brightness,  # 12
        Sharpness,  # 13
        Cutout  # 14
    ]
    return aug_list


def standardize(v, min, max):
    m = v * (max - min) + min
    return m


def get_augmentDict():
    augment_dict = {fn.__name__: fn for fn in augment_list()}
    return augment_dict

def get_augmentColorDict():
    aug_color_dict = {
        "ShearX": "blue",  # 0
        "ShearY": "orange",  # 1
        "TranslateX": "green",  # 2
        "TranslateY": "red",  # 3
        "Rotate": "purple",  # 4
        "AutoContrast": "brown",  # 5
        "Invert": "pink",  # 6
        "Equalize": "gray",  # 7
        "Solarize": "olive",  # 8
        "Posterize": "cyan",  # 9
        "Contrast": "deepskyblue",  # 10
        "Color": "magenta",  # 11
        "Brightness": "lime",  # 12
        "Sharpness": "navy",  # 13
        "Cutout": "tomato"  # 14
    }
    return aug_color_dict


def running_mean(x, N):
    """ x == an array of data. N == number of samples per average """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def visualize_dada_process(path, title=None, ma=0):
    """
    reads all augment files from path and creates a plot to vislaulize the process of learning
    plot is saved in current working directory
    """
    all_files = sorted(glob.glob(os.path.join(path, "*augment.txt*")))  # Files are indexed like <fname>.txt_1
    #all_files = sorted(glob.glob(path))  # Files are indexed like <fname>.txt_1
    all_ops = list(get_augmentDict().keys())

    # get indexes from filenames (numeric)
    idxs = np.zeros(len(all_files))
    for i, filename in enumerate(all_files):
        try:
            idxs[i] = int(filename[-2:])
        except:
            idxs[i] = int(filename[-1:])

    # sort filnames by numeric indexes
    all_files = [f for _, f in sorted(zip(idxs, all_files))]

    # TODO: Farben eindeutig zuteilen
     # read all files and measure avg position of each augment and gather them in trajectories
    trajectories = np.zeros((len(all_ops), len(all_files)))
    for op_i in range(len(all_ops)):
        for file_i in range(len(all_files)):
            avg_pos_op_i = avg_op_pos(all_files[file_i], all_ops[op_i])
            trajectories[op_i, file_i] = avg_pos_op_i
        trajectories[op_i] = np.pad(np.convolve(trajectories[op_i], np.ones(ma*2 +1)/(ma*2 + 1), mode='valid'), (2*ma, 0), 'edge')

    # create plot to visualize trajectories
    augment_color_dict = get_augmentColorDict()
    time_axis = range(len(all_files))
    for op_i in range(len(all_ops)):
        plt.plot(time_axis, trajectories[op_i], label=all_ops[op_i],
                 color=augment_color_dict[all_ops[op_i]])
    plt.xlabel("epochs trained")
    plt.ylabel("avg position")
    plt.legend()
    if title is None:
        plt.title("Avg positions óf all operations during dada training")
    else:
        plt.title(title)
    plt.savefig("dada_process.png")
    plt.savefig("dada_process.pdf")
    plt.show()



