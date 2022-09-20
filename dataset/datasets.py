from .randaugment import RandAugmentMC

import logging
from dataset.custom_dataset import *
from dataset.custom_img_folder import *

logger = logging.getLogger(__name__)


normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

"""# No longer used
DATASET_GETTERS = {'plant_village': get_custom_dataset,  # get_plant_village,
                   'mini_plant_village': get_custom_dataset,
                   'euro_sat': get_custom_dataset,# 'euro_sat': get_euro_sat,
                   'isic': get_img_folder,
                   'mini_isic': get_custom_dataset,
                   'chestx': get_custom_dataset,
                   'mini_chestx': get_custom_dataset,
                   'cem_holes': get_img_folder,
                   'cem_squares': get_img_folder,
                   'clipart': get_img_folder,
                   'infograph': get_img_folder,
                   'painting': get_img_folder,
                   'quikdraw': get_img_folder,
                   'real': get_img_folder,
                   'sketch': get_img_folder,
                   }
"""
# Dict to return dataset classes that are required to load the different datasets
DATASET_CLASSES = {'plant_village': (CustomDataset, CUSTOMSSL),
                   'euro_sat': (CustomDataset, CUSTOMSSL),
                   'isic': (ImageFolder, FOLDERSSL),
                   'chestx': (CustomDataset, CUSTOMSSL),
                   'cem_holes': (ImageFolder, FOLDERSSL),
                   'cem_squares': (ImageFolder, FOLDERSSL),
                   'toy': (ImageFolder, FOLDERSSL),
                   'clipart': (ImageFolder, FOLDERSSL),
                   'infograph': (ImageFolder, FOLDERSSL),
                   'painting': (ImageFolder, FOLDERSSL),
                   'quickdraw': (ImageFolder, FOLDERSSL),
                   'real': (ImageFolder, FOLDERSSL),
                   'sketch': (ImageFolder, FOLDERSSL),
                   'sketch10': (ImageFolder, FOLDERSSL)}

def get_balanced_class_sampler(args, targets):
    count = torch.zeros(args.num_classes)
    for i in range(args.num_classes):
        count[i] = len(np.where(np.array(targets) == i)[0])
    N = float(sum(count))
    class_weights = N / count
    sample_weights = np.zeros(len(targets))
    for i in range(args.num_classes):
        sample_weights[np.array(targets) == i] = class_weights[i]
    assert sample_weights.min() > 0, "Some samples have a weight of zero, they will never be used"
    return torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(targets))  # args.batch_size)



def get_dataloaders(args, root, augmentor):
    BaseDataset, SSLDataset = DATASET_CLASSES[args.dataset]
    if args.augment == "dada":
        if not args.nofixmatch:
            # for FixMatch and DADA use weak augment for labeled images
            transform_labeled = augmentor.weak
            transform_unlabeled = None  # unlabeled augment is managed by DADA
        else:
            # for Supervised training with DADA augmentation will be managed by DADA
            transform_labeled = None
            transform_unlabeled = None  # irrelevant as unlabeled data is not used

    else:
        if not args.nofixmatch:
            # for FixMatch use weak augment for labeled iamges
            transform_labeled = augmentor.weak

            transform_unlabeled = augmentor  # regular FixMatch transform (Strong + weak)
        else:
            # for Supervised training use chosen augmentation for labeled data
            transform_labeled = augmentor.strong
            transform_unlabeled = None  # irrelevant as unlabeled data is not used


    transform_test = augmentor.test

    #transform_val_aug = transform_labeled
    if BaseDataset is ImageFolder:
        base_dataset = BaseDataset(os.path.join(args.img_dir, "train"), transform=None, loader=safe_load_image)
    else:
        base_dataset = BaseDataset(args, transform=None, train=True, store_all_data=True)

    args.classes = base_dataset.classes
    # creating Index Lists (like [1, 5, 23, 109,...], defining which elements remain with labels)
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    if args.augment == "dada":
        # to learn augmentation we split the train set again in train and val set
        sss = StratifiedShuffleSplit(n_splits=5, test_size=args.val_portion, random_state=0)
        sss = sss.split(list(range(len(train_labeled_idxs))), torch.Tensor(base_dataset.targets)[train_labeled_idxs])
        train_labeled_idxs_idxs, val_labeled_idxs_idxs = next(sss)  # creating indicies for the index list (meta :D)
        val_labeled_idxs = train_labeled_idxs[val_labeled_idxs_idxs]
        train_labeled_idxs = train_labeled_idxs[train_labeled_idxs_idxs]
        # at this point the labeled val set is still included in the unlabeled date so we need to exclude them
        train_unlabeled_idxs = list(train_unlabeled_idxs)
        for ix in sorted(val_labeled_idxs, reverse=True):  # sorting might be unneeded but better safe than sorry
            train_unlabeled_idxs.remove(ix)
        train_unlabeled_idxs = np.array(train_unlabeled_idxs)

        if BaseDataset is ImageFolder:
            val_labeled_dataset = SSLDataset(args, val_labeled_idxs, train=True, transform=None,
                                            path=os.path.join(args.img_dir, "train"), loader=safe_load_image)
        else:
            val_labeled_dataset = SSLDataset(args, val_labeled_idxs, train=True, transform=None,
                                             images=base_dataset.total_images, targets=base_dataset.total_targets,
                                             classes=base_dataset.classes)




    if BaseDataset is ImageFolder:
        train_labeled_dataset = SSLDataset(args, train_labeled_idxs, train=True, transform=transform_labeled,
                                          path=os.path.join(args.img_dir, "train"), loader=safe_load_image)
        train_unlabeled_dataset = SSLDataset(args, train_unlabeled_idxs, train=True,
                                             transform=transform_unlabeled,
                                             path=os.path.join(args.img_dir, "train"), loader=safe_load_image)
        test_dataset = BaseDataset(os.path.join(args.img_dir, "val"), transform=transform_test, loader=safe_load_image)
    else:
        train_labeled_dataset = SSLDataset(args, train_labeled_idxs, train=True, transform=transform_labeled,
                                           images=base_dataset.total_images, targets=base_dataset.total_targets,
                                           classes=base_dataset.classes)
        train_unlabeled_dataset = SSLDataset(args, train_unlabeled_idxs, train=True,
                                             transform=transform_unlabeled,
                                             images=base_dataset.total_images, targets=base_dataset.total_targets,
                                             classes=base_dataset.classes)

        test_dataset = BaseDataset(args, transform=transform_test, train=False,
                                   images=base_dataset.total_images, targets=base_dataset.total_targets,
                                   classes=base_dataset.classes)

    print(f"Labeled train set contains {len(train_labeled_dataset)} images.")
    if args.augment == "dada":
        print(f"Labeled validation set contains {len(val_labeled_dataset)} images.")
    print(f"Unlabeled train set contains {len(train_unlabeled_dataset)} images.")
    print(f"test set contains {len(test_dataset)} images.")

    # creating samplers
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    if args.balance:
        train_labeled_sampler = get_balanced_class_sampler(args, train_labeled_dataset.targets)
    else:
        train_labeled_sampler = train_sampler(train_labeled_dataset.targets)
    if args.augment == "dada":
        # TODO: hier sollte eventuell gebalanced werden (wie beim Train set)
        val_labeled_sampler = train_sampler(val_labeled_dataset.targets)

    # Creating Datasets with learnable augmentation
    if args.augment == "dada" and args.nofixmatch:
        val_labeled_dataset = AugmentDataset(val_labeled_dataset, augmentor.pre, augmentor.after, transform_test,
                                             args.sub_policies, False, args.magnitudes)
        train_labeled_dataset = AugmentDataset(train_labeled_dataset, augmentor.pre, augmentor.after, transform_test,
                                               args.sub_policies, True, args.magnitudes)
    if args.augment == "dada" and not args.nofixmatch:
        val_labeled_dataset = AugmentDataset(val_labeled_dataset, augmentor.pre, augmentor.after, transform_test,
                                             args.sub_policies, False, args.magnitudes)
        train_unlabeled_dataset = AugmentDataset(train_unlabeled_dataset, augmentor.pre, augmentor.after, transform_test,
                                                 args.sub_policies, True, args.magnitudes, ssl=True)

    # creating dataloaders
    if args.augment == "dada":
        val_labeled_loader = torch.utils.data.DataLoader(
            val_labeled_dataset, batch_size=args.batch_size,
            sampler=val_labeled_sampler, drop_last=False,
            pin_memory=True, num_workers=args.num_workers)
        if args.nofixmatch:
            # For supervised training with DADA (DADA labeled loader, regular labeled loader):
            train_labeled_loader = torch.utils.data.DataLoader(
                train_labeled_dataset, batch_size=args.batch_size, shuffle=False,
                sampler=train_labeled_sampler, drop_last=False,
                pin_memory=True, num_workers=args.num_workers)

            train_unlabeled_loader = DataLoader(
                train_unlabeled_dataset,
                sampler=train_sampler(train_unlabeled_dataset),
                batch_size=args.batch_size * args.mu,
                num_workers=args.num_workers,
                drop_last=True)
        else:
            # For FixMatch training with DADA (regulara labeled loader DADA unlabeled loader):
            train_labeled_loader = DataLoader(
                train_labeled_dataset,
                sampler=train_labeled_sampler,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                drop_last=True)
            train_unlabeled_loader = torch.utils.data.DataLoader(
                train_unlabeled_dataset, batch_size=args.batch_size * args.mu, shuffle=False,
                sampler=train_sampler(train_unlabeled_dataset), drop_last=False,
                pin_memory=True, num_workers=0
            )
            print("FOR DADA augmentation num workers was set to 0 as it does not work otherwise")
    else:
        # Regular dataloaders for FixMatch and Supervised (without DADA):
        train_labeled_loader = DataLoader(
            train_labeled_dataset,
            sampler=train_labeled_sampler,  # train_sampler(labeled_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True)

        val_labeled_loader = None # We don't need val dataset whithout DADA

        train_unlabeled_loader = DataLoader(
            train_unlabeled_dataset,
            sampler=train_sampler(train_unlabeled_dataset),
            batch_size=args.batch_size * args.mu,
            num_workers=args.num_workers,
            drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size * (args.mu + 1),  # int(args.batch_size / 2),
        num_workers=args.num_workers
        )

    data_loaders = {"labeled": train_labeled_loader, "unlabeled": train_unlabeled_loader,
                    "val": val_labeled_loader, "test": test_loader}
    return data_loaders





