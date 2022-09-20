from machine import MACHINE # Import information about which machine we are on
#dataset_path = "/data/fixmatch"  # g4
if MACHINE == "cvai": # cvai
    dataset_path = '/mnt/CVAI/data/fixmatch'  #cvai-gpu
    ssd_path = "/data/fixmatch" # cvai-gpu
    interpreter_path = "/home/tim/miniconda3/envs/fixmatch/bin/python3"  # tim cvai-gpu
    machine = "cvai"
elif MACHINE == "satori": # satori
    dataset_path = "/nobackup/users/timfro/fixmatch_data"  #satori tim
    ssd_path = "/nobackup/users/timfro/fixmatch_data/pre_cropped"
    interpreter_path = "/nobackup/users/timfro/anaconda3/envs/fixmatch/bin/python3"
    machine = "satori"


def get_dataset_params(args, root, no_model=False):
    # default values (overwridden if specified)
    args.scale = (0.5, 1.)
    if hasattr(args, 'augment') and  args.augment == "dada":
        args.unrolled = True
    if args.dataset == "euro_sat":
        # 10% of training images = 2100
        args.preload_data = True
        args.other_class = False
        args.img_dir = root + '/euro_sat/image_data/'
        args.pkl_dir = root + '/euro_sat/'  # root + '/euro_sat/euro_sat.pkl'
        args.csv_dir = None
        args.classwise_pkl = True
        args.img_size = (64, 64, 3)
        args.min_class_labels = 0
        #args.cropsize = 32 if args.cropsize is None else args.cropsize
        # this is not required as its calculated in the code, but I'll keep it here for fun.
        args.mean, args.std = [0.34436897, 0.38029233, 0.40777751],  [0.20368513, 0.13663637, 0.11484352]

    elif args.dataset == "plant_village":
        # 10% of training images = 4300
        args.preload_data = True
        args.other_class = False
        args.orig_img_dir = root + '/plant_village/color'
        args.pkl_dir = root + '/plant_village/' # also place for classes.txt file
        args.img_dir = args.orig_img_dir #ssd_path + '/plant_village'
        args.csv_dir = None #ssd_path + '/plant_village/meta_data.csv'
        args.target_attr = "target"
        args.index_attr = "name"
        args.id_with_type = True
        args.classwise_pkl = True
        args.img_size = (256, 256, 3)
        #args.cropsize = 32 if args.cropsize is None else args.cropsize
        args.min_class_labels = 0
        args.mean, args.std = [0.46642168, 0.48910507, 0.41036344], [0.19924541, 0.17493751, 0.21752516]

    elif args.dataset == "mini_plant_village":
        args.preload_data = True
        args.img_dir = root + '/mini_plant_village/color'
        args.pkl_dir = root + '/mini_plant_village/'
        args.csv_dir = None
        args.classwise_pkl = True
        args.img_size = None

    elif args.dataset == "isic":
        args.preload_data = False
        args.other_class = False # Note isic has an 'other' class, but it is sorted manually
        #args.orig_img_dir = root + '/isic/ISIC-images'
        args.orig_img_dir = (root if machine == "satori" else ssd_path) + '/isic_folder'
        args.img_dir = ssd_path + '/isic_folder'
        args.pkl_dir = root + '/isic/'
        args.csv_dir = root + '/isic/ISIC-images/metadata.csv'
        args.target_attr = "meta.clinical.diagnosis"
        args.index_attr = "name"
        args.id_with_type = False
        args.classwise_pkl = True
        args.img_size = (256, 256, 3)
        #args.cropsize = 240 if args.cropsize is None else args.cropsize
        args.min_class_labels = 0
        args.mean, args.std = [0.7484, 0.5867, 0.5562], [0.1944, 0.1925, 0.2126]


    elif args.dataset == "mini_isic":
        args.preload_data = False
        args.other_class = False
        args.orig_img_dir = root + '/mini_isic/'  # origin of images
        args.img_dir = ssd_path + '/mini_isic'  # dir of preprocessed images
        args.pkl_dir = root + '/mini_isic/'
        args.csv_dir = root + '/isic/ISIC-images/metadata.csv'
        args.target_attr = "meta.clinical.diagnosis"
        args.index_attr = "name"
        args.id_with_type = False
        args.classwise_pkl = True
        args.img_size = (256, 256, 3)
        #args.cropsize = 32 if args.cropsize is None else args.cropsize
        args.min_class_labels = 5
        args.mean, args.std = [0.7484, 0.5867, 0.5562], [0.1944, 0.1925, 0.2126]

    elif args.dataset == "chestx":
        args.preload_data = False
        args.other_class = False
        args.orig_img_dir = root + '/chestx/images/'
        args.img_dir = ssd_path + '/chestx' #root + '/chestx/test_cropped'
        args.pkl_dir = root + '/chestx/'
        args.csv_dir = root + '/chestx/metadata.csv'
        args.target_attr = "Finding_Labels"
        args.index_attr = "Image_Index"
        args.id_with_type = True
        args.classwise_pkl = True
        args.img_size = (256, 256, 3)  # (512, 512, 3)
        #args.cropsize = 128 if args.cropsize is None else args.cropsize
        #print("Croppsize: ", args.cropsize)
        args.min_class_labels = 0
        args.mean, args.std = [0.4989, 0.4989, 0.4989], [0.2492, 0.2492, 0.2492]


    elif args.dataset == "mini_chestx":
        args.preload_data = False
        args.other_class = False
        args.orig_img_dir = root + '/mini_chestx/images/'
        args.img_dir = ssd_path + '/mini_chestx'
        args.pkl_dir = root + '/mini_chestx/'
        args.csv_dir = root + '/mini_chestx/metadata.csv'
        args.target_attr = "Finding_Labels"
        args.index_attr = "Image_Index"
        args.id_with_type = True
        args.classwise_pkl = True
        args.img_size = (256, 256, 3)  # (512, 512, 3)
        #args.cropsize = 128 if args.cropsize is None else args.cropsize
        print("Croppsize: ", args.cropsize)
        args.min_class_labels = 5
        args.mean, args.std = [0.4989, 0.4989, 0.4989], [0.2492, 0.2492, 0.2492]

    elif args.dataset == "cem_holes":
        # train has a total of 2466 train images -> 10% is ~250
        args.preload_data = False
        args.img_dir = root + '/cem_holes'
        args.img_size = (150, 150, 3)
        args.mean, args.std = [0.4912, 0.4912, 0.4912], [0.1493, 0.1493, 0.1493]

    elif args.dataset == "cem_squares":
        # train has a total of 7255 images -> 10% is ~725
        args.preload_data = False
        args.img_dir = root + '/cem_squares'
        args.img_size = (150, 150, 3)
        args.mean, args.std = [0.5906, 0.5906, 0.5906], [0.1863, 0.1863, 0.1863]

    elif args.dataset == "toy":
        args.preload_data = False
        args.img_dir = ssd_path + '/toy'
        args.img_size = (32, 32, 3)
        args.mean, args.std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    elif args.dataset in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch', 'sketch10']:
        # Sketch: 10%: 5600
        # Sketch10: 10%: 525
        # Clipart: 10%: 3927
        args.preload_data = False
        args.orig_img_dir = ssd_path + '/domainnet/' + args.dataset
        if machine == "cvai":
            args.img_dir = ssd_path + '/domainnet/' + args.dataset
        else:
            args.img_dir = ssd_path + '/domainnet_cropped/' + args.dataset
        args.img_size = (256, 256, 3)
        if args.dataset in ['sketch', 'sketch10']:
            args.mean, args.std = [0.8114, 0.8207, 0.8265], [0.2094, 0.2014, 0.1974]
        if args.dataset == 'clipart':
            args.mean, args.std = [0.6692, 0.7056, 0.7274], [0.3423, 0.3075, 0.3030]
        else:
            args.mean, args.std = [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]  # default values

    else:
        raise RuntimeError(f"dataset {args.dataset} is not implemented as custom dataset ")
    args.scale = (0.5, 1.)
    #args.cropsize = int(args.img_size[0] * args.cropratio)
    #print(f"Cropsize is set to {args.cropratio} * {args.img_size[0]} = {args.cropsize}")

    if no_model:
        return args

    # model parameters
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.image_channels = 3
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.image_channels = 3
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    elif args.dataset == 'plant_village':
        args.num_classes = 38
        args.image_channels = 3
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 4
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
        elif args.arch == 'resnet':
            args.model_depth = 50

    elif args.dataset == 'mini_plant_village':
        args.num_classes = 2
        args.image_channels = 3
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
        elif args.arch == 'resnet':
            args.model_depth = 50

    elif args.dataset == 'imagenet':
        args.num_classes = 1000
        args.image_channels = 3
        if args.arch == 'wideresnet':
            args.model_depth = 50
            args.model_width = 1
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
        elif args.arch == 'resnet':
            args.model_depth = 50

    elif args.dataset == 'mini_imagenet':
        args.num_classes = 100
        args.image_channels = 3
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
        elif args.arch == 'resnet':
            args.model_depth = 18

    elif args.dataset == 'euro_sat':
        args.num_classes = 10
        args.image_channels = 3
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
        elif args.arch == 'resnet':
            args.model_depth = 50

    elif args.dataset == 'isic':
        args.num_classes = 9  # aftergrouping all diseases into 9 categories including 'other' (nan does not count as class)
        args.image_channels = 3
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 4
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
        elif args.arch == 'resnet':
            args.model_depth = 50

    elif args.dataset == 'mini_isic':
        args.num_classes = 2
        args.image_channels = 3
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
        elif args.arch == 'resnet':
            args.model_depth = 50

    elif args.dataset == 'chestx':
        args.num_classes = 15  # only choosing classes with at least 50 labels. 836  # contains mixes of 14 different diseases
        args.image_channels = 3
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
        elif args.arch == 'resnet':
            args.model_depth = 50

    elif args.dataset == 'mini_chestx':
        args.num_classes = 17 #18  # Is this true?
        args.image_channels = 3
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
        elif args.arch == 'resnet':
            args.model_depth = 50

    elif args.dataset == 'cem_holes':
        args.num_classes = 2
        args.image_channels = 3
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
        elif args.arch == 'resnet':
            args.model_depth = 50

    elif args.dataset == 'cem_squares':
        args.num_classes = 3
        args.image_channels = 3
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
        elif args.arch == 'resnet':
            args.model_depth = 50

    elif args.dataset == 'toy':
        args.num_classes = 2
        args.image_channels = 3
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
        elif args.arch == 'resnet':
            args.model_depth = 50

    # domainNet
    elif args.dataset in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']:
        args.num_classes = 345
        args.image_channels = 3
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
        elif args.arch == 'resnet':
            args.model_depth = 50

    elif args.dataset == 'sketch10':
        args.num_classes = 10
        args.image_channels = 3
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
        elif args.arch == 'resnet':
            args.model_depth = 50


    else:
        raise RuntimeError(f"Dataset {args.dataset} has no setting yet")
    if args.arch == "simple":
        args.model_depth = 1  # this line is only needed to avoid undefined error

    return args