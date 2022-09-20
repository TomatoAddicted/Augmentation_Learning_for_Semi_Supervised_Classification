import subprocess
from datetime import datetime
from config import interpreter_path
import argparse
import os

fixmatch_parameters = ['--gpu-id', '--num-workers', '--dataset', '--num-labeled', "--expand-labels",'--arch','--total-steps',
              '--eval-step', '--start-epoch', '--batch-size',
              '--lr', '--warmup','--wdecay','--nesterov','--use-ema', '--ema-decay','--mu', '--lambda-u','--T',
              '--threshold', '--out', '--resume', '--seed', "--amp", "--opt_level", "--local_rank", '--no-progress',
              '--plt_env','--log_file', '--augment', '--nofixmatch', '--vanilla', '--balance', '--cropsize', '--tune']

def main():
    print("Running script.py (This Message should only be printed once)!")
    parser = argparse.ArgumentParser(description='Precropping data for fixmatch')
    parser.add_argument('--run', type=str, help='filename of experiment file')

    args = parser.parse_args()

    # reading file. First element of each line should either be default or the number of the run e.g.:
    # default --lr 0.03 --dataset chestx
    # 0 --cropsize 128
    # 1 --cropsize 32
    # if a value specified in default is specified again in one of the runs, it's overwritten.
    parameter_dicts = {}
    legal_identifiers = ["default", "0", "1", "2", "3"]
    identifiers = []
    with open(args.run, 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            elements = line.split(" ")
            if len(elements) <= 1:
                continue
            identifier = elements[0]
            identifiers.append(identifier)
            assert identifier in legal_identifiers, "Invalid beginning of line chose either 'default or" \
                                                                  " index of run (0 - 3):" + line
            parameter_dicts[identifier] = {}
            parameters = elements[1:]
            for i, param in enumerate(parameters):
                if param[:2] == '--':
                    value = parameters[i + 1] if (i + 1 < len(parameters)) and parameters[i + 1][:2] != '--' else ""
                    parameter_dicts[identifier][param] = value

    # collecting all parameters per run (selecing default values if not specified otherwise)
    if "default" in identifiers:
        num_runs = len(identifiers) - 1
        #final_parameter_dicts = [parameter_dicts["default"] for i in range(len(identifiers) - 1)]  # parameter dict for each run
    else:
        num_runs = len(identifiers)
    final_parameter_dicts = [{} for i in range(num_runs)]
    for identifier in identifiers:
        if identifier == "default":
            continue
        for param in parameter_dicts["default"]:
            final_parameter_dicts[eval(identifier)][param] = parameter_dicts["default"][param]
        for param in parameter_dicts[identifier]:
            final_parameter_dicts[eval(identifier)][param] = parameter_dicts[identifier][param]

    # Transforming parameters into list
    final_parameter_lists = [[] for i in range(num_runs)]
    for i, param_dict in enumerate(final_parameter_dicts):
        for param in param_dict:
            final_parameter_lists[i].append(param)
            value = param_dict[param]
            final_parameter_lists[i].append(value)
        # setting a unique gpu id for each task (if not already set)
        try:
            test = final_parameter_dicts[i]["--gpu-id"]
        except:
            final_parameter_lists[i].append("--gpu-id")
            final_parameter_lists[i].append(str(i))

        # adding out directory to each task (if not exists)
        try:
            out_dir = final_parameter_dicts[i]["--out"]
        except:
            task_folder = os.path.dirname(args.run)
            out_dir = os.path.join(task_folder, final_parameter_dicts[i]["--plt_env"])
            # create folder if not exists
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            final_parameter_lists[i].append("--out")
            final_parameter_lists[i].append(out_dir)

    # deleting empty arguments
    for i in range(len(final_parameter_lists)):
        final_parameter_lists[i] = list(filter(lambda a: a != '', final_parameter_lists[i]))
        #final_parameter_lists[i].remove("")

    # running the runs
    for i in range(num_runs):
        print("running: ", "train.py", "with parameters", final_parameter_lists[i])
        subprocess.Popen([interpreter_path, "/nobackup/users/timfro/fixmatch/train.py"] + final_parameter_lists[i])








if __name__ == "__main__":
    main()



"""
datasets = ["cifar10", "cifar100", "mini_imagenet", "euro_sat", "plant_village"]
augmentations = ["randaugment", "ctaugment"]
use_method = {"cifar10": "wideresnet",
                "cifar100": "wideresnet",
                "mini_imagenet": "resnet",
                "euro_sat": "wideresnet",
                "plant_village": "wideresnet"}

skip_dataset = {"cifar10": True,
                "cifar100": True,
                "mini_imagenet": True,
                "euro_sat": True,
                "plant_village": False}


skip_augmentation = {"randaugment": False,
                     "ctaugment": True}  # TODO: implement ct augment

label_cifar_10 = [10, 40, 250, 4000]
label_cifar_100 = [100, 400, 2500, 10000]
label_mini_imagenet = [400, 2500, 4000, 10000]  # not tested in Fixmatch paper but Featmatch used same numbers as for cifar100
#label_euro_sat = [10, 40, 250, 4000]
label_euro_sat = [4000, 250]
label_plant_village = [7600]  #[38, 152, 950, 7600]  # 1, 4, 25, 200 labels per class (38 classes)
label_isic = [150, 600, 3750, 30000]  # 10, 40, 250, 2000 labels per class

label_by_dataset = {"cifar10": label_cifar_10,
                    "cifar100": label_cifar_100,
                    "mini_imagenet": label_mini_imagenet,
                    "euro_sat": label_euro_sat,
                    "plant_village": label_plant_village}

process_count = 0
for dataset in datasets:
    if skip_dataset[dataset]:
        continue
    for augmentation in augmentations:
        if skip_augmentation[augmentation]:
            continue
        for n_labels in label_by_dataset[dataset]:
            now = datetime.now() # current date and time
            date = now.strftime("%d%m%y")
            plt_env = dataset + '_' + augmentation + '_' + str(n_labels) + 'labels_' + date + 't'
            out_dir = '/mnt/CVAI/data/fixmatch/result/' + plt_env
            # TODO: choose gpu with lowest memory usage and wait if all are occupied
            subprocess.Popen([interpreter_path, "train.py"] + arguments[dataset] +
                             ["--num-labeled", str(n_labels),
                              "--plt_env", plt_env,
                              "--log_file", plt_env,
                              "--gpu-id", str(process_count % 4),
                              "--out", str(out_dir),
                              "--arch", use_method[dataset],
                              #"--nofixmatch",
                              ])
            process_count += 1
            
"""
