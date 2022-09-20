import numpy as np
import subprocess
from config import interpreter_path

dataset = "plant_village"

arguments = ["--total-steps", "102400",
             "--arch",  "wideresnet",
             "--eval-step", "1024",
             "--dataset",  dataset,
             "--num-labeled", "250",
             "--nofixmatch",
             "--local_rank"]

lr_min = 1e-3
lr_max = 1
steps = 4
lr_range = [0.0001] #[0.0001, 0.001, 0.01, 0.1]#

process_count = 0
for lr in lr_range:
    print(f"starting training for lr = {lr}")
    subprocess.Popen([interpreter_path,

                      "train.py"] +
                     arguments +
                     ["--lr", str(lr),
                      "--plt_env", str(f"tuning_{dataset}_supervised_lr_{lr}"),
                      "--gpu-id", "0, 1",
                      #"--gpu-id", str(process_count),
                      ])
    process_count += 1
