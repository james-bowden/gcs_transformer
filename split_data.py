import glob
import os
import random
import argparse
import numpy as np
import shutil



def split_data(data_folder, train_frac=0.8, overwrite=False):

    data_files = glob.glob(os.path.join(data_folder, "*.pkl"))
    random.shuffle(data_files)

    train_eval_split = np.array_split(data_files, [int(train_frac * len(data_files))])
    train_files = train_eval_split[0]
    eval_files = train_eval_split[1]

    train_folder = os.path.join(data_folder, "train")
    eval_folder = os.path.join(data_folder, "eval")

    if os.path.exists(train_folder) and not overwrite:
        raise ValueError(f"Split already exists at {train_folder}")
    elif os.path.exists(train_folder) and overwrite:
        shutil.rmtree(train_folder)
        shutil.rmtree(eval_folder)
        os.makedirs(train_folder)
        os.makedirs(eval_folder)
    else:
        os.makedirs(train_folder)
        os.makedirs(eval_folder)
    
    with open(os.path.join(train_folder, "train_files.txt"), "w") as f:
        f.write("\n".join(train_files))
    with open(os.path.join(eval_folder, "eval_files.txt"), "w") as f:
        f.write("\n".join(eval_files))

    print("Split complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    
    split_data(args.data_folder, args.train_frac, args.overwrite)