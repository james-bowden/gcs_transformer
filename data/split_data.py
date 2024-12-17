import glob
import os
import random
import argparse
import numpy as np
import shutil



def split_data(data_folder, train_frac=0.8, overwrite=False):

    data_files = glob.glob(os.path.join(data_folder, "*.pkl"))
    seeds = [int(os.path.basename(file).split("_")[0]) for file in data_files]
    seeds = list(set(seeds))
    random.shuffle(seeds)

    train_eval_split = np.array_split(seeds, [int(train_frac * len(seeds))])
    
    # Split the seeds into train, val, and test
    train_seeds = train_eval_split[0]
    eval_split = np.array_split(train_eval_split[1], [int(0.75 * len(train_eval_split[1]))])
    val_seeds = eval_split[0]
    test_seeds = eval_split[1]

    # Get the files for each split
    train_files = [file for file in data_files if int(os.path.basename(file).split("_")[0]) in train_seeds]
    random.shuffle(train_files)
    val_files = [file for file in data_files if int(os.path.basename(file).split("_")[0]) in val_seeds]
    random.shuffle(val_files)
    test_files = [file for file in data_files if int(os.path.basename(file).split("_")[0]) in test_seeds]
    random.shuffle(test_files)

    # Create the train, val, and test folders
    train_folder = os.path.join(data_folder, "train")
    val_folder = os.path.join(data_folder, "val")
    test_folder = os.path.join(data_folder, "test")

    # Write the files to the train, val, and test folders
    if os.path.exists(train_folder) and not overwrite:
        raise ValueError(f"Split already exists at {train_folder}")
    elif os.path.exists(train_folder) and overwrite:
        shutil.rmtree(train_folder)
        shutil.rmtree(val_folder)
        shutil.rmtree(test_folder)
        os.makedirs(train_folder)
        os.makedirs(val_folder)
        os.makedirs(test_folder)
    else:
        os.makedirs(train_folder)
        os.makedirs(val_folder)
        os.makedirs(test_folder)
    
    with open(os.path.join(train_folder, "train_files.txt"), "w") as f:
        f.write("\n".join(train_files))
    with open(os.path.join(val_folder, "val_files.txt"), "w") as f:
        f.write("\n".join(val_files))
    with open(os.path.join(test_folder, "test_files.txt"), "w") as f:
        f.write("\n".join(test_files))

    print("Split complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    
    split_data(args.data_folder, args.train_frac, args.overwrite)