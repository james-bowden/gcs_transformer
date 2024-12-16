import glob 
import os
import shutil 
import numpy as np
from tqdm import tqdm
import pickle as pkl


max_traj_len = 512 
output_folder = "/hdd/optimization_data/tokenized_data_dir_512"
shutil.rmtree(output_folder, ignore_errors=True)
os.makedirs(output_folder, exist_ok=True)
files = glob.glob("/hdd/optimization_data/tokenized_data_dir/*.pkl")

for file in tqdm(files): 
    data = np.load(file, allow_pickle=True)
    map_array = data["map"]
    times = data["times"]
    trajs = data["trajs"]

    filtered_traj_count = 0
    filtered_times = []
    filtered_trajs = {}
    for i in range(len(trajs)):
        if trajs[i].shape[1] < max_traj_len - 2:
            filtered_times.append(times[i])
            filtered_trajs[filtered_traj_count] = trajs[i]
            filtered_traj_count += 1
    
    if len(filtered_trajs) < 3:
        continue

    output_dict = {"map": map_array, "times": filtered_times, "trajs": filtered_trajs}
    output_file = os.path.join(output_folder, os.path.basename(file))

    with open(output_file, "wb") as f:
        pkl.dump(output_dict, f)

