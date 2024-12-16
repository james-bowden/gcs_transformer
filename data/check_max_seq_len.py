import glob
import os
import numpy as np

data_files = glob.glob('/nfs/kun2/users/cglossop/tokenized_data_dir/*.pkl')
max_seq_len = 0
for file in data_files:
    data = np.load(file, allow_pickle=True)

    trajs = data['trajs']
    curr_traj_max_len = max([trajs[i].shape[1] for i in range(len(trajs))])
    max_seq_len = max(max_seq_len, curr_traj_max_len)


print("Final max seq len: ", max_seq_len)