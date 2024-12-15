import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import glob

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

class GCSDataset(Dataset):

    def __init__(
        self,
        data_file: str,
    ):

        # Get data files
        with open(os.path.join(data_file), "r") as f:
            self.data_files = [file.rstrip("\n") for file in f.readlines()]
        print("Number of data files: ", len(self.data_files))
        self._build_index()
        print("Number of samples: ", len(self.index))
    
    def _build_index(self):
        self.index = []
        for file in self.data_files:
            sample = np.load(file, allow_pickle=True)
            map_array = sample["map"]
            times = sample["times"]
            trajs = sample["trajs"]
            for traj_idx in range(len(trajs)):
                self.index.append((file, traj_idx))


    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        # Load in data
        file, traj_idx = self.index[idx]
        sample = np.load(file, allow_pickle=True)
        map_array = sample["map"]
        times = sample["times"]
        trajs = sample["trajs"][traj_idx]
        breakpoint()
        print("map_array: ", map_array.shape)
        print("times: ", times.shape)
        print("trajs: ", trajs.shape)

        return (torch.as_tensor(map_array, dtype=torch.float32), 
                torch.as_tensor(times, dtype=torch.float32), 
                torch.as_tensor(trajs, dtype=torch.float32))









