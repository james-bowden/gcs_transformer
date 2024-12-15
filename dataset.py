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
        data_folder: str,
    ):
    
    self.data_folder = data_folder
    self.data_index = []

    # Get data files
    self.data_files = glob.glob(os.path.join(self.data_folder, "*.npy"))


    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):

        # Load in data
        sample = np.load(self.data_files[idx])
        map_array = sample["map_array"]
        times = sample["times"]
        trajs = sample["trajs"]

        return (torch.as_tensor(map_array, dtype=torch.float32), 
                torch.as_tensor(times, dtype=torch.float32), 
                torch.as_tensor(trajs, dtype=torch.float32))









