import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import glob
import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


class GCSDataset(Dataset):

    def __init__(
        self,
        data_file: str,
        max_seq_len: int = 512,
        num_space_bins = 1024,
        num_time_bins = 512,
    ):
        self.num_intermediates = 3
        self.max_seq_len = max_seq_len
        self.sos_token = np.array([num_space_bins + num_time_bins])
        self.eos_token = np.array([num_space_bins + num_time_bins + 1])
        self.pad_token_id = num_space_bins + num_time_bins + 2
        # Get data files
        with open(os.path.join(data_file), "r") as f:
            self.data_files = [file.rstrip("\n") for file in f.readlines()]
        self._build_index()
        print("Number of samples: ", len(self.index))
    

    def _build_index(self):
        self.index = self.data_files
        # self.index = []
        # for file in self.data_files:
        #     sample = np.load(file, allow_pickle=True)
        #     map_array = sample["map"]
        #     times = sample["times"]
        #     trajs = sample["trajs"]
        #     traj_idx_3 = 0
        #     if not trajs[traj_idx_3].shape[1] < self.max_seq_len - 2:
        #         continue
        #     for traj_idx_2 in range(1, len(trajs)-1, 5):
        #         if not trajs[traj_idx_2].shape[1] < self.max_seq_len - 2:
        #             continue
        #         for traj_idx_1 in range(traj_idx_2, len(trajs), 5):
        #             if not trajs[traj_idx_1].shape[1] < self.max_seq_len - 2:
        #                 continue
                
        #             self.index.append((file, traj_idx_1, traj_idx_2, traj_idx_3))
        random.shuffle(self.index)

    def _add_tokens(self, traj):
        return np.concatenate([self.sos_token, traj, self.eos_token], axis=0)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        # OLD VERSION
        # # Load in data
        # # the first traj is the least optimal, randomly sampled from the rest of the trajs 
        # # the second traj is the second least optimal, randomly sampled from the rest of the trajs
        # # the last traj is the most optimal trajectory

        # file, traj_idx_1, traj_idx_2, traj_idx_3 = self.index[idx]
        # sample = np.load(file, allow_pickle=True)
        # map_array = sample["map"].squeeze()
        # time = np.array([sample["times"][traj_idx_1], sample["times"][traj_idx_2], sample["times"][traj_idx_3]])

        # NEW VERSION
        # Get data index
        traj_file = self.index[idx]
        sample = np.load(traj_file, allow_pickle=True)
        map_array = sample["map"].squeeze()
        times = sample["times"]
        trajs = sample["trajs"]

        traj_idx_3 = 0 # most optimal
        try:
            traj_idx_2 = random.choice(range(1, len(trajs)-1))
            traj_idx_1 = random.choice(range(traj_idx_2, len(trajs)))
        except:
            traj_idx_2 = 1
            traj_idx_1 = 2
            breakpoint()

        time = np.array([times[traj_idx_1], times[traj_idx_2], times[traj_idx_3]])

        traj_1 = self._add_tokens(sample["trajs"][traj_idx_1].squeeze())
        traj_2 = self._add_tokens(sample["trajs"][traj_idx_2].squeeze())
        traj_3 = self._add_tokens(sample["trajs"][traj_idx_3].squeeze())

        if len(traj_1) > self.max_seq_len or len(traj_2) > self.max_seq_len or len(traj_3) > self.max_seq_len:
            breakpoint()
        
        total_len = traj_1.shape[0] + traj_2.shape[0] + traj_3.shape[0]
        pad_len = self.max_seq_len*self.num_intermediates - total_len
        traj = np.concatenate([traj_1, traj_2, traj_3, torch.full((pad_len,), self.pad_token_id)], axis=0)
        return (torch.as_tensor(map_array, dtype=torch.int64), 
                torch.as_tensor(time, dtype=torch.int64), 
                torch.as_tensor(traj, dtype=torch.int64))









