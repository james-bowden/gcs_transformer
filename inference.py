import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import tqdm
import argparse
import wandb
import datetime

import torch
import torch.nn.functional as F
from torch import nn
import torchtune
import torch.optim as optim
from torch.utils.data import DataLoader

from model import GCSTransformer
from custom_uav_env import *

def load_model(config, model_path):
    device = config["device"]
    model = GCSTransformer(num_time_bins = config["num_time_bins"],
                           num_space_bins = config["num_space_bins"],
                           d_model = config["d_model"],
                           nhead=config["num_heads"],
                           nlayers=config["num_layers"],
                           num_map_tokens=config["num_map_tokens"],
                           num_intermediates=config["num_intermediates"],
                           dropout=config["dropout"],
                           max_seq_len=config["max_seq_len"],
                           pad_token_id=config["pad_token_id"],
                           sos_token=config["sos_token"],
                           eos_token=config["eos_token"],
                           device=device).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def inference(config, model, map_array):

    device = config["device"]
    sos_token = config["num_space_bins"] + config["num_time_bins"]
    eos_token = config["num_space_bins"] + config["num_time_bins"] + 1
    max_seq_len = config["max_seq_len"]*3

    # Initialize trajectory with start of sentence token
    traj = torch.tensor([[sos_token]], dtype=torch.int64, device=device)
    eos_count = 0
    for _ in range(max_seq_len):

        # Get source mask
        logits = model.predict(map_array, traj)
        
        next_token = logits.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_token = torch.tensor([[next_token]], device=device)

        # Concatenate previous input with predicted best word
        traj = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token:
            eos_count += 1
            if eos_count == 3:
                break

    return traj

def detokenize_traj(config, traj):
    
    bezier_curves = []
    num_space_bins = config["num_space_bins"]
    num_time_bins = config["num_time_bins"]
    space_min = config["space_min"]
    space_max = config["space_max"]
    time_min = config["time_min"]
    time_max = config["time_max"]

    prev_time = 0
    prev_location = np.array([0, 0, 2.])

    for bez_params in traj.reshape(-1, 19):
            delta_time = (bez_params[0] - num_space_bins) / num_time_bins * (time_max - time_min) + time_min
            start_time = prev_time
            end_time = prev_time + delta_time

            if delta_time < 1e-6:
                continue

            ctrl_pts = [prev_location]
            disc_diffs = bez_params[1:].reshape(-1, 3)
            for disc_diff in disc_diffs:
                diff = disc_diff / num_space_bins * (space_max - space_min) + space_min
                ctrl_pts.append(ctrl_pts[-1] + diff)

            prev_time = end_time
            prev_location = ctrl_pts[-1]

            ctrl_pts = np.array(ctrl_pts).T

            bc = BezierCurve(start_time, end_time, ctrl_pts)
            bezier_curves.append(bc)

        traj = CompositeTrajectory(bezier_curves)

def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load model
    model = load_model(config, args.model_path)

    # Load map array
    if args.sample_path is None:
        data_path = os.path.join(config["test_folder"], "test")
        sample_files = glob.glob(os.path.join(data_path, "*.pkl"))
        sample_file = random.choice(sample_files)
    else:
        sample_file = args.sample_path

    map_array = np.load(args.sample_path, allow_pickle=True)["map_array"].squeeze()
    map_array = torch.tensor(map_array, dtype=torch.int64).unsqueeze(0).to(config["device"])

    # Perform inference
    traj = inference(config, model, map_array)

    # Detokenize trajectory
    bezier_curves = detokenize_traj(config, traj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model_path", type=str, default="model.pth")
    parser.add_argument("--sample_path", type=str, default=None)
    args = parser.parse_args()
    main(args)