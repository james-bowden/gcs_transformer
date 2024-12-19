import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import tqdm
import argparse
import wandb
import datetime
import glob 
import random 
import time
from IPython.core.display import display, HTML
from matplotlib import collections as mc
import matplotlib.colors as mcolors
from matplotlib import cm
import pickle as pkl
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
import torchtune
import torch.optim as optim
from torch.utils.data import DataLoader

from model import GCSTransformer
from custom_uav_env import *


DEBUG = True

def load_model(config, model_path):
    device = config["device"]
    model = GCSTransformer(num_time_bins = config["num_time_bins"],
                            num_space_bins = config["num_space_bins"],
                            d_model = config["d_model"],
                            nhead=config["num_heads"],
                            nlayers=config["num_layers"],
                            num_map_tokens=config["num_map_tokens"],
                            num_channels=config["num_channels"]).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def beam_search(config, model, map_array, traj, sos_token, eos_token, beam_width=2):
    beam = [(traj, 0.0, 0, False)]
    max_length = config["max_seq_len"]
    for _ in range(max_length):
        new_beam = []
        for seq, score, eos_count, done in beam:
            if done: 
                new_beam.append((seq, score, eos_count, done))
                continue
            # Get the next token probabilities
            logits = model(map_array, seq)
            probs = torch.softmax(logits, dim=-1)

            # Get top-k probabilities and indices
            topk_probs, topk_indices = torch.topk(probs, beam_width)

            # Expand the beam with top-k candidates
            for prob, index in zip(topk_probs[0][0], topk_indices[0][0]):
                if index == eos_token:
                    new_eos_count = eos_count + 1
                    if new_eos_count == 3:
                        # Beam is done 
                        new_done = True
                else:
                    new_eos_count = eos_count
                    new_done = False
                new_seq = torch.cat([seq, index.reshape(1,1)], axis=1)
                new_score = score + torch.log(prob)
                new_beam.append((new_seq, new_score, new_eos_count, new_done))

        # Sort the new beam by score and keep top-k
        beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]

    # Return the best sequence
    return beam[0][0]

def inference(config, model, map_array):

    device = config["device"]
    sos_token = config["num_space_bins"] + config["num_time_bins"]
    eos_token = config["num_space_bins"] + config["num_time_bins"] + 1
    max_seq_len = config["max_seq_len"]*3

    traj = torch.tensor([[sos_token]], dtype=torch.int64, device=device)
    beam = False
    if beam:
        traj = beam_search(config, model, map_array, traj, sos_token, eos_token, beam_width=2)
    else:
        # Initialize trajectory with start of sentence token
        eos_count = 0
        for _ in range(max_seq_len):

            # Get source mask
            logits = model(map_array, traj)
            
            next_token = torch.distributions.Categorical(logits=logits[:,-1,:]).sample()
            next_token = torch.tensor([[next_token]], device=device)

            # Concatenate previous input with predicted best word
            traj = torch.cat((traj, next_token), dim=1)

            # Stop if model predicts end of sentence
            if next_token.view(-1).item() == eos_token:
                eos_count += 1
                if eos_count == 3:
                    break
    traj = traj.squeeze(0)
    sos_token_inds = torch.argwhere(traj == sos_token).flatten()
    eos_token_inds = torch.argwhere(traj == eos_token).flatten()
    traj_1 = traj[sos_token_inds[0]+1:eos_token_inds[0]]
    traj_2 = traj[sos_token_inds[1]+1:eos_token_inds[1]]
    traj_3 = traj[sos_token_inds[2]+1:eos_token_inds[2]]

    return [traj_1, traj_2, traj_3]

def make_env(config, sample_file):

    env_shape = config["env_shape"]
    x = np.arange(1, env_shape[0]-1)
    y = np.arange(1, env_shape[1]-1)
    xx, yy = np.meshgrid(x, y)
    possible_goals = np.stack([xx, yy]).reshape(2, -1).T * CELL_SIZE

    seed, remainder = sample_file.split("/")[-1].split("_")
    seed = int(seed)
    goal_ind, _ = remainder.split(".")
    goal_ind = int(goal_ind)

    uav_env = UavEnvironment(seed=seed, environment_shape=env_shape, DEFAULT_GOAL=possible_goals[goal_ind])

    return uav_env, seed, possible_goals[goal_ind]

def check_collision(uav_env, traj):
    collision = 0
    total = 0
    regions, edges_between_regions = uav_env.compile()
    for t in np.linspace(traj.start_time(), traj.end_time(), 1000):
        freespace = False
        for region in regions:
            if region.PointInSet(traj.value(t), tol=1e-2):
                freespace = True
                break
        if not freespace:
            collision += 1
            break
    
    total += 1
    collision_rate = collision / total
    return collision_rate

def dist_to_goal(uav_env, traj: CompositeTrajectory, goal):

    # Get the last point in the trajectory
    x_end, y_end = traj.value(traj.end_time()).reshape(-1)[:-1]

    # Calculate distance to goal
    dist = np.sqrt((x_end - goal[0])**2 + (y_end - goal[1])**2)

    return dist

def time_to_goal(uav_env, traj):

    # Get time to goal 
    t = traj.end_time() - traj.start_time()

    return t

def plot_traj(uav_env, traj_list, goal, seed):
    fig, ax = plt.subplots(figsize=(13, 10))
    info_map = {
        Building.make_external_wall: (5, "ext wall"),
        Building.make_external_window_left: (3, "ext window L"),
        Building.make_external_window_right: (3, "ext window R"),
        Building.make_external_windows: (3, "ext window"),
        Building.make_external_door: (1, "ext door"),
        Building.make_internal_horizontal_wall_left: (3, "int window L"),
        Building.make_internal_horizontal_wall_right: (3, "int window R"),
        Building.make_internal_vertical_wall: (3, "int window U"),
        Building.make_internal_door: (1, "int door"),

    }

    for idx, wall_type in enumerate(uav_env.walls.keys()):
        if wall_type == Building.make_internal_no_wall:
            continue

        linewidth, label = info_map[wall_type]

        xs, ys, direcs = [], [], []
        wall_segments = []
        for outdoor_wall in uav_env.walls[wall_type]:
            x, y, direc = outdoor_wall
            xs.append(x)
            ys.append(y)
            direcs.append(direc)
            
            if direc in [Direction.LEFT, Direction.RIGHT]:
                wall_start, wall_end = (x, y + CELL_SIZE / 2), (x, y - CELL_SIZE / 2)
            else:
                wall_start, wall_end = (x + CELL_SIZE / 2, y), (x - CELL_SIZE / 2, y)
            wall_segments.append((wall_start, wall_end))

        color_name = list(mcolors.TABLEAU_COLORS)[idx]
        color = mcolors.TABLEAU_COLORS[color_name]
        wall_collection = mc.LineCollection(wall_segments, linewidths=linewidth, colors=color)
        ax.add_collection(wall_collection)
        ax.scatter(xs, ys, label=label, c=color)
        ax.set_aspect('equal', adjustable='box')
    labels = ["worst", "intermediate", "best", "GT"]
    # Plot goal 
    ax.scatter(goal[0], goal[1], c="red", label="goal")
    for traj_idx, traj in enumerate(traj_list):
        xs = []
        ys = []
        for t in np.linspace(0, traj.end_time(), 100):
            x, y = traj.value(t).reshape(-1)[:-1]
            xs.append(x)
            ys.append(y)

        xs = np.array(xs)
        ys = np.array(ys)

        # time = times[traj_idx]
        ax.plot(xs, ys, label=labels[traj_idx])

    ax.legend()
    ax.set_title(f"Environment w/ seed: {str(seed)} and goal: {str(goal)}")
    print("Saving plot...")
    plt.savefig(f"test_traj_{seed}_{goal}.png")

    # fig.colorbar(cm.ScalarMappable(norm=my_norm, cmap=my_cmap), orientation="vertical", label="Float value")

def detokenize_traj(config, traj, gt=False):
    if type(traj) == np.ndarray:
        traj = torch.tensor(traj, dtype=torch.int64).to(config["device"])
    bezier_curves = []
    num_space_bins = config["num_space_bins"]
    num_time_bins = config["num_time_bins"]
    space_min = config["space_min"]
    space_max = config["space_max"]
    time_min = config["time_min"]
    time_max = config["time_max"]
    env_shape = config["env_shape"]

    prev_time = 0
    prev_location = [0, 0, 2.]
    traj = traj.cpu().numpy()
    time_inds = np.argwhere(traj >= num_space_bins)
    time_inds = time_inds.flatten()
    traj_split = np.split(traj, time_inds)

    for bez_params in traj_split:
        if len(bez_params) < 2 or bez_params.shape[0] == 0:
            continue
        if bez_params[1:].shape[0] < 18:
            bez_params = np.concatenate([bez_params, np.full(18 - len(bez_params[1:]), 512)])
        elif bez_params[1:].shape[0] > 18:
            # first get rid of leading zeros
            bez_params_time = bez_params[0].reshape(1,1)
            bez_params_space = bez_params[1:].reshape(1,-1)

            non_zero_inds = np.argwhere(bez_params != 0).flatten()
            bez_params = np.concatenate([bez_params_time, bez_params_space[:,non_zero_inds[0]:]], axis=1)

            if bez_params.shape[1] > 19:
                bez_params = bez_params[:,:19]
            elif bez_params.shape[1] < 19:
                bez_params = np.concatenate([bez_params, np.full(19 - bez_params.shape[1],512).reshape(1,-1)], axis=1)
        bez_params = bez_params.flatten()
        assert len(bez_params) == 19, f"bez_params: {bez_params.shape}"
            
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
    
    return traj

def evaluate_sample(config, model, sample_file, viz=True):  
    results = {"best" : {"collision_rate" : 0, "dist_to_goal": 0, "time" : 0}, 
               "intermediate" :{"collision_rate" : 0, "dist_to_goal": 0, "time" : 0}, 
               "worst" : {"collision_rate" : 0, "dist_to_goal": 0, "time" : 0},
                "gt" : {"collision_rate" : 0, "dist_to_goal": 0, "time" : 0}}
    traj_type = ["worst", "intermediate", "best", "gt"]

    # Load map array and ground truth trajectory
    map_array = np.load(sample_file, allow_pickle=True)["map"].squeeze()
    map_array = torch.tensor(map_array, dtype=torch.int64).unsqueeze(0).to(config["device"])
    gt_traj = np.load(sample_file, allow_pickle=True)["trajs"][0].reshape(-1)

    # Perform inference
    print(f"Performing inference on {sample_file}...")
    attempts = 0
    while attempts < 5:
        try:
            try:
                trajs = inference(config, model, map_array)
                break
            except:
                print(f"Attempt {attempts} of 5 failed. Retrying...")
                attempts += 1
                continue
        except KeyboardInterrupt:
            break

    # Detokenize trajectory
    detokenized_trajs = []
    for traj in trajs:
        bezier_curves = detokenize_traj(config, traj)
        detokenized_trajs.append(bezier_curves)

    detokenized_gt_traj = detokenize_traj(config, gt_traj, True)
    detokenized_trajs.append(detokenized_gt_traj)

    # Make environment
    uav_env, seed, goal = make_env(config, sample_file)

    ## METRICS
    for name, traj in zip(traj_type, detokenized_trajs):

        # Collision_rate
        collision_rate = check_collision(uav_env, traj)
        results[name]["collision_rate"] = collision_rate

        # Time to goal 
        time = time_to_goal(uav_env, traj)
        results[name]["time"] = time

        # distance to goal
        dist = dist_to_goal(uav_env, traj, goal)
        results[name]["dist_to_goal"] = dist
    
    # Distance to goal
    if viz:
        # Plot trajectory
        plot_traj(uav_env, detokenized_trajs, goal, seed)
    
    return results

def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load model
    model = load_model(config, args.model_path)

    # Load map array
    if args.eval_path is None:
        sample_files = config["train_data_folder"]
        with open(sample_files, "r") as f:
            sample_files = [line.rstrip("\n") for line in f.readlines()]
        sample_file = random.choice(sample_files)
        sample_file = sample_file.replace(("/").join(sample_file.split("/")[:-1]), "/data")
    elif args.single:
        sample_files = args.eval_path
    else:
        sample_files_path = args.eval_path
        with open(sample_files_path, "r") as f:
            sample_files = [line.rstrip("\n") for line in f.readlines()]
        if args.num_samples != -1:
            sample_files = random.sample(sample_files, args.num_samples) 
            sample_files = [sample_file.replace(("/").join(sample_file.split("/")[:-1]), "/data") for sample_file in sample_files]


    overall_results = {"best" : {"collision_rate" : [], "dist_to_goal": [], "time" : []}, 
               "intermediate" :{"collision_rate" : [], "dist_to_goal": [], "time" : []}, 
               "worst" : {"collision_rate" : [], "dist_to_goal": [], "time" : []},
                "gt" : {"collision_rate" : [], "dist_to_goal": [], "time" : []}}
    for sample in tqdm.tqdm(sample_files):
        results = evaluate_sample(config, model, sample)
        if DEBUG:
            print(results)
        
        # Accumulate results
        for name in results.keys():
            for metric in results[name].keys():
                overall_results[name][metric].append(results[name][metric])
        
    # Average results
    print("OVERALL RESULTS: ")
    print(overall_results)
    avg_results = {}
    for name in overall_results:
        avg_results[name] = {}
        for metric in overall_results[name]:
            avg_results[name][metric] = np.mean(overall_results[name][metric])

    print("AVERAGED RESULTS: ")
    print(avg_results)
    # Save results
    with open(f"{args.eval_name}_full_results.pkl", "wb") as f:
        pkl.dump(overall_results, f)
    with open(f"{args.eval_name}_results.pkl", "wb") as f:
        pkl.dump(avg_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model_path", type=str, default="model.pth")
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to evaluate (-1 if use all samples)")
    parser.add_argument("--eval_name", type=str, default="eval")
    args = parser.parse_args()
    main(args)