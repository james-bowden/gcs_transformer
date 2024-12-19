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
from data.dataset import GCSDataset


def loss_fn(pred_logits, labels, vocab_size):
    shifted_pred_logits = pred_logits[:, :-1].reshape(-1, vocab_size)
    shifted_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shifted_pred_logits, shifted_labels)

def train(config, args):
    device = config["device"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]

    # Make save dir for current run
    os.makedirs(config["save_path"], exist_ok=True)
    curr_save_path = "gcs_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(os.path.join(config["save_path"], curr_save_path))

    # create dataset 
    train_dataset = GCSDataset(config["train_data_folder"])
    eval_dataset = GCSDataset(config["val_data_folder"])

    # create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    print("Num batches: ", len(train_dataloader))

    # create model 
    policy = GCSTransformer(num_time_bins = config["num_time_bins"],
                            num_space_bins = config["num_space_bins"],
                            d_model = config["d_model"],
                            nhead=config["num_heads"],
                            nlayers=config["num_layers"],
                            num_map_tokens=config["num_map_tokens"],
                            num_channels=config["num_channels"]).to(device)
    if config["load_model"]:
        print("Loading model from: ", config["load_model"])
        policy.load_state_dict(torch.load(config["load_model"]))
        curr_epoch = int(config["load_model"].split("_")[-2])
    else:
        curr_epoch = 0
    
    # create optimizer 
    optimizer = optim.SGD(policy.parameters(), lr=0.01)

    # Set up training for number of epochs 
    for epoch in range(curr_epoch, num_epochs):
        print("Epoch: ", epoch)

        # Train model
        policy.train()
        num_batches = len(train_dataloader)
        tqdm_iter = tqdm.tqdm(
            train_dataloader,
            disable=False,
            dynamic_ncols=True,
            desc=f"Training epoch {epoch}",
        )
        for i, data in enumerate(tqdm_iter):
            (
                map_array,
                times,
                trajs,
            ) = data

            map_tensor = map_array.to(device)
            times_tensor = times.to(device)
            traj_tensor = trajs.to(device)

            optimizer.zero_grad()
            pred_logits = policy(map_tensor, traj_tensor)
            loss = loss_fn(pred_logits, traj_tensor, policy.traj_embs.num_embeddings)
            loss.backward()
            optimizer.step()
            print("Training Loss: ", round(loss.item(), 4), round(torch.mean((torch.argmax(pred_logits[:, :-1], dim=-1) == traj_tensor[:, 1:]) * 1.).item(), 4))
            wandb.log({"train_loss": loss.item()})
            wandb.log({"train_acc": torch.mean((torch.argmax(pred_logits[:, :-1], dim=-1) == traj_tensor[:, 1:]) * 1.).item()})

            if i % config["save_freq"] == 0:
                torch.save(policy.state_dict(), f"{os.path.join(config['save_path'], curr_save_path)}/model_{epoch}_{i}.pth")
                print(f"Model saved to {config['save_path']}/model_{epoch}_{i}.pth")


        policy.eval()
        with torch.no_grad():
            tqdm_iter = tqdm.tqdm(
                eval_dataloader,
                disable=False,
                dynamic_ncols=True,
                desc=f"Training epoch {epoch}",
            )
            avg_loss = []
            avg_acc = []
            for i, data in enumerate(tqdm_iter):
                (
                    map_array,
                    times,
                    trajs,
                ) = data
                map_array = map_array.to(device)
                times_tensor = times.to(device)
                traj_tensor = trajs.to(device)

                pred_logits = policy(map_array, traj_tensor)
                loss = loss_fn(pred_logits, traj_tensor, policy.traj_embs.num_embeddings)
                avg_loss.append(loss.item())
                avg_acc.append(torch.mean((torch.argmax(pred_logits[:, :-1], dim=-1) == traj_tensor[:, 1:]) * 1.).item())
                print("Validation Loss: ", round(loss.item(), 4), round(torch.mean((torch.argmax(pred_logits[:, :-1], dim=-1) == traj_tensor[:, 1:]) * 1.).item(), 4))
            wandb.log({"val_loss": np.mean(avg_loss)})
            wandb.log({"val_acc": np.mean(avg_acc)})

    wandb.log({})
    print()

def main(args):
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)
    if config["use_wandb"]:
        wandb.login()
        wandb.init(project="gcs_transformer",
                   entity="catglossop",)
        wandb.save(args.config, policy="now")

    print("Starting training...")
    train(config, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, default="config/default.yaml")
    args = parser.parse_args()
    main(args)




