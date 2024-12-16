import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import tqdm
import argparse
import wandb

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

    # Make save dir
    os.makedirs(config["save_path"], exist_ok=True)

    # create dataset 
    train_dataset = GCSDataset(config["train_data_folder"])
    eval_dataset = GCSDataset(config["eval_data_folder"])

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

    # create optimizer 
    optimizer = optim.SGD(policy.parameters(), lr=0.01)

    # Set up training for number of epochs 
    for epoch in range(num_epochs):
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

            if i % config["save_freq"] == 0:
                torch.save(policy.state_dict(), f"{config['save_path']}/model_{i}.pth")
                print(f"Model saved to {config['save_path']}/model_{i}.pth")


        policy.eval()
        with torch.no_grad():
            tqdm_iter = tqdm.tqdm(
                eval_dataloader,
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
                map_array = map_array.to(device)
                times = times.to(device)
                trajs = trajs.to(device)

                pred_logits = policy(map_array, trajs)
                loss = loss_fn(pred_logits, trajs, policy.traj_embs.num_embeddings)
                print("Validation Loss: ", round(loss.item(), 4), round(torch.mean((torch.argmax(pred_logits[:, :-1], dim=-1) == traj_tensor[:, 1:]) * 1.).item(), 4))
                wandb.log({"val_loss": loss.item()})

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




