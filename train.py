import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import tqdm
import argparse

import torch
import torch.nn.functional as F
from torch import nn
import torchtune
import torch.optim as optim
from torch.utils.data import DataLoader

from gcs_transformer.model import GCSTransformer
from gcs_transformer.dataset import GCSDataset


def loss_fn(pred_logits, labels, vocab_size):
    shifted_pred_logits = pred_logits[:, :-1].reshape(-1, vocab_size)
    shifted_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shifted_pred_logits, shifted_labels)

def train(config, args):
    device = args.device
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]

    # create dataset 
    train_dataset = GCSDataset(config["train_data_folder"])
    test_dataset = GCSDataset(config["test_data_folder"])

    # create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    # create model 
    policy = GCSTransformer(num_time_bins = config["num_time_bins"],
                            num_space_bins = config["num_space_bins"],
                            d_model = config["d_model"],
                            nhead=config["num_heads"],
                            nlayers=config["num_layers"],
                            num_map_tokens=config["num_map_tokens"],
                            num_channels=config["num_channels"],
                            device=config["device"]).to(device)

    # create optimizer 
    optimizer = optim.SGD(policy.parameters(), lr=0.01)

    # Set up training for number of epochs 
    for epoch in range(self.num_epochs):
        print("Epoch: ", epoch)

        # Train model
        policy.train()
        num_batches = len(train_dataloader)
        tqdm_iter = tqdm.tqdm(
            train_dataloader,
            disable=not use_tqdm,
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

            optimizer.zero_grad()
            pred_logits = policy(map_tensor, traj_tensor)
            loss = loss_fn(pred_logits, traj_tensor, policy.traj_embs.num_embeddings)
            loss.backward()
            optimizer.step()
            print("Training Loss: ", round(loss.item(), 4), round(torch.mean((torch.argmax(pred_logits[:, :-1], dim=-1) == traj_tensor[:, 1:]) * 1.).item(), 4))

        # Evaluate model
        if i % config["eval_freq"] == 0:

            policy.eval()
            with torch.no_grad():
                num_batches = len(train_dataloader)
                tqdm_iter = tqdm.tqdm(
                    test_dataloader,
                    disable=not use_tqdm,
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

        print(f"Epoch {epoch}, Loss: {total_loss}")

def main(args):
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)

    print("Starting training...")
    train(config, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()
    main(args)




