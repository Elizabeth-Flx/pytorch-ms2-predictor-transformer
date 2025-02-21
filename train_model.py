import torch
import torch.optim as optim
import torch.nn as nn
from model.model import TransMS2Predictor
from model.dataset import ProteomeToolsDataset
from torch.utils.data import DataLoader

from dotenv import load_dotenv
import os
import wandb

load_dotenv()  # Load variables from .env
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(
    project="ms2_pytorch",
    entity='elizabeth-lochert-flx'
)


# check if cuda is available
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.device(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = TransMS2Predictor().to(device)

loss_cos = nn.CosineSimilarity(dim=2, eps=1e-6)
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 1024


# load data

paths = (
    # "/cmnfs/proj/prosit_astral/datasets/proteome_tools_dlomix_format_train.parquet",
    "/cmnfs/proj/prosit_astral/datasets/proteome_tools_dlomix_format_val.parquet",
    # "/cmnfs/proj/prosit_astral/datasets/proteome_tools_dlomix_format_test.parquet"
)

import pandas as pd

dataset = ProteomeToolsDataset(paths)

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Data loaded")


print("Start training")




# print number of batches
print(len(train_loader))

for epoch in range(10):
    print (f"Epoch {epoch+1}")
    print("+" + "-"*18 + "+")

    for i in range(len(train_loader)):

        batch = next(iter(train_loader))

        x_sequence = batch[0].to(device)
        x_metadata = batch[1].to(device)
        y = batch[2][:, None, :].to(device)

        # Forward pass
        y_pred = model(x_sequence, x_metadata)
        loss = loss_cos(y_pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.mean().backward()

        # Update weights
        optimizer.step()

        
        wandb.log({"loss": loss.mean().item()})

        # Print progress
        tick_numbers = (len(train_loader) // 20) + 1
        # print(tick_numbers)
        if i % tick_numbers == 0:
            print("=", end="")


    # print loss
    print("")
    print(f"Loss: {loss.mean().item()}")
    print("")
    print("")