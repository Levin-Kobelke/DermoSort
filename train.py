# Importing dependencies
from torchvision import datasets
import torch
from torch.utils.data import DataLoader
from typing import Any, Callable, Optional, Tuple
import torch
import torch.optim as optim
from argparse import ArgumentParser
import logging
from utils import EfficientNetb4, get_dataset, get_transforms
import torch.nn as nn


def main():
    parser = ArgumentParser(description="Train Classifier")
    pin_memory = True
    num_workers = 16
    batch_size = 16
    num_epochs = 1
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--pin_memory", type=bool)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--num_epochs", type=int)

    args = parser.parse_args()
    pin_memory = args.pin_memory
    num_workers = args.num_workers
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    print(f"{pin_memory=}{num_workers=}{batch_size=}{num_epochs=}")
    dataset = get_dataset(train=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=True,
    )
    logging.info("creating Dataloader")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"using {device}")
    model = EfficientNetb4()
    model.to(device)
    logging.info("creating model")
    optimizer = optim.Adam(model.parameters(), lr=0.00003)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=10, T_mult=10
    )
    loss_fn = nn.CrossEntropyLoss()
    print(f"{pin_memory=}{num_workers=}{batch_size=}{num_epochs=}{device=}")
    running_loss = 0
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            target = batch[1]
            batch = batch[0]
            optimizer.zero_grad()
            batch, target = batch.to(device), target.to(device)
            outputs = model(batch)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % 100 == 0:
                print(f"{step=} of {len(dataloader)/batch_size}")
                print(f"{epoch=} loss:{running_loss/100} {scheduler.get_lr()}")
                running_loss = 0
        scheduler.step()
    torch.save(model.state_dict(), "./dermoscopy_classifier_v1_20ep.pt")
    logging.info("saving model state dict to workdirectory")


if __name__ == "__main__":
    main()
