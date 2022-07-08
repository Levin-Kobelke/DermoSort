#Importing dependencies
import numpy as np
import pandas as pd 
from torchvision import datasets
from torch.utils.data import Dataset
import torch
import cv2
import albumentations
from torch.utils.data import DataLoader
import os
from typing import Any, Callable, Optional, Tuple
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from argparse import ArgumentParser
import logging
import timm


class BasicDataset(Dataset):
    def __init__(
        self,
        csv: pd.core.frame.DataFrame,
        train: bool,
        transform: Optional[callable] = None,
    ):

        self.csv = csv.reset_index(drop=True)
        self.train = train
        self.transform = transform
        self.train_df = self.csv.sample(frac=0.8, random_state=200)
        self.test_df = self.csv.drop(self.train_df.index)
        if self.train:
            self.csv = self.train_df
        elif not self.train:
            self.csv = self.test_df
        self.targets = self.csv.target
        self.imgs = self.csv["filepath"]
        self.samples = self.imgs

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res["image"].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        data = torch.tensor(image).float()

        return data, torch.tensor(self.csv.iloc[index].target).long()

def get_transforms(image_size: int)-> Tuple[Callable, Callable]:
    """
    Returns a composite of albumination transforms to apply to images. First one is strong for training and second one is weak for validation/training
        Parameters:
            image_size (int): The image size to crop to (always cropped to square)
        Returns:
            transforms_train, transforms_val (Callable): Composite of transforms

    Augmentation strategy adapted from: Identifying Melanoma Images using EfficientNet Ensemble: Winning Solution to the SIIM-ISIC Melanoma Classification Challenge 

    """


    transforms_train = albumentations.Compose(
        [
            albumentations.Transpose(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightness(limit=0.2, p=0.75),
            albumentations.RandomContrast(limit=0.2, p=0.75),
            albumentations.OneOf(
                [
                    albumentations.MotionBlur(blur_limit=5),
                    albumentations.MedianBlur(blur_limit=5),
                    albumentations.GaussianBlur(blur_limit=5),
                    albumentations.GaussNoise(var_limit=(5.0, 30.0)),
                ],
                p=0.7,
            ),
            albumentations.OneOf(
                [
                    albumentations.OpticalDistortion(distort_limit=1.0),
                    albumentations.GridDistortion(num_steps=5, distort_limit=1.0),
                    albumentations.ElasticTransform(alpha=3),
                ],
                p=0.7,
            ),
            albumentations.CLAHE(clip_limit=4.0, p=0.7),
            albumentations.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5
            ),
            albumentations.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85
            ),
            albumentations.Resize(image_size, image_size),
            albumentations.Cutout(
                max_h_size=int(image_size * 0.375),
                max_w_size=int(image_size * 0.375),
                num_holes=1,
                p=0.7,
            ),
            albumentations.Normalize(),
        ]
    )

    transforms_val = albumentations.Compose(
        [albumentations.Resize(image_size, image_size), albumentations.Normalize()]
    )

    return transforms_train, transforms_val



def get_dataset(train: bool) -> object:
    """
    Returns instance of dataset
        Parameters:
            train (bool): true for trainloader, dalse for val/testloader
        Returns:
            instance of dataset
    """

    root = os.getcwd()


    datasets = ["d7p","ham10000","ph2","isic_2020"]
    dataframes = []
    for dataset in datasets:
        csv_file = f"{root}/{dataset}_binaryclass"
        df_train = pd.read_csv(csv_file)
        datafolder = dataset
        dataroot = "/home/l049e/Data/"
        data_dir = os.path.join(dataroot+datafolder)
        df_train["filepath"] = data_dir + "/" + df_train["filepath"]
        col_to_keep = ["filepath", "target"]
        df_train = df_train[col_to_keep]
        dataframes.append(df_train)
    
    df_train = pd.concat(dataframes)

    transforms_train, transforms_val = get_transforms(512)
    if train:
        transforms = transforms_train
    else:
        transforms = transforms_val
    pass_kwargs = {"csv": df_train, "train": train, "transform": transforms}
    return BasicDataset(**pass_kwargs)


class EfficientNetb4(nn.Module):
    def __init__(self):
        super(EfficientNetb4, self).__init__()
        num_classes = 2
        self.model = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            num_classes=num_classes,
            drop_rate=0,
        )
        self.model.reset_classifier(num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

from pickletools import optimize
from sched import scheduler
from zmq import device
from torch.utils.data.distributed import DistributedSampler
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

from torch.utils.data.distributed import DistributedSampler
def prepare(rank, world_size, batch_size=16, pin_memory=False, num_workers=0):
    dataset = get_dataset(train=True)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader
def main(rank, world_size):
    # setup the process groups
    setup(rank, world_size)
    logging.info("Setting up")
    # prepare the dataloader
    dataloader = prepare(rank, world_size)
    logging.info("creating Dataloader")

    model = EfficientNetb4()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    logging.info("creating DDP model")
    num_epochs = 30
    optimizer = optim.Adam(model.parameters(),lr = 0.00003 )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10,T_mult=10)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        dist.barrier()
        for step, batch in enumerate(dataloader):
            outputs = model(batch[0])
            labels = batch[1]
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        print("Epoch")
        scheduler.step()
    torch.save(model.state_dict(), "./dermoscopy_classifier.pt")
    logging.info("saving model state dict to workdirectory")
    cleanup()

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = 1    
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size
    )
