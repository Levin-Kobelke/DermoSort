import cv2
import albumentations
import timm
from torch.utils.data import Dataset
import torch
from typing import Any, Callable, Optional, Tuple
import os
import numpy as np
import pandas as pd
import torch.nn as nn


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


def get_transforms(image_size: int) -> Tuple[Callable, Callable]:
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

    datasets = ["d7p", "ham10000", "ph2", "isic_2020"]
    dataframes = []
    for dataset in datasets:
        csv_file = f"{root}/{dataset}_binaryclass"
        df_train = pd.read_csv(csv_file)
        datafolder = dataset
        dataroot = "/home/l049e/Data/"
        # local vs cluster
        # dataroot = "/dkfz/cluster/gpu/data/OE0612/l049e"
        data_dir = os.path.join(dataroot + datafolder)
        for i in range(len(df_train)):
            start, end = df_train["filepath"].iloc[i].split(".")
            df_train["filepath"].iloc[i] = data_dir + "/" + start + "_512." + end

        col_to_keep = ["filepath", "target"]
        df_train = df_train[col_to_keep]
        dataframes.append(df_train)

    df_train = pd.concat(dataframes)

    transforms_train, transforms_val = get_transforms(256)
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
