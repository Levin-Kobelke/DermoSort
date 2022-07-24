from time import sleep
from typing import Any, Callable, Optional, Tuple
import os
import imghdr
from utils import EfficientNetb4
import torch
import cv2
import numpy as np
import torch.nn as nn
from utils import get_transforms
import shutil
from argparse import ArgumentParser


def search(folder: str) -> list[str]:
    """
    Searches for items in a folder and returns the filepath
        Parameters:
            folder (str): folder to search for images
        Returns:
            List of filepaths to images in the folder
    """
    items = os.listdir(folder)
    imgpaths = []
    for item in items:
        path = folder + "/" + item
        valid = ["bmp", "png", "jpeg"]
        try:
            if imghdr.what(path) in valid:
                imgpaths.append(path)
        except:
            pass
    return imgpaths


class Classifier:
    def __init__(self) -> None:
        self.model = EfficientNetb4()
        self.model.load_state_dict(torch.load("dermoscopy_classifier_v1_20ep.pt"))
        self.model.eval()
        _, self.transform_test = get_transforms(256)

    def classify(self, file: str) -> int:
        """
        Takes a filepath and returns the class to sort it into
            Parameters:
                file (str): Path to img
            Returns:
                int from 0-1
                    0: Benign high confidence
                    1: Possibly malignant

        """
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = self.transform_test(image=image)
        image = res["image"].astype(np.float32)

        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        data = torch.tensor(image).float()

        out = self.model(data)
        threshold = 0.01
        softmaxLayer = nn.Softmax()
        softmax_output = softmaxLayer(torch.as_tensor(out))
        predicted = np.where(softmax_output[:, 1] > threshold, 1, 0)
        return np.squeeze(predicted)


def sort(rootfolder: str, file: str, outclass: int):
    """
    Takes a filepath and a Class and sorts the file into the
    correct folder
        Parameters:
            file (str): Path to img
            outclass (int): Class of the img
    """
    splitted = file.split("/")
    if outclass == 1:
        target_path = os.path.join(rootfolder, "danger", splitted[-1])
    elif outclass == 0:
        target_path = os.path.join(rootfolder, "trusted", splitted[-1])

    shutil.move(file, target_path)


def main(rootfolder: Optional[str]):
    parser = ArgumentParser(description="Train Classifier")
    parser.add_argument("--root_folder", type=str)
    parser.add_argument("--loop", type=bool)

    args = parser.parse_args()
    classifier = Classifier()
    loop = args.loop
    if rootfolder is None:
        rootfolder = args.root_folder
    path_danger = os.path.join(rootfolder, "danger")
    path_trusted = os.path.join(rootfolder, "trusted")
    if not os.path.exists(path_danger):
        os.mkdir(path_danger)
    if not os.path.exists(path_trusted):
        os.mkdir(path_trusted)
    # while True:
    fileList: list[str] = search(rootfolder)
    for file in fileList:
        outclass: int = classifier.classify(file)
        sort(rootfolder, file, outclass)
    # sleep(10)
    if loop == True:
        while True:
            fileList: list[str] = search(rootfolder)
            for file in fileList:
                outclass: int = classifier.classify(file)
                sort(rootfolder, file, outclass)
            sleep(10)


if __name__ == "__main__":
    main(rootfolder=None)
