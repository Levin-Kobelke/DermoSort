from train import EfficientNetb4
from train import get_dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn as nn


def main():
    model = EfficientNetb4()
    model.load_state_dict(torch.load("dermoscopy_classifier_v1_20ep.pt"))
    model.eval()
    dataset = get_dataset(train=False)
    dataloader = DataLoader(
        dataset=dataset, batch_size=16, shuffle=False, num_workers=20
    )
    output = np.empty((0, 2))
    labels = np.empty((0, 1))
    for images, label in dataloader:
        out = model(images)
        output = np.append(output, out.data)
        labels = np.append(labels, label)

    output = np.resize(output, (int(len(output) / 2), 2))
    threshold = 0.01
    softmaxLayer = nn.Softmax()
    softmax_output = softmaxLayer(torch.as_tensor(output))
    predicted = np.where(softmax_output[:, 1] > threshold, 1, 0)
    y_score = softmax_output[:, 1]
    y_true = labels
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    print(f"{auc=}")
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    confMat = ConfusionMatrixDisplay.from_predictions(labels, predicted)
    confMat.figure_.savefig("confMat.png")


if __name__ == "__main__":
    main()
