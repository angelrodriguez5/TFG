from __future__ import division
from statistics import mean, stdev

from Yolo.models import *
from Yolo.utils.logger import *
from Yolo.utils.utils import *
from Yolo.utils.datasets import *
from Yolo.utils.parse_config import *
from Yolo.utils.exportResults import *
from Yolo.test import *

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data configuration name: (weights, testset)
    crossval = {"tttvx":("config/bestWeights/tttvx.pth", "data/crossvalidation/1-tttvx/positive_test.txt"),
                "xtttv":("config/bestWeights/xtttv.pth", "data/crossvalidation/2-xtttv/positive_test.txt"),
                "vxttt":("config/bestWeights/vxttt.pth", "data/crossvalidation/3-vxttt/positive_test.txt"),
                "tvxtt":("config/bestWeights/tvxtt.pth", "data/crossvalidation/4-tvxtt/positive_test.txt"),
                "ttvxt":("config/bestWeights/ttvxt.pth", "data/crossvalidation/5-ttvxt/positive_test.txt")}

    # Initiate model
    model = Darknet("config/customModelDef.cfg").to(device)
    model.apply(weights_init_normal)

    # Placeholder for the metrics of the different models
    data = []
    for name, (weights, test) in crossval.items():
        print("Processing " + name)
        # Load weights
        model.load_state_dict(torch.load(weights))
        # Get dataloader
        dataset = ListDataset(test, augment=False, multiscale=False)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

        model.eval()

        # Calculate metrics
        precision, recall, AP, f1, ap_class, loss = evaluate(
                model,
                path=test,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=416,
                batch_size=1,
            )
        # Save metrics
        data.append([precision,recall,f1])

    # Separate metrics
    prcs, rcls, f1s = zip(*data)

    # Calculate mean and standard deviation
    x = ["precision","recall","f1 score"]
    y = [mean(prcs), mean(rcls), mean(f1s)]
    e = [stdev(prcs), stdev(rcls), stdev(f1s)]

    # Plot mean as red squares and stved as blue lines
    plt.errorbar(x, y, e, fmt='none', zorder=0)
    plt.scatter(x,y, c='r', marker='s')
    plt.show()
    plt.savefig("../Dropbox/DropboxTFG/crossvalid.png")