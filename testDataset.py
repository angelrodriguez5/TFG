from __future__ import division

import os
import sys
import time
import datetime
import argparse

from Yolo.models import *
from Yolo.utils.utils import *
from Yolo.utils.datasets import *

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

class Options(object):
    '''
    Static Configuration options
    '''
    # Model
    model_def = "Analyser/model/model.cfg"
    classes = "Analyser/model/classes.names"
    weights = "Analyser/model/weights.pth"

    # Detection paramenters
    conf_thres = 0.8
    nms_thres = 0.4

    # How many frames to be processed at the same time
    batch_size = 1
    # Number of frames to be skipped between samples
    frame_skip = 240
    n_cpu = 0
    img_size = 416

    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--video", type=str, default="Analyser/test/neg_DSC_1106.MOV", help="path to the video")
    # parser.add_argument("--video", type=str, default="Analyser/test/pos_DSC_1107.MOV", help="path to the video")
    # kwargs = parser.parse_args()
    # video_path = kwargs.video
    # print("Video: " + video_path)

    opt = Options()

    # Video dataset
    video_path = "Analyser/test/pos_DSC_1107.MOV"
    videoloader = DataLoader(
        VideoDataset(video_path, img_size=opt.img_size, frame_skip=opt.frame_skip),
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # Folder dataset
    folder_path = "Analyser/test/folder"
    folderloader = DataLoader(
        ImageFolder(folder_path, img_size=opt.img_size),
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # List dataset
    list_path = "Analyser/test/list.txt"
    dataset = ListDataset(list_path, img_size=opt.img_size, augment=False, multiscale=False, unlabeled=True)
    listloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu,
        collate_fn=dataset.collate_fn
    )

    fig, ax = plt.subplots()

    for batch_i, ((path, img2), (path, img3, targets))  in enumerate(zip(folderloader, listloader)):

        # i1 = img1[0].permute(1, 2, 0).numpy()
        # plt.subplot(311)
        # plt.imshow(i1)

        i2 = img2[0].permute(1, 2, 0).numpy()
        plt.subplot(312)
        plt.title("folder loader")
        plt.imshow(i2)

        i3 = img3[0].permute(1, 2, 0).numpy()
        plt.subplot(313)
        plt.title("list loader")
        plt.imshow(i3)
        
        fig.canvas.draw()
        plt.waitforbuttonpress()
