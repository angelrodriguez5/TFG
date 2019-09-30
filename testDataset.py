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
    frame_skip = 200
    n_cpu = 0
    img_size = 416

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--video", type=str, default="Analyser/test/neg_DSC_1106.MOV", help="path to the video")
    parser.add_argument("--video", type=str, default="Analyser/test/pos_DSC_1107.MOV", help="path to the video")
    kwargs = parser.parse_args()
    video_path = kwargs.video
    print("Video: " + video_path)

    opt = Options()

    dataloader = DataLoader(
        VideoDataset(video_path, img_size=opt.img_size, frame_skip=opt.frame_skip),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    frames = []
    num_of_detections = []
    total_area = []

    for batch_i, (frame_nums, input_imgs) in enumerate(tqdm.tqdm(dataloader, desc="Analysing video")):

        for (frame, img) in list(zip(frame_nums, input_imgs)):
            
            # Transform image tensor to PIL  image
            img = img.permute(1, 2, 0).numpy()

            fig, ax = plt.subplots()
            ax.imshow(img)

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig(f"{frame}.png", bbox_inches="tight", pad_inches=0.0)
            plt.close()
