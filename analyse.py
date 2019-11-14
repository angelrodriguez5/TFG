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

from tkinter import Tk
from tkinter.filedialog import askopenfilename

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
    batch_size = 4
    # Number of frames to be skipped between samples
    frame_skip = 5000
    n_cpu = 0
    img_size = 416


def print_expert_timings(ax, videoName, framerate, expert=None):
    '''
    Print haemorrhage, lysis and coagulation times for a given video.
    If expert is passed, only the timings of one expert are printed, else all are printed.
    '''
    # Colors for haemorrhage, lysis and coagulation lines
    colors = ['r','g','k']

    # Data structure
    # Video : Stages (haemorrhage, lysis, coagulation)
    #   Stage : [values] (array of times of experts in miliseconds)
    data = {"DSC_1089.MOV":
                {"Haemorrhage": [2,0,2,1,1],
                 "Lysis":       [8,0,6,5,5],
                 "Coagulation": [42,0,7,24,18]},

            "DSC_1098.MOV":
                {"Haemorrhage": [8,9,4,3,3],
                 "Lysis":       [10,18,6,5,8],
                 "Coagulation": [16,74,12,21,15]},

            "DSC_1104.MOV":
                {"Haemorrhage": [1,4,2,1,1],
                 "Lysis":       [3,41,4,2,6],
                 "Coagulation": [20,81,5,8,12]},

            "DSC_1107.MOV":
                {"Haemorrhage": [2,10,2,2,1],
                 "Lysis":       [7,59,5,4,9],
                 "Coagulation": [24,91,9,14,13]},

            "DSC_1109.MOV":
                {"Haemorrhage": [1,2,1,1,1],
                 "Lysis":       [2,10,2,2,2],
                 "Coagulation": [20,34,4,12,6]}
    }

    # Check if video is in dictionary
    video = data.get(videoName, None)
    if video is None:
        print("Selected video %s has no expert annotations" % videoName)
        return

    # Iterate over stages
    for stage, color in zip(list(video.values()), colors):
        if expert is not None:
            # Print one expert's values
            # Can crash if expert index out of range
            value = stage[expert]
            # ms to frame
            frame = value * framerate
            ax.axvline(x=frame, color=color)
            pass
        else:
            # Print all values
            for value in stage:
                frame = value * framerate
                ax.axvline(x=frame, color=color)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="path to the video")
    # parser.add_argument("--video", type=str, default="Analyser/test/pos_DSC_1107.MOV", help="path to the video")
    kwargs = parser.parse_args()
    video_path = kwargs.video

    # If no video is passed as argument, open file explorer
    if not video_path:
        Tk().withdraw()
        # Supported file extensions
        videotypes = [("Video", "*.mp4 *.mov *.avi")]
        video_path = askopenfilename(title="Open a video",filetypes=videotypes)

    opt = Options()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    # Load weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(opt.weights))
    else:
        model.load_state_dict(torch.load(opt.weights, map_location='cpu'))

    model.eval()  # Set in evaluation mode

    # Dataset and dataloader
    dataset = VideoDataset(video_path, img_size=opt.img_size, frame_skip=opt.frame_skip)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # Parameters to print experts' timings
    framerate = dataset.get_framrate()
    videoName = os.path.basename(video_path)

    classes = load_classes(opt.classes)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    frames = []
    num_of_detections = []
    total_area = []

    for batch_i, (frame_nums, input_imgs) in enumerate(tqdm.tqdm(dataloader, desc="Analysing video")):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            outputs = model(input_imgs)
            outputs = non_max_suppression(outputs, opt.conf_thres, opt.nms_thres)

        # Save frame numbers for graphs
        frames.extend(frame_nums)
        # Extract data from frames
        for frame_detections in outputs:

            if frame_detections is not None:
                num_of_detections.append(len(frame_detections))

                # Calculate total area of bleeding
                area = np.zeros((opt.img_size, opt.img_size))
                for *coords, conf, cls_conf, cls_pred in frame_detections:
                    # Mark as 1 the areas detected
                    x1, y1, x2, y2 = [int(x) for x in coords]
                    area [y1:y2+1, x1:x2+1] = 1

                # the sum of all the elements in the array is the bleeding area in px^2
                total_area.append(area.sum())
            else:
                # If nothing was detected in a frame add zeros
                num_of_detections.append(0)
                total_area.append(0)

    # Create subplots and set title
    fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True)
    fig.suptitle(videoName)

    ax1.set_title("Area of bleeding")
    ax1.plot(frames, total_area)
    print_expert_timings(ax1, videoName, framerate)

    ax2.set_title("Total number of detections")
    ax2.plot(frames, num_of_detections)
    print_expert_timings(ax2, videoName, framerate)

    plt.show()
    # plt.savefig('/home/angel/Dropbox/DropboxTFG/%s.png' % args.videoName)
    # plt.close()