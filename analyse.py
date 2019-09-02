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
    frame_skip = 2500
    n_cpu = 0
    img_size = 416

    
if __name__ == "__main__":
    # TEMPORARY STATIC VIDEO PATH
    video_path = "Analyser/test/DSC_1104.MOV"
    # -----
    opt = Options()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    # Load weights
    model.load_state_dict(torch.load(opt.weights))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        VideoDataset(video_path, img_size=opt.img_size, frame_skip=opt.frame_skip),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.classes)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    frames = []
    num_of_detections = []
    total_area = []

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (frame_nums, input_imgs) in enumerate(tqdm.tqdm(dataloader, desc="Analysing video")):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            output = model(input_imgs)
            output = non_max_suppression(output, opt.conf_thres, opt.nms_thres)

        # Save frame numbers for graphs
        frames.extend(frame_nums)
        # Extract data from frames
        for detections in output:
            if detections is not None:
                num_of_detections.append(len(detections))
                # Calculate total area of bleeding
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    print ("First detection in frame")
                    print ("x1 : %s" % x1)
                    print ("x2 : %s" % x2)
                    print ("y1 : %s" % y1)
                    print ("y2 : %s" % y2)
                    print ("------------")
                    break
            else:
                # If nothing was detected in a frame add zeros
                num_of_detections.append(0)
                total_area.append(0)
                print("No detections in frame")

    plt.plot(frames, num_of_detections, 'r')
    plt.savefig('/home/angel/Dropbox/DropboxTFG/test.png')
    
    # # Bounding-box colors
    # cmap = plt.get_cmap("tab20b")
    # colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # print("\nSaving images:")
    # # Iterate through images and save plot of detections
    # for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    #     print("(%d) Image: '%s'" % (img_i, path))

    #     # Create plot
    #     img = np.array(Image.open(path))
    #     plt.figure()
    #     fig, ax = plt.subplots(1)
    #     ax.imshow(img)

    #     # Draw bounding boxes and labels of detections
    #     if detections is not None:
    #         # Rescale boxes to original image
    #         detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
    #         unique_labels = detections[:, -1].cpu().unique()
    #         n_cls_preds = len(unique_labels)
    #         bbox_colors = random.sample(colors, n_cls_preds)
    #         for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

    #             print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

    #             box_w = x2 - x1
    #             box_h = y2 - y1

    #             color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
    #             # Create a Rectangle patch
    #             bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor='b', facecolor="none")
    #             # Add the bbox to the plot
    #             ax.add_patch(bbox)
    #             '''
    #             # Add label
    #             plt.text(
    #                 x1,
    #                 y1,
    #                 s=classes[int(cls_pred)],
    #                 color="white",
    #                 verticalalignment="top",
    #                 bbox={"color": color, "pad": 0},
    #             )
    #             '''

    #     # Save generated image with detections
    #     plt.axis("off")
    #     plt.gca().xaxis.set_major_locator(NullLocator())
    #     plt.gca().yaxis.set_major_locator(NullLocator())
    #     filename = path.split("/")[-1].split(".")[0]
    #     plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
    #     plt.close()
        