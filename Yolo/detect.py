from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.CustomDatasetExporter import Mark

import os
import sys
import time
import datetime
import argparse
import math

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def isCorrectDetection(detected, target):
    threshold = 25
    # Distance between centers 
    dx, dy = detected.get_center()
    tx, ty = target.get_center()

    distance = math.sqrt( ((dx-tx)**2)+((dy-ty)**2) )

    return distance <= threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/test_img", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/customModelDef.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="currentWeights.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        imgDim = img.shape[:2]
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # If .txt file exists, load existing marks
        targetMarks = []
        fileName = path.replace("test_img", "test_tag").replace(".png", ".txt")

        if (os.path.isfile(fileName)):
            f = open(fileName, 'r')
            for line in f:
                # Cast array of string to floats
                array = [float(x) for x in line.split()]
                # Cast class number to int
                array[0] = int(array[0])
                # Populate target list with marks in the file
                mark = Mark.buildFromNorm(array, imgDim)
                targetMarks.append(mark)

        # Draw bounding boxes and labels of detections
        detectedMarks = []
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, imgDim)

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                
                classNum = int(cls_pred)
                cx = math.ceil((x1 + x2) / 2)
                cy = math.ceil((y1 + y2) / 2)
                w = x2 - x1
                h = y2 - y1
                # Populate detection list with marks
                mark = Mark(classNum,(cx,cy), (w,h))
                detectedMarks.append(mark)

        # Filter marks and set colors acordingly for display
        C_TP = 'g'
        C_FP = 'r'
        C_FN = 'b'
        displayMarks = []
        # confusion count variables
        TP = 0
        FP = 0
        FN = 0
        P = len(targetMarks)
        # For each detection check if it was a target
        for detected in detectedMarks:
            tp = False
            for target in targetMarks:
                if (isCorrectDetection(detected,target)):
                    # True positive
                    tp = True
                    TP += 1
                    detected.set_color(C_TP)
                    # Delete target to avoid counting duplicate detections as true positives
                    targetMarks.remove(target)
                    break

            # Something was detected but is not one of the targets
            if (not tp):
                # False positive
                FP += 1
                detected.set_color(C_FP)

            # Add detection with update color to list
            displayMarks.append(detected)

        # The rest of the targets are false negatives
        for target in targetMarks:
            target.set_color(C_FN)
        FN = len(targetMarks)
        displayMarks += targetMarks

        # Add marks to plot
        for mark in displayMarks:
            ax.add_patch(mark)

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
