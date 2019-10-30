from __future__ import division

from Yolo.models import *
from Yolo.utils.utils import *
from Yolo.utils.datasets import *
from Yolo.utils.parse_config import *
from Yolo.utils.CustomDatasetExporter import Mark

from terminaltables import AsciiTable

from matplotlib.ticker import NullLocator

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgLoss = []
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Validating model")):
        # import targets to CUDA
        cudaTargets = Variable(targets.to(device), requires_grad=False)
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            loss, outputs = model(imgs, cudaTargets)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        imgLoss += [loss.item()]
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # In case of no outputs, load dummy sample metrics to avoid crashing
    if (len(sample_metrics) == 0):
        #Dummy metrics
        sample_metrics = [[[0],[0], torch.Tensor([0]), torch.Tensor([0])]]

    # Concatenate sample statistics
    false_negatives, true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # Calculate precision, recall and f1 for the whole batch
    # to avoid images with fewer targets weighting more
    # true_positives is an array of arrays of flags corresponding to tp detections for each image
    tp = true_positives.sum()
    # total number of detections - true positives
    fp = (np.prod(true_positives.shape)) - tp
    fn = false_negatives.sum()

    recall = tp / (tp + fn + 1e-16)
    precision = tp / (tp + fp + 1e-16)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)

    return precision, recall, AP, f1, ap_class, np.array(imgLoss)

def addBox(ax, box, color):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    # Create a Rectangle patch
    patch = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor=color, facecolor="none")
    ax.add_patch(patch)

def printTestImageResults(paths, img_size, epoch, false_negatives, true_positives, targets, outputs):
    """ Save a copy of the images with color coded rectangles denoting
        true positives, false positives and false negatives """
    # Color code for the rectangles
    C_TP = 'g'
    C_FP = 'r'
    C_FN = 'b'

    os.makedirs("output/test_epoch_%d"%(epoch), exist_ok=True)

    print("\nSaving images:")
    for sample_i in range(len(paths)):
        # Create plot
        img = np.array(Image.open(paths[sample_i]))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        tp_count = 0
        fp_count = 0
        fn_count = 0
        if outputs[sample_i] is not None:
            # Rescale boxes to original image
            output = rescale_boxes(outputs[sample_i], img_size, img.shape[:2])
            pred_boxes = output[:, :4]

            for i, box in enumerate(pred_boxes):
                if true_positives[sample_i][i]:
                    # True positive
                    addBox(ax, box, C_TP)
                    tp_count += 1
                else:
                    # False positive
                    addBox(ax, box, C_FP)
                    fp_count += 1

        if targets is not None:
            annotations = targets[targets[:, 0] == sample_i][:, 1:]
            target_boxes = rescale_boxes(annotations[:, 1:], img_size, img.shape[:2])

            for i, box in enumerate(target_boxes):
                if false_negatives[sample_i][i]:
                    # False negative
                    addBox(ax, box, C_FN)
                    fn_count += 1

        print("Img %d: %s" % (sample_i, paths[sample_i]))
        print("    #TP: %d" % tp_count)
        print("    #FP: %d" % fp_count)
        print("    #FN: %d" % fn_count)
        print(" --------- ")

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = paths[sample_i].split("/")[-1].split(".")[0]
        plt.savefig("output/test_epoch_%d/%s.png" % (epoch,filename), bbox_inches="tight", pad_inches=0.0)
        plt.close('all')

def performTest(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, epoch):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgLoss = []
    labels = []
    sample_metrics = []  # List of tuples (FN, TP, confs, pred)
    for batch_i, (img_paths, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # import targets to CUDA
        cudaTargets = Variable(targets.to(device), requires_grad=False)
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        # Test model
        with torch.no_grad():
            loss, outputs = model(imgs, cudaTargets)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        imgLoss += [loss.item()]
        batch_stats = get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        sample_metrics += batch_stats

        # If detections were made
        if len(batch_stats):
            false_negatives, true_positives, _, _ = list(zip(*batch_stats))
            # Save images with color coded detections and targets
            printTestImageResults(img_paths, img_size, epoch, false_negatives, true_positives, targets, outputs)

    # In case of no outputs, load dummy sample metrics to avoid crashing
    if (len(sample_metrics) == 0):
        # Dummy metrics
        sample_metrics = [[[0], [0], torch.Tensor([0]), torch.Tensor([0])]]

    # Concatenate sample statistics
    false_negatives, true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # Calculate precision, recall and f1 for the whole batch
    # to avoid images with fewer targets weighting more
    # true_positives is an array of arrays of flags corresponding to tp detections for each image
    tp = true_positives.sum()
    # total number of detections - true positives
    fp = (np.prod(true_positives.shape)) - tp
    fn = false_negatives.sum()

    recall = tp / (tp + fn + 1e-16)
    precision = tp / (tp + fp + 1e-16)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)

    return precision, recall, AP, f1, ap_class, np.array(imgLoss)

def performNegativeTest(model, path, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False, unlabeled=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    false_positives = []
    for batch_i, (img_paths, imgs, _) in enumerate(dataloader):

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        # Test model
        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        # All detections are false positives
        batch_fp = [0 if x is None else len(x) for x in outputs]
        false_positives.extend(batch_fp)

    return np.array(false_positives)
