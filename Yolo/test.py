from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils.CustomDatasetExporter import Mark

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
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
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
        sample_metrics = [[[0], torch.Tensor([0]), torch.Tensor([0])]]

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class, np.array(imgLoss)

# Mark -> Mark -> Bool
def isCorrectDetection(detected, target):
    # TODO iou instead of distance
    threshold = 25
    # Distance between centers 
    dx, dy = detected.getCenter()
    dw, dh = detected.get_width(), detected.get_height()
    tx, ty = target.getCenter()
    tw, th = target.get_width(), target.get_height()

    distance = math.sqrt( ((dx-tx)**2)+((dy-ty)**2) )

    return distance <= threshold

def performTest__OLD__(model, classes, image_folder, epoch, conf_thres=0.8, nms_thres=0.4, batch_size=1, n_cpu=0, img_size=416):
    model.eval()  # Set in evaluation mode

    os.makedirs("output/test_epoch_%d"%(epoch), exist_ok=True)

    dataloader = DataLoader(
        ImageFolder(image_folder, img_size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
    )

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
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Dictionary to store confusion matrix of test images
    confusionDict = {}

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))
        
        # Create plot
        img = np.array(Image.open(path))
        height, width, channels = img.shape
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Load test set marks
        targetMarks = []
        fileName = path.replace("_img", "_tag").replace(".png", ".txt")
        if (os.path.isfile(fileName)):
            f = open(fileName, 'r')
            for line in f:
                # Cast array of string to floats
                array = [float(x) for x in line.split()]
                # Cast class number to int
                array[0] = int(array[0])
                # Populate target list with marks in the file
                mark = Mark.buildFromNorm(array, (width, height))
                targetMarks.append(mark)

        # Transform detections to marks
        detectedMarks = []
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, img_size, img.shape[:2])

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
                if (isCorrectDetection(detected, target)):
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

            # Add detection with updated color to list
            displayMarks.append(detected)

        # The targets that weren't found are false negatives
        for target in targetMarks:
            target.set_color(C_FN)
        FN = len(targetMarks)
        displayMarks += targetMarks

        # Confusion matrix
        recall = TP / P
        miss_rate = FN / P
        precission =  0 if (TP + FP) == 0 else TP / (TP + FP)
        # Print confusion matrix
        print('Confusion matrix for image: %s' % (path))
        table = [["recall", "miss rate", "Precission"], [recall, miss_rate, precission]]
        print(AsciiTable(table).table)
        print('\n')
        # Add confusion matrix to dictionary
        myConfusion = {"recall":recall, "miss_rate":miss_rate, "precission":precission}
        confusionDict[path] = myConfusion

        # Add marks to plot
        for mark in displayMarks:
            ax.add_patch(mark)

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig("output/test_epoch_%d/%s.png" % (epoch,filename), bbox_inches="tight", pad_inches=0.0)
        plt.close('all')

    # Log confusion of test images
    print("Done testing!")

def performTest(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, negative_test):
    """
    returns (precision, recall, ap, fi, ap_class, loss, fp)
    if negative_test then fp stores the number of detections (i.e. false positives) per image and the rest of elemenets are empty
    otherwise fp is empty
    """

    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgLoss = []
    fp = []
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
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

        if negative_test:
            fp += len(outputs)
        else:
            imgLoss += [loss.item()]
            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

        # TODO save images with marks

    if negative_test:
        return [], [], [], [], [], [], np.array(fp)

    # In case of no outputs, load dummy sample metrics to avoid crashing
    if (len(sample_metrics) == 0):
        #Dummy metrics
        sample_metrics = [[[0], torch.Tensor([0]), torch.Tensor([0])]]

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class, np.array(imgLoss), []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/customModelDef.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_99.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class, validLoss, totalImgs = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
