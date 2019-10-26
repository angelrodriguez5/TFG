from __future__ import division
from statistics import mean

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

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="gradient_10_StepOnVideoChange", help="name of the folder to save logs, checkpoints...")
    parser.add_argument("--epochs", type=int, default=150, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=10, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/customModelDef.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default="Yolo/weights/darknet53.conv.74", help="if specified starts from checkpoint model") # default="weights/darknet53.conv.74"
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--test_interval", type=int, default=10, help="interval evaluations on test set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    pos_test_path = data_config["pos_test"]
    neg_test_path = data_config["neg_test"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    print ("memory allocated : %f" % torch.cuda.memory_allocated())
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        model.train()

        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if (batches_done + 1) % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]

                metric_table += [[metric, *row_metrics]]

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        # Calculate training metrics over the whole set instead of batch by batch
        print("--- Training epoch metrics ---")
        precision, recall, AP, f1, ap_class, loss = evaluate(
                model,
                path=train_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=1,
            )
        trainning_metrics = [
            ("tra_precision", precision),
            ("tra_recall", recall),
            ("tra_mAP", AP.mean()),
            ("tra_f1", f1),
            ("tra_loss", loss.sum())
        ]
        logger.list_of_scalars_summary(trainning_metrics, epoch)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class, loss = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=1,
            )
            evaluation_metrics = [
                ("val_precision", precision),
                ("val_recall", recall),
                ("val_mAP", AP.mean()),
                ("val_f1", f1),
                ("val_loss", loss.sum())
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class mAP
            print(f"---- mAP {AP.mean()}")
            print()

        if epoch != 0 and epoch % opt.test_interval == 0:
            print("\n---- Testing Model ----")

            # Test the model on images with bleeding
            print("\n---- Positive test ----")
            precision, recall, AP, f1, ap_class, loss = performTest(
                model,
                path=pos_test_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=1,
                epoch=epoch
            )
            pos_test_metrics = [
                ("test_precision", precision),
                ("test_recall", recall),
                ("test_mAP", AP.mean()),
                ("test_f1", f1),
                ("test_loss", loss.sum())
            ]
            logger.list_of_scalars_summary(pos_test_metrics, epoch)

            print(f"---- mAP {AP.mean()}")
            
            # Test the model on images without bleeding
            print("\n---- Negative test ----")
            
            false_positives = performNegativeTest(
                model,
                path=neg_test_path,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=1
            )

            neg_test_metrics = [
                ("neg_test_#FP", false_positives.sum()),
                ("neg_test_mFP", false_positives.mean())
            ]
            logger.list_of_scalars_summary(neg_test_metrics, epoch)
            
        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)


    # Export the results of the training
    exportResults(opt.experiment_name)
