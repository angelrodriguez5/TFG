import glob
import random
import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import Yolo.utils.augmentations as augment
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True, unlabeled=False):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.unlabeled = unlabeled

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            # 25% not augment
            # 25% flip
            # 25% change color
            # 25% flip + change color
            rnd = np.random.random()
            if rnd < 0.25:
                img, targets = augment.horisontal_flip(img, targets)
            elif rnd < 0.5:
                img = augment.fixed_color_shift(img)
            elif rnd < 0.75:
                img, targets = augment.horisontal_flip(img, targets)
                img = augment.fixed_color_shift(img)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        if not self.unlabeled:
            # Remove empty placeholder targets
            targets = [boxes for boxes in targets if boxes is not None]
            # Add sample index to targets
            for i, boxes in enumerate(targets):
                boxes[:, 0] = i
            targets = torch.cat(targets, 0)

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


class VideoDataset(Dataset):
    def __init__(self, video_path, img_size=416, frame_skip=0):
        # Open video capture
        self.capture = cv2.VideoCapture(video_path)
        # get total number of frames
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # Number of frames to be skipped between returned images
        self.frame_skip = frame_skip
        self.img_size = img_size

    def __getitem__(self, index):
        # Get frame number from index
        if self.frame_skip:
            frame_num = (index * self.frame_skip) % self.total_frames
        else:
            frame_num = index % self.total_frames
        # jump to selected frame
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        state, img = self.capture.read()
        if state:
            # Correct reading
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Frame was not read, return a black frame
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        # Frames to secs
        s = frame_num / float(self.get_framrate())

        return s, img

    def __len__(self):
        return (int(self.total_frames / self.frame_skip) if self.frame_skip else self.total_frames)

    def __del__(self):
        self.capture.release()

    def get_framrate(self):
        return self.capture.get(cv2.CAP_PROP_FPS)
