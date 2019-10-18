import random
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from torchvision.transforms import ToTensor

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def fixed_color_shift(img):
    # Tensor to RGB(0..1) image
    i = img[0].permute(1, 2, 0).numpy()
    # RGB to HSV (0..360, 0..1, 0..1)
    i = cv2.cvtColor(i, cv2.COLOR_RGB2HSV)
    # Transformations in HSV space (2.2%, 10%, 10%)
    h_off = random.uniform(-8,8)
    s_off = random.uniform(-0.1,0.1)
    v_off = random.uniform(-0.1,0.1)
    i[:,:,0] += h_off
    i[:,:,1] += s_off
    i[:,:,2] += v_off
    # HSV to RGB (0..1) to Tensor
    return ToTensor()(cv2.cvtColor(i, cv2.COLOR_HSV2RGB))