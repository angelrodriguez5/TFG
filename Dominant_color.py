from __future__ import division

import cv2
import numpy as np
import tqdm

from torch.utils.data import DataLoader
from Yolo.utils.datasets import ImageFolder
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import pyplot as plt

def dominant_colours(filename):
    ''' Returns a 3-tuple of lists of pairs (component_value, repetitions) for the HSV components of the image denoted by filename in HSV colour space.
        The image is rescaled to 150x150 so the total number of pixels analysed is 22500'''
    # Resizing for time efficiency
    dim = (150,150)
    image = cv2.imread(filename)
    image = cv2.resize(image, dim)
    # Transform to HSV (0..179, 0..255, 0..255)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract hues and their repetitions
    h = image[:,:,0]
    h_dic = {}
    for hue in h.flatten():
        if hue in h_dic:
            h_dic[hue] +=1
        else:
            h_dic[hue] = 1
    
    # Extract saturations and their repetitions
    s = image[:,:,1]
    s_dic = {}
    for sat in s.flatten():
        if sat in s_dic:
            s_dic[sat] +=1
        else:
            s_dic[sat] = 1

    # Extract values and their repetitions
    v = image[:,:,2]
    v_dic = {}
    for val in v.flatten():
        if val in v_dic:
            v_dic[val] +=1
        else:
            v_dic[val] = 1

    # Transform dictionary to list of pairs (hue, repetitions) and sort it by descending number of repetitions
    hues = sorted(list(h_dic.items()), key=lambda x: x[1], reverse=True)
    sats = sorted(list(s_dic.items()), key=lambda x: x[1], reverse=True)
    vals = sorted(list(v_dic.items()), key=lambda x: x[1], reverse=True)
    return (hues, sats, vals)

def list_of_top_values(lst, sample_total, percent):
    '''
    Given a list of (value, repetitions) returns the top values that account for a greater or equal
    percentage to the value given (between 0..1) out of the sample_total
    '''
    tmp = []
    acc_percent = 0
    for value, count in lst:
        tmp.append(value)
        acc_percent += float(count) / sample_total
        if acc_percent >= percent:
            return tmp


def normalise(lst, inrange=(0,255), outrange=(0,1)):
    in_lower, in_upper = inrange
    out_lower, out_upper = outrange

    # Transformation function
    fun = lambda x: (out_upper - out_lower) * ((x - in_lower) / (in_upper - in_lower)) + out_lower

    return list(map(fun, lst))

if (__name__ == '__main__'):
    # Folder dataset
    folder_path = "C:\\Users\\pickl\\Documents\\UDC2018\\TFG\\Colour_study"
    dataloader = DataLoader(
        ImageFolder(folder_path, img_size=416),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    dominant_hues = set()
    dominant_sats = set()
    dominant_vals = set()
    for i, (paths, imgs) in enumerate(tqdm.tqdm(dataloader, "Analysing images: ")):
        # Assume batch size = 1
        hues, sats, vals = dominant_colours(paths[0])

        # Extract top percentage of HSV
        percent = 0.5
        samples = 22500
        dominant_hues.update(list_of_top_values(hues, samples, percent))
        dominant_sats.update(list_of_top_values(sats, samples, percent))
        dominant_vals.update(list_of_top_values(vals, samples, percent))
        
    dominant_hues = normalise(dominant_hues, (0,179), (0,359))
    dominant_sats = normalise(dominant_sats)
    dominant_vals = normalise(dominant_vals)
    hue_range = (min(dominant_hues), max(dominant_hues))
    sat_range = (min(dominant_sats), max(dominant_sats))
    val_range = (min(dominant_vals), max(dominant_vals))
    print("Dominant values across all images:")
    print(" --Hue: %d..%d" % hue_range)
    print(" --Sat: %f..%f" % sat_range)
    print(" --Val: %f..%f" % val_range)


