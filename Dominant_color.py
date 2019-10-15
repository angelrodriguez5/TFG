import cv2
import numpy as np


from torch.utils.data import DataLoader
from Yolo.utils.datasets import ImageFolder
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import pyplot as plt

def dominant_colours(filename):
    ''' Returns a list of pairs (hue, repetitions) for the hue values of the image denoted by filename in HSV colour space.
        The image is rescaled to 150x150 so the total number of pixels is 22500'''
    # Resizing for time efficiency
    dim = (150,150)
    image = cv2.imread(filename)
    image = cv2.resize(image, dim)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Get hSV color values
    hues = image[:,:,0]
    count_dic = {}
    for hue in hues.flatten():
        if hue in count_dic:
            count_dic[hue] +=1
        else:
            count_dic[hue] = 1
    
    # Transform dictionary to list of pairs (hue, repetitions)
    lst = list(count_dic.items())
    # Sort in descending order of count
    return sorted(lst, key=lambda x: x[1], reverse=True)
    

if (__name__ == '__main__'):
    # Folder dataset
    folder_path = "C:\\Users\\pickl\\Documents\\UDC2018\\TFG\\Colour_study"
    dataloader = DataLoader(
        ImageFolder(folder_path, img_size=416),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    dominant_hues = []
    for i, (paths, imgs) in enumerate(dataloader):
        print("Analysing image: "+paths[0])
        # Assume batch size = 1
        hues = dominant_colours(paths[0])

        # Hue histogram
        # plt.bar(*zip(*hues))
        # plt.show()

        # Percentage of image by colours
        total_px = 22500
        acc_percent = 0
        tmp = []
        for hue, count in hues:
            tmp.append(hue)
            acc_percent += float(count) / total_px * 100
            if acc_percent >= 60:
                print ("%f%% of image taken by hues: %s" %(acc_percent, str(tmp)))
                dominant_hues.append(tmp)
                break
            if acc_percent >= 50:
                print ("%f%% of image taken by hues: %s" %(acc_percent, str(tmp)))
                continue
            if acc_percent >= 40:
                print ("%f%% of image taken by hues: %s" %(acc_percent, str(tmp)))
                continue
            if acc_percent >= 30:
                print ("%f%% of image taken by hues: %s" %(acc_percent, str(tmp)))
                continue
    
    dominant_set = np.unique(np.array(dominant_hues).flatten())
    print("Dominant hues across all images: " + str(dominant_set))
