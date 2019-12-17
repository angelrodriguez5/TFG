import os
import cv2
import glob
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askdirectory

def convertImages(directory):
    # Find all .png files in current directory
    imgs = [f for f in glob.glob("%s/*.png" % directory)]
    
    converted_dir = directory + "/converted"
    os.makedirs(converted_dir, exist_ok=True)

    for path in imgs:
        img = cv2.imread(path)
        name = os.path.splitext(os.path.basename(path))[0]
        height, width, channels = img.shape

        # Calculate width of a 4/3 ratio image with current height
        newWidth = int((height / 3) * 4)
        leftStartIndex = width - newWidth

        # Extract two images with overlap
        imgL = img[:, :newWidth,:]
        imgR = img[:, leftStartIndex:,:]

        # Save images
        cv2.imwrite("%s/%s-L.png"%(converted_dir, name), imgL)
        cv2.imwrite("%s/%s-R.png"%(converted_dir, name), imgR)

if __name__ == "__main__":
    # we don't want a full GUI, so keep the root window from appearing
    Tk().withdraw()
    # show an "Open" dialog box and return the path to the selected folder
    directory = askdirectory()

    if (directory):
        # Execute GUI
        convertImages(directory)
    else:
        print('Cancelled')




