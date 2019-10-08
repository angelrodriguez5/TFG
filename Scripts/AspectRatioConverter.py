import os
import cv2
import glob
import numpy as np

if __name__ == "__main__":
    # Find all .png files in current directory
    imgs = [f for f in glob.glob("*.png")]
    
    os.makedirs("converted", exist_ok=True)

    for path in imgs:
        img = cv2.imread(path)
        name = path.split(".")[0]
        height, width, channels = img.shape

        # Calculate width of a 4/3 ratio image with current height
        newWidth = int((height / 3) * 4)
        leftStartIndex = width - newWidth

        # Extract two images with overlap
        imgL = img[:, :newWidth,:]
        imgR = img[:, leftStartIndex:,:]

        # Save images
        cv2.imwrite("converted/%s-L.png"%(name), imgL)
        cv2.imwrite("converted/%s-R.png"%(name), imgR)




