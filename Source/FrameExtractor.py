import cv2
import numpy as np
import os

# Create directory
dirName = "Frames"
try:
    os.mkdir(dirName)
except FileExistsError:
    print("Directory " , dirName ,  " already exists")
    # Cancel script
    exit(1)

fileName = "vid/test.mp4"
cap = cv2.VideoCapture(fileName)

# get total number of frames
totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
count = 0
# Get one image for every 60 frames
for i in range(0, totalFrames, 30):
    # jump to selected frame witout reading all the intermediate frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    status, frame = cap.read()

    # Save frame as a png image in the folder
    name = dirName + "/frame" + str(i) + ".png"
    # Flip frame to save it in the correct orientation
    save = cv2.flip(cv2.transpose(frame), 0)
    cv2.imwrite(name, save)
    count += 1

print ("Saved %d frames" % count)
cap.release()