import cv2
import numpy as np
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def exportFrames(fileName):
    # Open video capture
    cap = cv2.VideoCapture(fileName)

    # Create the name of the directory from the name of the video
    dirName = os.path.splitext(fileName)[0] + '_frames'
    try:
        os.mkdir(dirName)
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
        # Cancel script
        exit(1)

    # get total number of frames
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    vidName = os.path.splitext(os.path.basename(fileName))[0]

    # Get one image for every 60 frames
    for i in range(0, totalFrames, 240):
        # jump to selected frame witout reading all the intermediate frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        status, frame = cap.read()

        # Save frame as a png image in the directory
        name = dirName + "/" + vidName + "-frame" + str(i) + ".png"
        # Flip frame to save it in the correct orientation
        # save = cv2.flip(cv2.transpose(frame), 0)
        cv2.imwrite(name, frame)
        count += 1

    print ("Saved %d frames" % count)
    cap.release()


def main():
    # we don't want a full GUI, so keep the root window from appearing
    Tk().withdraw()
    # Supported file extensions
    ftypes = [("Video", "*.mov *.mp4 *.wmv")]
    # show an "Open" dialog box and return the path to the selected file
    filename = askopenfilename(filetypes=ftypes)

    if (filename):
        # Execute GUI
        exportFrames(filename)
    else:
        print('Cancelled')



if (__name__ == '__main__'):
    # Run program
    main()