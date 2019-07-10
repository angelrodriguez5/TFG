import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# List of chosen points
marks = []

# Visualisation mode
pixelMode = False

# region Mark
class Mark(patches.Rectangle):

    DEFAULT_COLOR = 'b'
    DEFAULT_SIZE = (7,5)

    ''' Instance attributes
        - center
        - classNum
        - customSize
    '''

    def __init__(self, center, classNum, customSize = None):
        # Create patch
        super().__init__(center, 1, 1)

        self.center = center
        self.classNum = classNum    
        self.customSize = customSize  
        # Update the view to match current mode
        self.updateView()

    def getCenter(self):
        return self.center

    def updateView(self):
        if (pixelMode):
            # Draw a square that fills the center pixel
            self.set_xy(self.calcPointXY())
            self.set_width(1)
            self.set_height(1)
            # Solid color to hide center pixel completely
            self.set_color(self.DEFAULT_COLOR)
            self.fill = True

        else:
            # Draw a box around the center
            if (self.customSize):
                width, height = self.customSize
            else:
                width, height = self.DEFAULT_SIZE

            self.set_xy(self.calcBoxXY())
            self.set_width(width)
            self.set_height(height)
            # Draw border with transparent fill
            self.set_color(self.DEFAULT_COLOR)
            self.fill = False
            self.set_linewidth(1)

    def calcPointXY(self):
        # Pyplot prints the pixel (0,0) of an image around the point (0,0) of the plot
        # so we need to offset the position of the mark to properly aling with the pixel in the image
        # Offset xy so the square is drawn over the center pixel
        newX = self.center[0] - .5
        newY = self.center[1] - .5
        return (newX, newY)

    def calcBoxXY(self):
        # Calculate the position of the corner of the box taking into account the offset
        if (self.customSize):
            newX = self.center[0] - math.floor(self.customSize[0] / 2) - .5
            newY = self.center[1] - math.floor(self.customSize[1] / 2) - .5
        else:
            newX = self.center[0] - math.floor(self.DEFAULT_SIZE[0] / 2) - .5
            newY = self.center[1] - math.floor(self.DEFAULT_SIZE[1] / 2) - .5
        return (newX, newY)

    def normalise(self, imgDimensions):
        # Return a tuple with the class number, center coordinates and dimensions
        # normalised between 0 and 1 with respect to the size of an image
        cx, cy = map(lambda x,y: x / (y - 1), self.center, imgDimensions)
        if (self.customSize):
            w, h = map(lambda x,y: x / y, self.customSize, imgDimensions)
        else:
            w, h = map(lambda x,y: x / y, self.DEFAULT_SIZE, imgDimensions)

        return (self.classNum, cx, cy, w, h)

    @staticmethod
    def buildFromNorm(normForm, imgDimensions):
        classNum, nx, ny, nw, nh = normForm
        width, height = imgDimensions
        # Translate normalised form to image pixels
        cx = round((width - 1) * nx)
        cy = round((height - 1) * ny)
        w = round(width * nw)
        h = round(height * nh)

        return Mark((cx,cy), classNum, (w,h))


#endregion

# region States
# Interface
class State(object):

    def processClick(self, event, classNum):
        raise NotImplementedError


class AddPointState(State):

    # Click on center
    def processClick(self, event, classNum):
        # Reference to the subplot clicked
        ax = event.inaxes
        # Round event data to get the pixel
        # The pixel (0,0) of an image is plotted between (-0.5, -0.5) and (0.5, 0.5)
        xy = (int(round(event.xdata)), int(round(event.ydata)))
        try:
            # Try to find the mark in the list
            mark = next(x for x in marks if x.getCenter() == xy)
            # If it is found return
            return
        except StopIteration:
            # if it is not found, add it to the mark list
            # Create mark at center point with default size
            mark = Mark(xy, classNum)
            marks.append(mark)
            # And to the figure
            ax.add_patch(mark)
            print ("New Mark added at: %s" % (xy,))

class AddRegionState(State):
    ''' Instance attributes:
        -secondClick
        -topLeft
    '''
    def __init__(self):
        self.secondClick = False
        self.topLeft = None

    # Click on top left and bottom right to select a region
    def processClick(self, event, classNum):
        # Round event data to get the pixel
        # The pixel (0,0) of an image is plotted between (-0.5, -0.5) and (0.5, 0.5)
        xy = (int(round(event.xdata)), int(round(event.ydata)))
        # If it is the second click then calculate the mark position
        # and add it to the list
        if (self.secondClick):
            self.secondClick = False
            print ("Second click at: %s" % (xy,))
            # Get center of the region based on two clicks
            cxy = self.calculateCenter(self.topLeft, xy)
            size = self.calculateSize(self.topLeft, xy)
            try:
                # Try to find the mark in the list
                # PROBLEM with concentric regions of different size
                mark = next(x for x in marks if x.getCenter() == cxy)
                # If it is found return
                return
            except StopIteration:
                # if it is not found, add mark with custom size to the list
                mark = Mark(cxy, classNum, size)
                marks.append(mark)
            
                # Reference to the subplot clicked
                ax = event.inaxes
                # And to the figure
                ax.add_patch(mark)
                print ("New Mark added at: %s" % (cxy,))
        else:
            # If it is the first click, store its position and prepare for second click
            self.topLeft = xy
            self.secondClick = True
            print ("New region top left at: %s" % (xy,))

    def calculateCenter(self, topLeft, bottomRight):
        left, top = topLeft
        right, bot = bottomRight
        # Check that pixels are positioned correctly
        if (top > bot or left > right):
            raise Exception

        # Calculate center pixel favouring bottom and right for pair length sections
        cx = math.ceil((left + right) / 2)
        cy = math.ceil((top + bot) / 2)

        return (cx,cy)

    def calculateSize(self, topLeft, bottomRight):
        left, top = topLeft
        right, bot = bottomRight
        # Check that pixels are positioned correctly
        if (top > bot or left > right):
            raise Exception

        width = math.floor(right - left + 1)
        height = math.floor(bot - top + 1)

        return (width, height)

class MoveState(State):

    def processClick(self, event, classNum):
        print ('Moving')


class DeleteState(State):

    def processClick(self, event, classNum):
        # Round event data to get the pixel
        # The pixel (0,0) of an image is plotted between (-0.5, -0.5) and (0.5, 0.5)
        xy = (int(round(event.xdata)), int(round(event.ydata)))
        try:
            mark = next(x for x in marks if x.getCenter() == xy)
            # Mark found
            marks.remove(mark)
            mark.remove()
            print ("Mark removed at: %s" % (xy,))
        except StopIteration:
            # Mark not found
            # TODO notify user
            pass

#endregion

class GUI(object):

    ADD_CENTER_STATE = 0
    ADD_REGION_STATE = 1
    MOV_STATE = 2
    DEL_STATE = 3

    '''Instance attributes
        -imgPath
        -img
        -state
    '''

    def __init__(self, path):
        # Save path to image
        self.imgPath = path
        # Open image in bgr
        self.img = cv2.imread(self.imgPath)
        # Conver to rbg to show it
        rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Configure figure
        fig, ax = plt.subplots()
        plt.imshow(rgb)
        cidButton = fig.canvas.mpl_connect('button_press_event', self.onClick)
        cidKey = fig.canvas.mpl_connect('key_press_event', self.onKey)

        # Set initial state
        self.setState(self.MOV_STATE)

        # If .txt file exists, load existing marks
        fileName = os.path.splitext(self.imgPath)[0] + '.txt'
        if (os.path.isfile(fileName)):
            f = open(fileName, 'r')
            height, width, channels = self.img.shape
            while (True):
                line = f.readline()
                if line == '': break
                # Cast array of string to floats
                array = [float(x) for x in line.split()]
                # Cast class number to int
                array[0] = int(array[0])
                # Populate mark list with existing marks in the file
                mark = Mark.buildFromNorm(tuple(array), (width, height))
                marks.append(mark)

            # Add marks to plot
            for mark in marks:
                ax.add_patch(mark)
        #end if

        # Show figure
        plt.show()
    
    def onClick(self, event):
        classNum = 0
        # Check that click is in bounds
        height, width, channels = self.img.shape
        if (event is None):
            print('Click outside of canvas')
            return

        if (event.xdata < -0.5 or
            event.xdata > width - 0.5 or
            event.ydata < -0.5 or
            event.ydata > height - 0.5):

            print ('Click out of bounds')
            return
        
        # Delegate click management to the state
        self.state.processClick(event, classNum)
        # Update figure
        # TODO return a boolean saying if redraw is needed or not
        event.canvas.draw()

    def onKey(self, event):
        key = event.key
        # Map keys to actions
        if   (key == "1"):
            self.setState(self.ADD_CENTER_STATE)

        elif (key == "2"):
            self.setState(self.ADD_REGION_STATE)

        elif (key == "3"):
            self.setState(self.MOV_STATE)

        elif (key == "4"):
            self.setState(self.DEL_STATE)

        elif (key == "0"):
            self.exportRegions()

        elif (key == "º"):
            global pixelMode 
            pixelMode = not pixelMode
            for mark in marks:
                mark.updateView()

        # update canvas to show the title change
        event.canvas.draw()

    def setState(self, code):
        # Change title and state acording to the code
        if (code == self.ADD_CENTER_STATE):
            plt.title("Add point")
            self.state = AddPointState()

        elif (code == self.ADD_REGION_STATE):
            plt.title("Add region")
            self.state = AddRegionState()

        elif (code == self.MOV_STATE):
            plt.title("Move")
            self.state = MoveState()

        elif (code == self.DEL_STATE):
            plt.title("Delete")
            self.state = DeleteState()

    def exportRegions (self):
        # Save the text file in the same dir and with the same name as the image
        # Change image extension for .txt
        fileName = os.path.splitext(self.imgPath)[0] + '.txt'
        # Open or create the txt and overwrite it with current
        f = open(fileName, "w")
        
        height, width, channels = self.img.shape
        # Iterate over mark list
        for mark in marks:
            # Save normalised data for each mark
            line = '%d %.10f %.10f %.10f %.10f\n' % mark.normalise((width, height))
            f.write(line)
            
        print ("Exported %d subImages" % len(marks))

        # Close the file
        f.close()


def main():
    # we don't want a full GUI, so keep the root window from appearing
    Tk().withdraw()
    # Supported file extensions
    ftypes = [("Image", "*.png *.jpg *.jpeg *.bmp")]
    # show an "Open" dialog box and return the path to the selected file
    filename = askopenfilename(filetypes=ftypes)

    if (filename):
        # Execute GUI
        gui = GUI(filename)
    else:
        print('Cancelled')

if (__name__ == '__main__'):
    # Run program
    main()