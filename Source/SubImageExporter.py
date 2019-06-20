import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os

# Config constants
REGION_SIZE = (7,5)
PATH = "img/test.png"

# List of chosen points
marks = []

# Visualisation mode
pixelMode = True

# region Mark
class Mark(patches.Rectangle):

    DEFAULT_COLOR = 'b'

    ''' Instance attributes
        - center
    '''

    def __init__(self, center):
        # Default, 1 pixel square that shadows a pixel of an image
        # Pyplot prints the pixel (0,0) of an image around the point (0,0) of the plot
        # so we need to offset the position of the mark to properly aling with the pixel in the image
        self.center = center
        xy = self.calcPointXY()
        super().__init__(xy, 1, 1)
        
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
            (width, height) = REGION_SIZE
            self.set_xy(self.calcBoxXY())
            self.set_width(width)
            self.set_height(height)
            # Draw border with transparent fill
            self.set_color(self.DEFAULT_COLOR)
            self.fill = False
            self.set_linewidth(1)

    def calcPointXY(self):
        # Offset xy so the square is drawn over the center pixel
        newX = self.center[0] - .5
        newY = self.center[1] - .5
        return (newX, newY)

    def calcBoxXY(self):
        # Calculate the position of the corner of the box taking into account the offset
        newX = self.center[0] - math.floor(REGION_SIZE[0] / 2) - .5
        newY = self.center[1] - math.floor(REGION_SIZE[1] / 2) - .5
        return (newX, newY) 

#endregion

# region States
# Interface
class State(object):

    def processClick(self, event):
        raise NotImplementedError


class AddState(State):

    def processClick(self, event):
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
        except StopIteration as error:
            # if it is not found, add it to the mark list
            mark = Mark(xy)
            marks.append(mark)
            # And to the figure
            ax.add_patch(mark)
            print ("New Mark added at: %s" % (xy,))


class MoveState(State):

    def processClick(self, event):
        print ('Moving')


class DeleteState(State):

    def processClick(self, event):
        # Reference to the subplot clicked
        ax = event.inaxes
        # Round event data to get the pixel
        # The pixel (0,0) of an image is plotted between (-0.5, -0.5) and (0.5, 0.5)
        xy = (int(round(event.xdata)), int(round(event.ydata)))
        try:
            mark = next(x for x in marks if x.getCenter() == xy)
            # Mark found
            marks.remove(mark)
            mark.remove()
            print ("Mark removed at: %s" % (xy,))
        except StopIteration as error:
            # Mark not found
            # TODO notify user
            pass

#endregion

class GUI(object):

    ADD_STATE = 0
    MOV_STATE = 1
    DEL_STATE = 2

    '''Instance attributes
        -state
        -img
    '''

    def __init__(self):
        
        # Open image in bgr
        self.img = cv2.imread(PATH)
        # Conver to rbg to show it
        rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Configure figure
        fig = plt.figure()
        plt.imshow(rgb)
        cidButton = fig.canvas.mpl_connect('button_press_event', self.onClick)
        cidKey = fig.canvas.mpl_connect('key_press_event', self.onKey)

        # Set initial state
        self.setState(self.MOV_STATE)

        # Show figure
        plt.show()
    
    def onClick(self, event):
        # Delegate click management to the state
        self.state.processClick(event)
        # Update figure
        # TODO return a boolean saying if redraw is needed or not
        event.canvas.draw()

    def onKey(self, event):
        key = event.key
        # Map keys to actions
        if   (key == "1"):
            self.setState(self.ADD_STATE)

        elif (key == "2"):
            self.setState(self.MOV_STATE)

        elif (key == "3"):
            self.setState(self.DEL_STATE)

        elif (key == "4"):
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
        if (code == self.ADD_STATE):
            plt.title("Add")
            self.state = AddState()

        elif (code == self.MOV_STATE):
            plt.title("Move")
            self.state = MoveState()

        elif (code == self.DEL_STATE):
            plt.title("Delete")
            self.state = DeleteState()

    def exportRegions (self):
        # Create folder
        dirName = "Subimages"
        try:
            os.mkdir(dirName)
        except FileExistsError:
            print("Directory " , dirName ,  " already exists")
            # Cancel export
            return

        # Iterate over mark list
        secNum = 0
        for mark in marks:
            secNum += 1

            # Get subimage centered around mark center and with size REGION_SIZE
            (x, y) = mark.getCenter()
            beginX = x - math.floor(REGION_SIZE[0] / 2)
            beginY = y - math.floor(REGION_SIZE[1] / 2)
            subImg = self.img[  beginY : beginY + REGION_SIZE[1], 
                                beginX : beginX + REGION_SIZE[0]]

            # Save subimage
            name = dirName + "/subimg" + str(secNum) + ".png"
            cv2.imwrite(name, subImg)

        print ("Exported " + str(secNum) + " subImages")

# Execute GUI
gui = GUI()