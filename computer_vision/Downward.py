"""
This is the Python code written for the Downward jupyter notebook.
"""
from __future__ import print_function

# Basic libraries for calculations and visualizations.
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# Grab all the files in the folder.
import glob

# Extra practice topics.
from matplotlib.patches import Rectangle
from scipy.stats import linregress
import matplotlib.patches as patches
from matplotlib.lines import Line2D

def show_image_set(images):
    """
    Display the set of the images in a 4x4 grid created by matplotlib.
    """
    fig, ax = plt.subplots(nrows=4, ncols=4)
    
    curx, cury = 0, 0
    for image in images:
        ax[curx, cury].imshow(image)

        cury += 1
        if cury == 4:
            curx, cury = curx+1, 0

    fig.set_size_inches(10, 7)

    fig.show()


def get_largest_contour(image, grayed: bool=True):
    """
    Get the largest contour in the given image.
    """
    
    if grayed == False:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    if contours is None or len(contours) == 0:
        return None
    
    curContour = None
    curArea = 0

    for contour in contours:
        if cv.contourArea(contour) >= curArea:
            curArea = cv.contourArea(contour)
            curContour = contour
    
    return curContour


def main(args=None):
    # Step 1: Read the images.
    images = glob.glob("computer_vision/downward/*.jpg")
    images.sort()
    images = [cv.imread(image, cv.IMREAD_GRAYSCALE) for image in images]
    # print(f"Length of the dataset is {len(images)}")
    # show_image_set(images)

    # Step 2: Dilution and erosion to get the images.
    kernel_5x5 = np.ones((5, 5), dtype=np.float32)
    for i in range(len(images)):
        img = images[i]
        img = cv.dilate(img, kernel_5x5, iterations=2)
        img = cv.erode(img, kernel_5x5, iterations=2)
        images[i] = img
    show_image_set(images)

    # Step 3: Apply a threshold to the image set.
    kernel_3x3 = np.ones((3, 3), dtype=np.float32)
    for i in range(len(images)):
        img = images[i]
        ret, img = cv.threshold(img, 230, 255, cv.THRESH_BINARY)

        # Erode then dilute to remove dots.
        img = cv.erode(img, kernel_3x3, iterations=1)
        img = cv.dilate(img, kernel_3x3, iterations=1)

        images[i] = img
    show_image_set(images)

    # Step 4: Find the largest contour among the set of the contours.
    masks: list = []
    for i in range(len(images)):
        img = images[i]
        contour = get_largest_contour(img)
        if contour is None:
            continue
        img = cv.drawContours(np.zeros(img.shape), contour, -1, (0, 255, 0), 3)
        images[i] = img
    show_image_set(images)

    # Final Step: show the diagram.
    plt.show()
    


if __name__ == "__main__":
    main()
