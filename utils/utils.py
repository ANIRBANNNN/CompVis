import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def openRGB(path):
    assert os.path.exists(path), f"Image path doesn't exist. {path}"
    return mpimg.imread(path)

def showRGB(imgarray):
    plt.imshow(imgarray)
    # plt.show() 

def conv2d(img, filter):
    pass

class filters:
    def generateKernel(type='mean', dim=(3,3)):
        pass

    
