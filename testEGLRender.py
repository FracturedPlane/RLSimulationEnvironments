
#include "eglRender.h"
import sys
sys.path.append("lib")
sys.path.append("rendering")
from rendering import eglRenderer
import matplotlib.pyplot as plt
import numpy as np
from OpenGL.GL import *

if __name__ == '__main__':
     
    eglRenderer._init()
    
    for  i in range(10):
        eglRenderer.setPosition(i * 0.02, i * 0.02, 0)
        eglRenderer.setPosition2(i * -0.02, i * -0.02, 0)
        eglRenderer.draw()
        eglRenderer.save_PPM()
        eglRenderer.setCameraPosition(i * 0.02, i * 0.02, 0)
        eglRenderer.setDrawAgent(True)
        eglRenderer.setDrawObject(False)
        eglRenderer.draw()
        img_ = eglRenderer.getPixels(0, 0, 1000, 1000)
        img_ = np.reshape(img_, (1000, 1000, 3))
        eglRenderer.setCameraPosition(i * -0.02, i * -0.02, 0)
        eglRenderer.setDrawAgent(False)
        eglRenderer.setDrawObject(True)
        eglRenderer.draw()
        img2_ = eglRenderer.getPixels(0, 0, 1000, 1000)
        img2_ = np.reshape(img2_, (1000, 1000, 3))
        # img_ = getViewData()
        print("img_ shape", img_.shape, " sum: ", np.sum(img_))
        plt.figure(1)
        plt.imshow(img_, origin='lower')
        plt.title("visual Data: ")
        
        plt.figure(2)
        plt.imshow(img2_, origin='lower')
        plt.title("imitation visual Data: ")

        plt.show()
    eglRenderer.finish()
