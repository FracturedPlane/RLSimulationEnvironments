
#include "eglRender.h"
import sys
sys.path.append("lib")
sys.path.append("rendering")
from rendering import eglRenderer

if __name__ == '__main__':
     
    eglRenderer._init()
    
    for  i in range(5):
        eglRenderer.setPosition(i * 0.02, i * 0.02, 0)
        eglRenderer.setPosition2(i * -0.02, i * -0.02, 0)
        eglRenderer.draw()
        eglRenderer.save_PPM()
    
    eglRenderer.finish()
