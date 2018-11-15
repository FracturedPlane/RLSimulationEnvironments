"""
A 2D bouncing ball environment

"""


import sys, os, random, time
from math import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
# from twisted.protocols import stateful
import copy
import math
from rlsimenv.CannonGame import CannonGame

class CannonImitationGame(CannonGame):
    
    def __init__(self, settings):
        """
        
        """
        super(CannonImitationGame,self).__init__(settings)
        
    def getState(self):
        """ get the next self._num_points points"""
        state = []
        if ("process_visual_data" in self._game_settings
            and (self._game_settings["process_visual_data"] == True)
            and ("use_dual_state_representations" in self._game_settings
                 and (self._game_settings["use_dual_state_representations"] == True))):
            
            charState = self.getCharacterState()
            state_ = []
            state_.extend(charState)
            state.append(np.array(state_))
            
            ob = np.array(self.getVisualState())
            ob = np.reshape(np.array(ob), (-1, ob.size))
            state.append(ob)
            
            return [state]

        if ("process_visual_data" in self._game_settings
            and (self._game_settings["process_visual_data"] == True)):
            # print("Getting visual state")
            ob = np.array(self.getVisualState())
            ob = np.reshape(np.array(ob), (-1, ob.size))
            # print("ob shape: ", ob.shape)
            return ob
        pos = self._obstacle.getPosition()
        charState = self.getCharacterState()
        state = []
        state.extend(charState)
        return state
    
    