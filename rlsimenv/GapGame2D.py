"""
A 2D bouncing ball environment

"""


import sys, os, random
from math import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
# from twisted.protocols import stateful
import copy
import math
from .GapGame1D import *

    

class GapGame2D(GapGame1D):
    def __init__(self, settings):
        """Creates a ragdoll of standard size at the given offset."""
        
        super(GapGame2D,self).__init__(settings)
        self._x = []
        self._y = []
        self._step = 0
        
        
    def updateAction(self, action):
        pos = self._obstacle.getPosition()
        vel = self._obstacle.getLinearVel()
        # print ("Position Before action: ", pos)
        new_vel = np.array([vel[0] + action[0], action[1]])
        new_vel = clampAction(new_vel, self._game_settings["velocity_bounds"])
        # print("New action: ", new_vel)
        time = (new_vel[1]/9.81)*2 # time for rise and fall
        self._obstacle.setLinearVel((new_vel[0], new_vel[1], 0))
        
        self._x = []
        self._y = []
        self._step = 0
        
        steps=16
        # hopTime=1.0
        vel = self._obstacle.getLinearVel()
        time_ = (vel[1]/9.81)*2 # time for rise and fall
        dist = vel[0] * time_
        x = np.array(np.linspace(-0.5, 0.5, steps))
        y = np.array(list(map(self._computeHeight, x)))
        y = (y + math.fabs(float(np.amin(y)))) * action[1]
        x = np.array(np.linspace(0.0, 1.0, steps)) * dist
        # x = (x + 0.5) * action[0]
        x_ = (x + pos[0])
        self._x = x_
        self._y = y
        """
        dist = new_vel[0] * time
        self._obstacle.setPosition(pos + np.array([dist, 0.0, 0.0]))
        """
        
    def update(self):
        pos = self._obstacle.getPosition()
        new_vel = self._obstacle.getLinearVel()
        time_ = (new_vel[1]/9.81)*2 # time for rise and fall
        self.simulateAction(new_vel)
        dist = new_vel[0] * time_
        self._obstacle.setPosition(pos + np.array([dist, 0.0, 0.0]))
        # print ("Position After action: ", pos + np.array([dist, 0.0, 0.0]))

        # print (pos)

        # self._terrainData = self.generateTerrain()
        self._state_num=self._state_num+1
        # state = self.getState()
        # print ("state length: " + str(len(state)))
        # print (state)
        vel_dif = np.abs(self._target_velocity - new_vel[0])
        reward = math.exp((vel_dif*vel_dif)*self._target_vel_weight)
        # print("reward: ", reward)
        self.__reward = reward
        
    def calcReward(self):
        return self.__reward
        
    def display(self):
        if self._game_settings['render']:
            # self._obstacle.setPosition([self._x[self._step], self._y[self._step], 0.0] )
            pos_ = self._obstacle.getPosition()
            # print ("New obstacle position: ", pos_)
            
            glutPostRedisplay()
            self.onDraw()
            self._step +=1
    
    def needUpdatedAction(self):
        if self._step >= len(self._x):
            return True
        return False
        
    def actContinuous(self, action, bootstrapping=False):
        # print ("Action: ", action)
        
        if ( self._end_of_Epoch_Flag ) :
            return 0
        pos = self._obstacle.getPosition()
        vel = self._obstacle.getLinearVel()
        # print ("Position Before action: ", pos)
        # print ("action: ", action)
        new_vel = np.array([vel[0] + action[0], action[1]])
        new_vel = clampAction(new_vel, self._game_settings["velocity_bounds"])
        # print("New action: ", new_vel)
        time_ = (new_vel[1]/9.81)*2 # time for rise and fall
        self._obstacle.setLinearVel((new_vel[0], new_vel[1], 0))
        ## Move forward along X
        self.simulateAction(new_vel)
        dist = new_vel[0] * time_
        self._obstacle.setPosition(pos + np.array([dist, 0.0, 0.0]))
        # print ("Position After action: ", pos + np.array([dist, 0.0, 0.0]))

        # print (pos)

        # self._terrainData = self.generateTerrain()
        self._state_num=self._state_num+1
        # state = self.getState()
        # print ("state length: " + str(len(state)))
        # print (state)
        vel_dif = np.abs(self._target_velocity - new_vel[0])
        reward = math.exp((vel_dif*vel_dif)*self._target_vel_weight)
        # print("reward: ", reward)
        self.__reward = reward
        return reward
        # obstacle.addForce((0.0,100.0,0.0))
        
    def visualizeAction(self, action):
        print ("Action: ", action)
        pos = self._obstacle.getPosition()
        vel = self._obstacle.getLinearVel()
        print ("Velocity: ", vel)
        new_vel = np.array([vel[0] + action[0], action[1]])
        # new_vel = action[0]
        new_vel = clampAction(new_vel, self._game_settings["velocity_bounds"])
        ## compute new location for landing.
        time__ = self._computeTime(action[1]) * 2.0
        new_pos = pos[0] + (new_vel[0] * time__)
        self._obstacle2.setPosition((new_pos, 0,0))
    
    def visualizeActions(self, actions, dirs):
                # print ("Action: ", action)
        pos = self._obstacle.getPosition()
        vel = self._obstacle.getLinearVel()
        for a in range(len(actions)):
            new_vel = np.array([vel[0] + actions[a][0], actions[a][1]])
            # new_vel = action[0]
            # new_vel = clampAction(new_vel, self._game_settings["velocity_bounds"])
            ## compute new location for landing.
            time__ = self._computeTime(actions[a][1]) * 2.0
            new_pos = pos[0] + (new_vel[0] * time__)
            # self._obstacle2.setPosition((new_pos, 0,0)) 
            # print ("new obs location: ", new_pos)  
            self._obstacles[a].setPosition((new_pos, 0,2.1)) 
            self._obstacles[a].setDir(dirs[a]) 
    
    def visualizeNextState(self, terrain, action, terrain_dx):
        pos = self._obstacle.getPosition() 
        vel = self._obstacle.getLinearVel()
        terrain = pos[1] - terrain
        self._nextTerrainData = terrain
        new_vel = np.array([vel[0] + action[0], action[1]])
        # new_vel = action[0]
        new_vel = clampAction(new_vel, self._game_settings["velocity_bounds"])
        # self._obstacle.setLinearVel((action[0],4.0,0.0))
        time = (action[1]/9.81)*2 # time for rise and fall
        self._nextTerrainStartX = pos[0] + (time * new_vel[0]) + terrain_dx
        # self._nextTerrainStartX = pos[0] + terrain_dx
        # drawTerrain(terrain, translateX, translateY=0.0, colour=(0.4, 0.4, 0.8, 0.0), wirefame=False):
    
