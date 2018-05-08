"""
A 2D bouncing ball environment

"""


import sys, os, random, time
sys.path.append("../")
from math import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import ode
# from twisted.protocols import stateful
import copy
import math

# from ..model.ModelUtil import clampAction
def clampAction(actionV, bounds):
    """
    bounds[0] is lower bounds
    bounds[1] is upper bounds
    """
    for i in range(len(actionV)):
        if actionV[i] < bounds[0][i]:
            actionV[i] = bounds[0][i]
        elif actionV[i] > bounds[1][i]:
            actionV[i] = bounds[1][i]
    return actionV 

from .BallGame1D import *

class BallGame2D(BallGame1D):
    def __init__(self, settings):
        """Creates a ragdoll of standard size at the given offset."""
        super(BallGame2D,self).__init__(settings)
        
        
    def updateAction(self, action_):
        # print ("Action: ", action)
        pos = self._obstacle.getPosition()
        vel = self._obstacle.getLinearVel()
        new_vel = np.array([vel[0] + action_[0], action_[1]])
        # new_vel = action[0]
        new_vel = clampAction(new_vel, self._game_settings["velocity_bounds"])
        self._obstacle.setLinearVel((new_vel[0], new_vel[1], 0.0))
        contact = False
        
    def actContinuous(self, action, bootstrapping=False):
        # print ("Action: ", action)
        pos = self._obstacle.getPosition()
        vel = self._obstacle.getLinearVel()
        new_vel = np.array([vel[0] + action[0], action[1]])
        # new_vel = action[0]
        new_vel = clampAction(new_vel, self._game_settings["velocity_bounds"])
        self._obstacle.setLinearVel((new_vel[0], new_vel[1], 0.0))
        contact = False
        vel_sum=0
        updates=0
        while ( ( pos[1] >= (-5)) and (not contact)): # object does not fall off map..
        # while ( ( True ) and (not contact)):
            # print ("Before vel:, ", self.calcVelocity(bootstrapping=bootstrapping))
            contact = self.simulateAction()
            pos = self._obstacle.getPosition()
            pos = (pos[0], pos[1], 0.0)
            self._obstacle.setPosition(pos)
            updates+=1
            vel_sum += self.calcVelocity(bootstrapping=bootstrapping)
        
        # self._terrainData = self.generateTerrain()
        self._state_num=self._state_num+1
        # state = self.getState()
        # print ("state length: " + str(len(state)))
        # print (state)
        pos = self._obstacle.getPosition()
        # if ( not self.agentHasFallen() ):
        pos = (pos[0], self._ballRadius+self._ballEpsilon, 0.0)
        self._obstacle.setPosition(pos)
        ## The contact seems to be reducing the velocity
        # self._obstacle.setLinearVel((new_vel[0], new_vel[1], 0.0))
        # print ("Before vel:, ", self.calcVelocity(bootstrapping=bootstrapping))
        avg_vel = vel_sum/updates
        # print("avg_vel: ", avg_vel)
        return avg_vel
        # obstacle.addForce((0.0,100.0,0.0))
        
    def visualizeAction(self, action):
                # print ("Action: ", action)
        pos = self._obstacle.getPosition()
        vel = self._obstacle.getLinearVel()
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
            new_vel = clampAction(new_vel, self._game_settings["velocity_bounds"])
            ## compute new location for landing.
            time__ = self._computeTime(actions[a][1]) * 2.0
            new_pos = pos[0] + (new_vel[0] * time__)
            # self._obstacle2.setPosition((new_pos, 0,0))   
            self._obstacles[a].setPosition((new_pos, 0,0)) 
            self._obstacles[a].setDir(dirs[a])  
        
    def visualizeNextState(self, terrain, action, terrain_dx):
        self._nextTerrainData = terrain
        pos = self._obstacle.getPosition() 
        vel = self._obstacle.getLinearVel()
        new_vel = np.array([vel[0] + action[0], action[1]])
        # new_vel = action[0]
        new_vel = clampAction(new_vel, self._game_settings["velocity_bounds"])
        # self._obstacle.setLinearVel((action[0],4.0,0.0))
        time = (action[1]/9.81)*2 # time for rise and fall
        self._nextTerrainStartX = pos[0] + (time * new_vel[0]) + terrain_dx
        # self._nextTerrainStartX = pos[0] + terrain_dx
        # drawTerrain(terrain, translateX, translateY=0.0, colour=(0.4, 0.4, 0.8, 0.0), wirefame=False):

    def simulateAction(self):
        """
            Returns True if a contact was detected
        
        """
        if self._Paused:
            return
        t = self._dt - (time.time() - self._lasttime)    
        if self._game_settings['render']:
            if (t > 0):
                time.sleep(t)
            
        for i in range(self._stepsPerFrame):
            # Detect collisions and create contact joints
            self._space.collide((self._world, self._contactgroup), near_callback)
    
            # Simulation step (with slow motion)
            self._world.step(self._dt / self._stepsPerFrame / self._SloMo)
    
            self._numiter += 1
    
            # apply internal ragdoll forces
            # ragdoll.update()
            # pos = self._obstacle.getPosition()
            # print ("Ball pos: ", pos)
            # self._obstacle.addTorque((0.0,0.0,0.2));
                
            contacts = ode.collide(self._floor, self._obsgeom)
            # print ("Num contacts: " + str(len(contacts)))
            if (len(contacts)> 0):
                # print ("Num contacts: " + str(len(contacts)))
                # print ("Constact info: ", contacts[0].getContactGeomParams())
                return True
            
            # Remove all contact joints
            # for joint_ in self._contactgroup:
            #     print ("Joint: " + str(joint_))
            self._contactgroup.empty()
            
        if self._game_settings['render']:
            glutPostRedisplay()
            self.onDraw()
        return False
    
    def getSimState(self):
        charState = self.getCharacterSimState()
        x = self._terrainStartX
        terrain = copy.deepcopy(self._terrainData)
        return (charState, terrain, x)
    
    def setSimState(self, state_):
        (charState, terrain, x) = state_
        self.setCharacterSimState(charState)
        self._terrainStartX = x
        self.setTerrainData(terrain)
        
    def getCharacterSimState(self):
        # add angular velocity
        pos = self._obstacle.getPosition()
        angularVel = self._obstacle.getAngularVel()
        #add rotation
        rot = list(self._obstacle.getQuaternion())
        vel = self._obstacle.getLinearVel()
        return (pos, vel, rot, angularVel)
    
    def setCharacterSimState(self, state_):
        # add angular velocity
        (pos, vel, rot, angularVel) = state_
        self._obstacle.setPosition(pos)
        self._obstacle.setAngularVel(angularVel)
        ##add rotation
        self._obstacle.setQuaternion(rot)
        self._obstacle.setLinearVel(vel)
        
    def getStateFromSimState(self, state_):
        """
            Takes the global simulation state and returns the less discriptive 
            state used for learning
        """
        (charState, terrain_, x) = state_
        # print( "Terrain: ", terrain_)
        state__out = self.getState(terrainData_=terrain_, startX_=x, charState_=charState)
        return state__out
    
    def getCharacterState(self, charState=None):
        if ( charState == None ):
            charState=self.getCharacterSimState()
        # add angular velocity
        (pos, vel, rot, angularVel) = charState
        angularVel = list(angularVel)
        #add rotation
        rot = list(rot)
        angularVel.extend(rot)
        # vel = self._obstacle.getLinearVel()
        angularVel.append(vel[0])
        return angularVel
    
    
    def getState(self, terrainData_=None, startX_=None, charState_=None):
        if ( terrainData_ == None ):
            terrainData_ = self._terrainData
        if ( startX_ == None ):
            startX_ = self._terrainStartX
        if ( charState_ == None ):
            charState_ = self.getCharacterSimState()
        """ get the next self._num_points points"""
        (pos, vel, rot, angularVel) = charState_
        charState = self.getCharacterState(charState=charState_)
        num_extra_feature=1
        ## Get the index of the next terrain sample
        start = self.getTerrainIndex()
        ## Is that sample + terrain_state_size beyond the current terrain extent
        if (start+self._num_points+num_extra_feature >= (len(terrainData_))):
            # print ("State not big enough ", len(terrainData_))
            if (self._validating):
                self.generateValidationTerrain(0)
            else:
                self.generateTerrain()
        start = self.getTerrainIndex()
        assert start+self._num_points+num_extra_feature < (len(terrainData_)), "Ball is exceeding terrain length %r after %r actions" % (start+self._num_points+num_extra_feature-1, self._state_num)
        # print ("Terrain Data: ", terrainData_)
        state=np.zeros((self._num_points+num_extra_feature+len(charState)))
        if pos[0] < 0: #something bad happened...
            return state
        else: # good things are going on...
            # state[0:self._num_points] = copy.deepcopy(pos[1]-terrainData_[start:start+self._num_points])
            state[0:self._num_points] = copy.deepcopy(terrainData_[start:start+self._num_points])
            # state = copy.deepcopy(terrainData_[start:start+self._num_points+1])
            # print ("Start: ", start, " State Data: ", state)
            state[self._num_points] = fabs(float(math.floor(start)*self._terrainScale)-(pos[0]-startX_)) # X distance to first sample
            # state[self._num_points+1] = (pos[1]) # current height of character, This was returning huge nagative values... -1.5x+14
            # print ("Dist to next point: ", state[len(state)-1])
            
            # add character State
            state[self._num_points+num_extra_feature:self._num_points+num_extra_feature+len(charState)] = charState
        
        return state
 

if __name__ == '__main__':
    import json
    _settings={}
    # game = BallGame2D(settings)
    if (len(sys.argv)) > 1:
        _settings=json.load(open(sys.argv[1]))
        print (json.dumps(_settings))
        _settings['render']=True
        game = BallGame2D(_settings)
    else:
        _settings['render']=True
        game = BallGame2D(_settings)
    game.init()
    for j in range(100):
        # game.generateEnvironmentSample()
        game.generateValidationEnvironmentSample(j)
        print ("Starting new epoch")
        game.initEpoch()
        i=0
        while not game.endOfEpoch():
        # for i in range(50):
            # state = game.getState()
            
            # action = model.predict(state)
            _action = ((np.random.random([1]))[0] * 2.0) - 1.0
            action = [_action,4.0]
            _action = ((np.random.random([1]))[0] * 2.5) + 2.5
            action[1] = _action 
            state = game.getState()
            pos = game._obstacle.getPosition()
            # drawTerrain(state, pos[0], translateY=0.0, colour=(0.6, 0.6, 0.9, 1.0))
            # print ("State: " + str(state[-8:]))
            # print ("character State: " + str(game.getCharacterState()))
            # print ("rot Vel: " + str(game._obstacle.getQuaternion()))
            
            # print (state)
            
            game.visualizeState(state[:len(state)-1], action, state[_settings['num_terrain_samples']])
            game.visualizeAction(action)
            reward = game.actContinuous(action)
            
            if (game.agentHasFallen()):
                print (" *****Agent fell in a hole")
            if (game.hitWall()):
                print ("******Agent has hit a wall")
            if ( reward < 0.00001 ):
                print("******Agent has 0 reward?")
            
            if ( (not (game.agentHasFallen() or game.hitWall())) and (reward < 0.00001) ):
                print ("*** This bounce game is wrong...")
            
            print ("Reward: " + str(reward) + " on action: " + str(i) + " Enof of episode: ", game.endOfEpoch())
            
            # print ("Number of geoms in space: ", game._space.getNumGeoms())
            # print ("Random rotation matrix", list(np.reshape(rand_rotation_matrix(), (1,9))[0]))
            i=i+1
            game._lasttime = time.time()
            
    game.finish()
