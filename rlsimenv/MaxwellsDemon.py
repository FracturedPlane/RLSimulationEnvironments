
import gym
import numpy as np
import os
import pybullet as p
import pybullet_data

import time

from rlsimenv.EnvWrapper import ActionSpace
from rlsimenv.Environment import Environment

class MaxwellsDemonEnv(Environment):
    """Implements gym.Env"""
    count = 0
    
    def __init__(self, max_steps=256, seed=1234, gui=False):
        super(MaxwellsDemonEnv,self).__init__()
        self._GRAVITY = -9.8
        self._dt = 1/20.0
        self.dt = self._dt
        self._iters=2000 
        self._map_area=6
        
        self._game_settings = {"include_egocentric_vision": True}
        self.action_space = gym.spaces.Box(low=np.array([-1.2, -1.2, 0]), high=np.array([1.2,1.2,1]))

        print("gui count", MaxwellsDemonEnv.count)        
        if gui:
            # self._object.setPosition([self._x[self._step], self._y[self._step], 0.0] )
            self._physicsClient = p.connect(p.GUI)
            MaxwellsDemonEnv.count += 1
        else:
            self._physicsClient = p.connect(p.DIRECT)
            
        RLSIMENV_PATH = os.environ['RLSIMENV_PATH']
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        #p.setRealTimeSimulation(True)
        p.setGravity(0,0,self._GRAVITY)
        p.setTimeStep(self._dt)
        planeId = p.loadURDF("plane.urdf")
        
        cubeStartPos = [0,0,0.5]
        cubeStartOrientation = p.getQuaternionFromEuler([0.,0,0])
        self._agent = p.loadURDF("sphere2.urdf", cubeStartPos, cubeStartOrientation, useFixedBase=1) 
        
        self._target = p.loadURDF("sphere2red.urdf", cubeStartPos, cubeStartOrientation, useFixedBase=1)
        
        p.setAdditionalSearchPath(RLSIMENV_PATH + '/rlsimenv/data')
        
        #### Add walls
        self._blocks = []
        ### Right walls
        cube_locations = [[self._map_area, y, 0.5] for y in range(-self._map_area, -2)]
        cube_locations.extend([[self._map_area, y, 0.5] for y in range(2, self._map_area)])
        
        
        ### Left wall
        cube_locations.extend([[-self._map_area, y, 0.5] for y in range(-self._map_area, self._map_area)])
        ### Top Wall
        cube_locations.extend([[y, self._map_area, 0.5] for y in range(-self._map_area, self._map_area)])
        ### Bottom Wall
        cube_locations.extend([[y, -self._map_area, 0.5] for y in range(-self._map_area, self._map_area)])
        ### Add small room
        ### Add Right wall
        cube_locations.extend([[self._map_area+(self._map_area//2), y, 0.5] for y in range(-self._map_area//2, self._map_area//2)])
        ### Top wall 
        cube_locations.extend([[y, self._map_area//2, 0.5] for y in range(self._map_area, self._map_area+(self._map_area//2))])
        ### Bottom wall 
        cube_locations.extend([[y, -self._map_area//2, 0.5] for y in range(self._map_area, self._map_area+(self._map_area//2))])
        # print ("cube_locations: ", cube_locations)
        for loc in cube_locations:
            blockId = p.loadURDF("cube2.urdf",
                    loc,
                    cubeStartOrientation,
                    useFixedBase=1) 
            self._blocks.append(blockId)
            
        self._doors = []
        door_locations = [[self._map_area, y, 0] for y in range(-2, 2)]
        for loc in door_locations:
            blockId = p.loadURDF("cube2.urdf",
                    loc,
                    cubeStartOrientation,
                    useFixedBase=1) 
            self._doors.append(blockId)
        
        
        #disable the default velocity motors 
        #and set some position control with small force to emulate joint friction/return to a rest pose
        jointFrictionForce=1
        for joint in range (p.getNumJoints(self._agent)):
            p.setJointMotorControl2(self._agent,joint,p.POSITION_CONTROL,force=jointFrictionForce)
        
        #for i in range(10000):     
        #     p.setJointMotorControl2(botId, 1, p.TORQUE_CONTROL, force=1098.0)
        #     p.stepSimulation()
        #import ipdb
        #ipdb.set_trace()
        p.setRealTimeSimulation(1)
        
        lo = self.getObservation()[0] * 0.0
        hi = lo + 1.0
        self._game_settings['state_bounds'] = [lo, hi]
        self._state_length = len(self._game_settings['state_bounds'][0])
        print ("self._state_length: ", self._state_length)
        # self._observation_space = ActionSpace(self._game_settings['state_bounds'])
        self.observation_space = gym.spaces.Box(low=lo, high=hi)
        self._screen_size=[200,200,3]
        
    def getNumAgents(self):
        return 1
    
    def display(self):
        pass
    
    def render(self, **kwargs):
        img = self.getViewData()
        return img
    
    def getViewData(self):
        (w,y,img,depth,segment) = p.getCameraImage(*self._screen_size)
        # print (img)
        return img
    
    @property
    def sim(self):
        # Hack to match gym_wrapper interface.
        return self
    
    def init(self):
        pass
        
    def reset(self):
        self._done = False
        x = (np.random.rand()-0.5) * self._map_area * 2.0
        y = (np.random.rand()-0.5) * self._map_area * 2.0
        p.resetBasePositionAndOrientation(self._agent, [x,y,0.5], p.getQuaternionFromEuler([0.,0,0]))
        p.resetBaseVelocity(self._agent, [0,0,0], [0,0,0])
        
        x = (np.random.rand()-0.5) * self._map_area * 2.0
        y = (np.random.rand()-0.5) * self._map_area * 2.0
        p.resetBasePositionAndOrientation(self._target, [x,y,0.5], p.getQuaternionFromEuler([0.,0,0]))
        p.resetBaseVelocity(self._target, [0,0,0], [0,0,0])
        
        ### Reset obstacles
        """
        for i in range(len(self._blocks)):
            x = (np.random.rand()-0.5) * self._map_area * 2.0
            y = (np.random.rand()-0.5) * self._map_area * 2.0
            p.resetBasePositionAndOrientation(self._blocks[i], [x,y,0.5], p.getQuaternionFromEuler([0.,0,0]))
            p.resetBaseVelocity(self._blocks[i], [0,0,0], [0,0,0]) 
        """
        return self.getObservation()
    
    def getObservation(self):
        import numpy as np
        out = []
        if self._game_settings.get("include_egocentric_vision", False):
            localMap = self.getlocalMapObservation()
            out.extend(localMap)
        data = p.getBaseVelocity(self._agent)
        ### linear vel
        out.extend([data[0][0], data[0][1]])
        ### angular vel
        pos = np.array(p.getBasePositionAndOrientation(self._agent)[0])
        posT = np.array(p.getBasePositionAndOrientation(self._target)[0])
        goalDirection = posT-pos
        out.extend([goalDirection[0], goalDirection[1]])
        out = np.array([np.array(out)])
        return out
    
    def getState(self):
        return self.getObservation()
    
    def computeReward(self, state=None):
        import numpy as np
        pos = np.array(p.getBasePositionAndOrientation(self._agent)[0])
        posT = np.array(p.getBasePositionAndOrientation(self._target)[0])
        
        
        goalDirection = posT-pos
        goalDistance = np.sqrt((goalDirection*goalDirection).sum(axis=0))
        goalDir = self.getTargetDirection()
        agentVel = np.array(p.getBaseVelocity(self._agent)[0])
        velDiff = goalDir - agentVel
        diffMag = np.sqrt((velDiff*velDiff).sum(axis=0))
        ### heading towards goal
        reward = np.exp((diffMag*diffMag) * -2.0) + np.exp((goalDistance*goalDistance) * -2.0)
        """
        if (goalDistance < 1.5):
            ### Reached goal
            reward = reward + self._map_area
        """
        ### Check contacts with obstacles
        """
        for box_id in self._blocks:
            contacts = p.getContactPoints(self._agent, box_id)
            # print ("contacts: ", contacts)
            if len(contacts) > 0:
                reward = reward + -self._map_area
                break
        """
        return reward
        
    def getTargetDirection(self):
        ### raycast around the area of the agent
        import numpy as np
        
        pos = np.array(p.getBasePositionAndOrientation(self._agent)[0])
        posT = np.array(p.getBasePositionAndOrientation(self._target)[0])
        goalDirection = posT-pos
        goalDirection = goalDirection / np.sqrt((goalDirection*goalDirection).sum(axis=0))
        return goalDirection
    
    def getlocalMapObservation(self):
        ### raycast around the area of the agent
        """
            For now this includes the agent in the center of the computation
        """
        import numpy as np
        
        pos = p.getBasePositionAndOrientation(self._agent)[0]
        ### number of samples
        size = 16
        ### width of box
        dimensions = 8.0
        toRays = []
        for i in range(0,size):
            for j in range(0,size):
                toRays.append([(1.0/(size * 1.0))*i*dimensions,(1.0/(size * 1.0))*j*dimensions,0])
        assert (len(toRays) == (size*size))
        toRays = np.array(toRays)
        ### Adjust to put agent in middle of map
        toRays = toRays + pos - np.array([dimensions/2.0, dimensions/2.0, 0])
        # print ("toRays:", toRays )
        
        fromRays = toRays + np.array([0,0,5])
        rayResults = p.rayTestBatch(fromRays, toRays)
        intersections = [ray[0] for ray in rayResults]
        # print (intersections)
        ### fix intersections that could be goal
        
        for ray in range(len(intersections)):
            if (intersections[ray] in [self._target, self._agent]):
                # print ("bad index: ", ray)
                intersections[ray] = -1
        
        """
        if ( len(bad_indecies) > 0):
            # print ("self._target: ", self._target)
            intersections[bad_indecies] = -1
        """
        intersections = np.array(np.greater(intersections, 0), dtype="int")
        return intersections
    
    def updateAction(self, action):
        import numpy as np
        for door in self._doors:
            pos_d = np.array(p.getBasePositionAndOrientation(door)[0])
            pos_d[2] = action[2] - 0.5
            p.resetBasePositionAndOrientation(door, pos_d, p.getQuaternionFromEuler([0.,0,0]))
        ### apply delta position change.
        action = np.array([action[0], action[1], 0])
        p.resetBaseVelocity(self._agent, linearVelocity=action, angularVelocity=[0,0,0])
        vel = p.getBaseVelocity(self._agent)[0]
        
    def update(self):
        import numpy as np
        pos = np.array(p.getBasePositionAndOrientation(self._agent)[0])
        vel = np.array(p.getBaseVelocity(self._agent)[0])
        pos =  pos + (vel*self._dt)
        pos[2] = 0.5
        ### Need to do this so the intersections are computed
        p.stepSimulation()
        p.resetBasePositionAndOrientation(self._agent, pos, p.getQuaternionFromEuler([0.,0,0]))
        p.resetBaseVelocity(self._agent, linearVelocity=[0,0,0], angularVelocity=[0,0,0])
        
        pos_t = np.array(p.getBasePositionAndOrientation(self._target)[0])
        x = ((np.random.rand()-0.5) * self._map_area * 0.05) + pos_t[0]
        y = ((np.random.rand()-0.5) * self._map_area * 0.05) + pos_t[1]
        if (x > self._map_area):
            x = self._map_area
        if (x < -self._map_area):
            x = -self._map_area
        if (y > self._map_area):
            y = self._map_area
        if (y < -self._map_area):
            y = self._map_area
            
        p.resetBasePositionAndOrientation(self._target, [x, y, 0.5], p.getQuaternionFromEuler([0.,0,0]))
        p.resetBaseVelocity(self._target, [0,0,0], [0,0,0])
        
        # time.sleep(1)
        reward = self.computeReward(state=None)
        # print("reward: ", reward)
        self.__reward = reward
        
    def finish(self):
        pass
        
    def calcReward(self):
        return self.__reward
        
    def agentHasFallen(self):
        return self.endOfEpoch()
    
    def endOfEpoch(self):
        import numpy as np
        
        pos = np.array(p.getBasePositionAndOrientation(self._agent)[0])
        posT = np.array(p.getBasePositionAndOrientation(self._target)[0])
        goalDirection = posT-pos
        goalDistance = np.sqrt((goalDirection*goalDirection).sum(axis=0))
        if (goalDistance < 1.0
            or (pos[0] > self._map_area)
            or (pos[1] > self._map_area)
            or (pos[0] < -self._map_area)
            or (pos[1] < -self._map_area)):
            return True
        else:
            return False
        
    def seed(self, seed):
        import numpy as np
        """
            Set the random seed for the simulator
            This is helpful if you are running many simulations in parallel you don't
            want them to be producing the same results if they all init their random number 
            generator the same.
        """
        # random.seed(seed)
        np.random.seed(seed)

    def render(self, mode='rgb_array', **kwargs):
        if mode == 'rgb_array':
            return np.random.random((64, 64, 3))
        else:
            raise ValueError("Unhandled rendering mode")

class MaxwellsDemonEnvWithGUI(MaxwellsDemonEnv):
    def __init__(self, max_steps=256, seed=1234, gui=True):
        super().__init__(max_steps=256, seed=1234, gui=True)
