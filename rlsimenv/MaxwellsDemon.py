
import gym
import numpy as np
import os
import pybullet
import pybullet_data

from rlsimenv.Environment import Environment

THIS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(THIS_DIR, "data")

class MaxwellsDemonEnv(Environment):
    """Implements gym.Env"""
    count = 0
    
    def __init__(self, max_steps=256, seed=1234, gui=False):
        super(MaxwellsDemonEnv,self).__init__()
        self._GRAVITY = -9.8
        self._dt = 1/20.0
        self.dt = self._dt
        self._iters = 2000 
        self._map_area = 6
        # self._render_shape = [128, 128, 3]
        self._render_shape = [256, 256, 3]        
        self._observation_shape = [64, 64, 3]
        
        self._game_settings = {"include_egocentric_vision": True}
        self.action_space = gym.spaces.Box(low=np.array([-1.2, -1.2, 0]), high=np.array([1.2,1.2,1]))

        print("gui count", MaxwellsDemonEnv.count)        
        if gui:
            # self._object.setPosition([self._x[self._step], self._y[self._step], 0.0] )
            self._physicsClient = pybullet.connect(pybullet.GUI)
            MaxwellsDemonEnv.count += 1
        else:
            self._physicsClient = pybullet.connect(pybullet.DIRECT)
            
        RLSIMENV_PATH = os.environ['RLSIMENV_PATH']
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.resetSimulation()
        #pybullet.setRealTimeSimulation(True)
        pybullet.setGravity(0,0,self._GRAVITY)
        pybullet.setTimeStep(self._dt)
        # _planeId = pybullet.loadURDF("plane.urdf", )
        pybullet.loadURDF("plane.urdf")
        
        cubeStartPos = [0,0,0.5]
        cubeStartOrientation = pybullet.getQuaternionFromEuler([0.,0,0])
        # TODO where do these URDFs exist?
        self._agent = pybullet.loadURDF("sphere2.urdf", cubeStartPos, cubeStartOrientation, useFixedBase=1) 
        
        self._target = pybullet.loadURDF("sphere2red.urdf", cubeStartPos, cubeStartOrientation, useFixedBase=1)
        
        pybullet.setAdditionalSearchPath(RLSIMENV_PATH + '/rlsimenv/data')
        
        # Add walls
        self._blocks = []
        # Right walls
        cube_locations = [[self._map_area, y, 0.5] for y in range(-self._map_area, -2)]
        cube_locations.extend([[self._map_area, y, 0.5] for y in range(2, self._map_area)])
        
        # Left wall
        cube_locations.extend([[-self._map_area, y, 0.5] for y in range(-self._map_area, self._map_area)])
        # Top Wall
        cube_locations.extend([[y, self._map_area, 0.5] for y in range(-self._map_area, self._map_area)])
        # Bottom Wall
        cube_locations.extend([[y, -self._map_area, 0.5] for y in range(-self._map_area, self._map_area)])
        # Add small room
        # Add Right wall
        cube_locations.extend([[self._map_area+(self._map_area//2), y, 0.5] for y in range(-self._map_area//2, self._map_area//2)])
        # Top wall 
        cube_locations.extend([[y, self._map_area//2, 0.5] for y in range(self._map_area, self._map_area+(self._map_area//2))])
        # Bottom wall 
        cube_locations.extend([[y, -self._map_area//2, 0.5] for y in range(self._map_area, self._map_area+(self._map_area//2))])
        # print ("cube_locations: ", cube_locations)
        for loc in cube_locations:
            blockId = pybullet.loadURDF(DATA_DIR + "/cube2.urdf",
                                        loc,
                                        cubeStartOrientation,
                                        useFixedBase=1) 
            self._blocks.append(blockId)
            
        self._doors = []
        door_locations = [[self._map_area, y, 0] for y in range(-2, 2)]
        for loc in door_locations:
            blockId = pybullet.loadURDF(DATA_DIR + "/cube2.urdf",
                                        loc,
                                        cubeStartOrientation,
                                        useFixedBase=1) 
            self._doors.append(blockId)
        
        # disable the default velocity motors 
        #and set some position control with small force to emulate joint friction/return to a rest pose
        jointFrictionForce = 1
        for joint in range(pybullet.getNumJoints(self._agent)):
            pybullet.setJointMotorControl2(self._agent,joint,pybullet.POSITION_CONTROL,force=jointFrictionForce)
        
        #for i in range(10000):     
        #     pybullet.setJointMotorControl2(botId, 1, pybullet.TORQUE_CONTROL, force=1098.0)
        #     pybullet.stepSimulation()
        #import ipdb
        #ipdb.set_trace()
        pybullet.setRealTimeSimulation(1)
        
        # lo = self.getObservation()["pixels"] * 0.0
        # hi = lo + 1.0
        lo = 0.
        hi = 1.

        self._game_settings['state_bounds'] = [lo, hi]
        
        # self._observation_space = ActionSpace(self._game_settings['state_bounds'])
        # self.observation_space = gym.spaces.Box(low=lo, high=hi, shape=(64,64,3))
        self.observation_space = gym.spaces.Box(low=lo, high=hi, shape=(64,64,3))

    def getNumAgents(self):
        return 1
    
    def display(self):
        pass
    
    def getViewData(self):
        com_p, com_o = pybullet.getBasePositionAndOrientation(self._agent)
        rot_matrix = pybullet.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        # init_camera_vector = (0, 0, -1) # z-axis
        # init_up_vector = (0, 1, 0) # y-axis
        # Rotated vectors
        # camera_vector = rot_matrix.dot(init_camera_vector)
        # up_vector = rot_matrix.dot(init_up_vector)
        # view_matrix = p.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
        view_matrix = pybullet.computeViewMatrix(
            cameraEyePosition=[0, 0, 15],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 1, 0])
        fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
        projection_matrix = pybullet.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)
        # img = p.getCameraImage(1000, 1000, view_matrix)
        (w,y,img,depth,segment) = pybullet.getCameraImage(width=self._render_shape[0],
                                                          height=self._render_shape[1], 
                                                          viewMatrix=view_matrix,
                                                          projectionMatrix=projection_matrix)
        # print (img)
        return img[..., :3]
    
    @property
    def sim(self):
        # Hack to match gym_wrapper interface.
        return self
    
    def init(self):
        pass

    def initEpoch(self):
        return self.reset()
        
    def reset(self):
        self._done = False
        x = (np.random.rand()-0.5) * self._map_area * 2.0
        y = (np.random.rand()-0.5) * self._map_area * 2.0
        pybullet.resetBasePositionAndOrientation(self._agent, [x,y,0.5], pybullet.getQuaternionFromEuler([0.,0,0]))
        pybullet.resetBaseVelocity(self._agent, [0,0,0], [0,0,0])
        
        x = (np.random.rand()-0.5) * self._map_area * 2.0
        y = (np.random.rand()-0.5) * self._map_area * 2.0
        pybullet.resetBasePositionAndOrientation(self._target, [x,y,0.5], pybullet.getQuaternionFromEuler([0.,0,0]))
        pybullet.resetBaseVelocity(self._target, [0,0,0], [0,0,0])
        
        # Reset obstacles
        """
        for i in range(len(self._blocks)):
            x = (np.random.rand()-0.5) * self._map_area * 2.0
            y = (np.random.rand()-0.5) * self._map_area * 2.0
            p.resetBasePositionAndOrientation(self._blocks[i], [x,y,0.5], p.getQuaternionFromEuler([0.,0,0]))
            p.resetBaseVelocity(self._blocks[i], [0,0,0], [0,0,0]) 
        """
        return self.getObservation()
    
    def getObservation(self):
        return np.array(self.getlocalMapObservation())
        # out = {}
        # # out["pixels"] = np.array(self.getlocalMapObservation()).flatten()
        # out["pixels"] = np.array(self.getlocalMapObservation())
        # """
        # data = p.getBaseVelocity(self._agent)
        # # linear vel
        # out.extend([data[0][0], data[0][1]])
        # # angular vel
        # pos = np.array(p.getBasePositionAndOrientation(self._agent)[0])
        # posT = np.array(p.getBasePositionAndOrientation(self._target)[0])
        # goalDirection = posT-pos
        # out.extend([goalDirection[0], goalDirection[1]])
        # out = np.array([np.array(out)])
        # """
        # return out
    
    def getState(self):
        return self.getObservation()
    
    def computeReward(self, state=None):
        
        pos = np.array(pybullet.getBasePositionAndOrientation(self._agent)[0])
        posT = np.array(pybullet.getBasePositionAndOrientation(self._target)[0])
        
        goalDirection = posT-pos
        goalDistance = np.sqrt((goalDirection*goalDirection).sum(axis=0))
        goalDir = self.getTargetDirection()
        agentVel = np.array(pybullet.getBaseVelocity(self._agent)[0])
        velDiff = goalDir - agentVel
        diffMag = np.sqrt((velDiff*velDiff).sum(axis=0))
        # heading towards goal
        reward = np.exp((diffMag*diffMag) * -2.0) + np.exp((goalDistance*goalDistance) * -2.0)
        """
        if (goalDistance < 1.5):
            # Reached goal
            reward = reward + self._map_area
        """
        # Check contacts with obstacles
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
        # raycast around the area of the agent
        
        pos = np.array(pybullet.getBasePositionAndOrientation(self._agent)[0])
        posT = np.array(pybullet.getBasePositionAndOrientation(self._target)[0])
        goalDirection = posT-pos
        goalDirection = goalDirection / np.sqrt((goalDirection*goalDirection).sum(axis=0))
        return goalDirection
    
    def getlocalMapObservation(self):
        # raycast around the area of the agent
        """
            For now this includes the agent in the center of the computation
        """
        
        com_p, com_o = pybullet.getBasePositionAndOrientation(self._agent)
        rot_matrix = pybullet.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        # init_camera_vector = (0, 0, -1) # z-axis
        # init_up_vector = (0, 1, 0) # y-axis
        # Rotated vectors
        # camera_vector = rot_matrix.dot(init_camera_vector)
        # up_vector = rot_matrix.dot(init_up_vector)
        # view_matrix = p.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
        view_matrix = pybullet.computeViewMatrix(cameraEyePosition=[com_p[0], com_p[1], 5],
                                                 cameraTargetPosition=[com_p[0], com_p[1], 0],
                                                 cameraUpVector=[0, 1, 0])
        fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
        projection_matrix = pybullet.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)
        # img = pybullet.getCameraImage(1000, 1000, view_matrix)
        (w,y,img,depth,segment) = pybullet.getCameraImage(
            width=self._observation_shape[0],
            height=self._observation_shape[1], 
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix)
        # print (img)
        # Don't want alpha channel
        img = img[..., :3]
        return img
    
    def updateAction(self, action):
        
        for door in self._doors:
            pos_d = np.array(pybullet.getBasePositionAndOrientation(door)[0])
            pos_d[2] = action[2] - 0.5
            pybullet.resetBasePositionAndOrientation(door, pos_d, pybullet.getQuaternionFromEuler([0.,0,0]))
        # apply delta position change.
        action = np.array([action[0], action[1], 0])
        pybullet.resetBaseVelocity(self._agent, linearVelocity=action, angularVelocity=[0,0,0])
        # vel = pybullet.getBaseVelocity(self._agent)[0]
        
    def update(self):
        
        pos = np.array(pybullet.getBasePositionAndOrientation(self._agent)[0])
        vel = np.array(pybullet.getBaseVelocity(self._agent)[0])
        pos = pos + (vel*self._dt)
        pos[2] = 0.5

        # Need to do this so the intersections are computed
        pybullet.stepSimulation()
        pybullet.resetBasePositionAndOrientation(self._agent, pos, pybullet.getQuaternionFromEuler([0.,0,0]))
        pybullet.resetBaseVelocity(self._agent, linearVelocity=[0,0,0], angularVelocity=[0,0,0])
        
        pos_t = np.array(pybullet.getBasePositionAndOrientation(self._target)[0])
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
            
        pybullet.resetBasePositionAndOrientation(self._target, [x, y, 0.5], pybullet.getQuaternionFromEuler([0.,0,0]))
        pybullet.resetBaseVelocity(self._target, [0,0,0], [0,0,0])
        
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
        
        pos = np.array(pybullet.getBasePositionAndOrientation(self._agent)[0])
        posT = np.array(pybullet.getBasePositionAndOrientation(self._target)[0])
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
        
        """
            Set the random seed for the simulator
            This is helpful if you are running many simulations in parallel you don't
            want them to be producing the same results if they all init their random number 
            generator the same.
        """
        # random.seed(seed)
        np.random.seed(seed)

    def setRandomSeed(self, seed): return self.seed(seed)

    def render(self, mode='rgb_array', **kwargs):
        if mode == 'rgb_array':
            img = self.getViewData()
            return img
        else:
            raise ValueError("Unhandled rendering mode")

    def getVisualState(self):
        return self.render()

    def getImitationVisualState(self):
        """TODO ????

        :returns: 
        :rtype: 

        """
        return self.render()

class MaxwellsDemonEnvWithGUI(MaxwellsDemonEnv):
    def __init__(self, max_steps=256, seed=1234, gui=True):
        super().__init__(max_steps=256, seed=1234, gui=True)
