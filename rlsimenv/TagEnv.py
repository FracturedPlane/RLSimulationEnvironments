
import gym
import math
import numpy as np
import os
import pdb
import pybullet
import pybullet_data

from rlsimenv.Environment import Environment
import rlsimenv.class_util as classu
import rlsimenv.math_util as mathu

THIS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(THIS_DIR, "data")

class TagEnv(Environment):
    """Implements gym.Env"""
    count = 0

    @classu.hidden_member_initialize
    def __init__(self,
                 max_steps=500,
                 seed=1234,
                 gui=False,
                 map_width=3,
                 chamber_fraction=1/3,
                 observation_height=7,
                 iters=2000,
                 render_shape=(128*2, 128*2, 3),
                 observation_shape=(10, 10, 1),
                 obs_fov=45,
                 obs_aspect=1.0,
                 obs_nearplane=0.01,
                 obs_farplane=100,
                 reset_upon_touch=False,
                 touch_thresh=1.,
                 n_particles=2,
                 observation_stack=2,
                 flat_obs=True,
                 grayscale=True,
                 fixed_view=True
                 ):
        super(TagEnv, self).__init__()
        
        from collections import deque 
        
        self._GRAVITY = -9.8
        self._dt = 1/200.0
        self.sim_steps = 5
        # Temp. Doesn't currently make sense if smaller.
        assert(map_width >= 2.)
        self._map_area = self._map_width ** 2
        # This constant doesn't affect the created width.
        self._cube_width = 1.

        self.dt = self._dt
        self._game_settings = {"include_egocentric_vision": True}
        # self.action_space = gym.spaces.Box(low=np.array([-1.2, -1.2, 0]), high=np.array([1.2,1.2,1]))
        
        if gui:
            # self._object.setPosition([self._x[self._step], self._y[self._step], 0.0] )
            self._physicsClient = pybullet.connect(pybullet.GUI)
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
        
        cubeStartPos = [0, 0, 0.5]
        cubeStartOrientation = pybullet.getQuaternionFromEuler([0.,0,0])
        # These exist as part of the pybullet installation.
        self._demon = pybullet.loadURDF(DATA_DIR + "/sphere2_yellow.urdf", cubeStartPos, cubeStartOrientation, useFixedBase=0)

        self._particles = []
        self._particle_states = []
        for _ in range(self._n_particles):
            self._particles.append(pybullet.loadURDF(DATA_DIR + "/sphere2_red2.urdf", cubeStartPos, cubeStartOrientation, useFixedBase=0))
            self._particle_states.append({"fixed": False, "drag": 1.0})
            pybullet.setCollisionFilterPair(self._demon, self._particles[-1], -1, -1, enableCollision=0)
            
        self._boxes = []
        for _ in range(0):
            self._boxes.append(pybullet.loadURDF(DATA_DIR + "/cube2.urdf", cubeStartPos, cubeStartOrientation, useFixedBase=0))
            pybullet.setCollisionFilterPair(self._demon, self._boxes[-1], -1, -1, enableCollision=0)
            pybullet.changeVisualShape(self._boxes[-1], -1, rgbaColor=[0.2, 0.2, 0.8, 1.0])
            
        pybullet.setAdditionalSearchPath(RLSIMENV_PATH + '/rlsimenv/data')
        
        # Add walls
        self._blocks = []

        self._wall_heights = [0.5, 1.5]
        self.action_space = gym.spaces.Box(low=np.array([-2.5, -2.5, -5]), high=np.array([2.5, 2.5, 5])) #self._wall_heights[-1]]))
        # self._wall_heights = (0.5, 1.5)

        x_full_right = (1+self._chamber_fraction) * self._map_width + self._cube_width
        x_inner_right = self._map_width

        for z in self._wall_heights:

            cube_locations = []

            # Right walls
            # cube_locations.extend([[self._map_width, y, z] for y in range(-self._map_width, -2)])
            # cube_locations.extend([[self._map_width, y, z] for y in range(2, self._map_width)])

            # Right wall
            cube_locations.extend([[self._map_width, y, z] for y in np.arange(-self._map_width, self._map_width)])
            # Left wall
            cube_locations.extend([[-self._map_width, y, z] for y in range(-self._map_width, self._map_width)])
            # Top Wall
            cube_locations.extend([[x, self._map_width, z] for x in np.arange(-self._map_width, self._map_width)])
            # Bottom Wall
            cube_locations.extend([[x, -self._map_width, z] for x in np.arange(-self._map_width, self._map_width)])
            
            # Add small room
            # Add Right wall
            # "Small" small room
            # cube_locations.extend([[self._map_width+(self._map_width//2), y, z] for y in range(-self._map_width//2, self._map_width//2)])
            # cube_locations.extend([[self._map_width, y, z] for y in range(-self._map_width, self._map_width)])
            
            # Top wall 
            # cube_locations.extend([[x, self._map_width//2, z] for x in range(self._map_width, self._map_width+(self._map_width//2))])
            # Bottom wall 
            # cube_locations.extend([[x, -self._map_width//2, z] for x in range(self._map_width, self._map_width+(self._map_width//2))])
            
            # print ("cube_locations: ", cube_locations)
            
            for loc in cube_locations:
                blockId = pybullet.loadURDF(DATA_DIR + "/cube2.urdf", loc, cubeStartOrientation, useFixedBase=1) 
                self._blocks.append(blockId)
            

        ### We can get away with a single box.
        self._roof = [] 
        self._roof.append(pybullet.loadURDF(DATA_DIR + "/cube2.urdf",
                                            [1,0,7.0],
                                            cubeStartOrientation,
                                            useFixedBase=1,
                                            globalScaling=10))
        pybullet.changeVisualShape(self._roof[-1], -1, rgbaColor=[.4, .4, .4, 0.0])

        for body in self._particles + self._blocks + self._roof:
            pybullet.changeDynamics(body,
                                    -1,
                                    rollingFriction=0.,
                                    spinningFriction=0.0,
                                    lateralFriction=0.0,
                                    linearDamping=0.0,
                                    angularDamping=0.0,
                                    restitution=1.0,
                                    maxJointVelocity=10)
        
        # disable the default velocity motors 
        #and set some position control with small force to emulate joint friction/return to a rest pose
        jointFrictionForce = 1
        for joint in range(pybullet.getNumJoints(self._demon)):
            pybullet.setJointMotorControl2(self._demon,joint,pybullet.POSITION_CONTROL,force=jointFrictionForce)
        
        #for i in range(10000):     
        #     pybullet.setJointMotorControl2(botId, 1, pybullet.TORQUE_CONTROL, force=1098.0)
        #     pybullet.stepSimulation()
        #import ipdb
        #ipdb.set_trace()
        pybullet.setRealTimeSimulation(1)
        
        observation_stack
        self._obs_stack = [[0],[1]]
        # lo = self.getObservation()["pixels"] * 0.0
        # hi = lo + 1.0
        lo = np.zeros((np.prod(self._observation_shape)* self._observation_stack))
        hi = np.ones((np.prod(self._observation_shape) * self._observation_stack))

        self._game_settings['state_bounds'] = [lo, hi]
        
        self._obs_stack = deque() 
        for _ in range(self._observation_stack):
            self._obs_stack.append(np.zeros(np.prod(self._observation_shape))) 
        # self._obs_stack = [lo] * self._observation_stack
        
        # self._observation_space = ActionSpace(self._game_settings['state_bounds'])
        if (self._flat_obs):
            self.observation_space = gym.spaces.Box(low=lo, high=hi)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=self._observation_shape)

    def getNumAgents(self):
        return 1
    
    def display(self):
        pass
    
    def getViewData(self):
        com_p, com_o = pybullet.getBasePositionAndOrientation(self._demon)
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
            cameraEyePosition=[1.5, 0, self._observation_height],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 1, 0])
        projection_matrix = pybullet.computeProjectionMatrixFOV(
            self._obs_fov, self._obs_aspect, self._obs_nearplane, self._obs_farplane)
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
        x = (np.random.rand()-0.5) * (self._map_width - 1) * 2.0
        y = (np.random.rand()-0.5) * (self._map_width - 1) * 2.0
        pybullet.resetBasePositionAndOrientation(self._demon, [x,y,0.5], pybullet.getQuaternionFromEuler([0.,0,0]))
        pybullet.resetBaseVelocity(self._demon, [0,0,0], [0,0,0])
        
        x = (np.random.rand()-0.5) * (self._map_width - 1) * 2.0
        y = (np.random.rand()-0.5) * (self._map_width - 1) * 2.0

        for particle, particle_state in zip(self._particles, self._particle_states):
            x = (np.random.rand()-0.5) * (self._map_width - 1) * 2.0
            y = (np.random.rand()-0.5) * (self._map_width - 1) * 2.0
            pybullet.resetBasePositionAndOrientation(particle, [x,y,0.5], pybullet.getQuaternionFromEuler([0.,0,0]))
            initial_vel = np.random.normal(loc=0.5, size=(3,), scale=1.)
            initial_vel[-1] = 0.
            pybullet.resetBaseVelocity(particle, linearVelocity=initial_vel)
            particle_state["fixed"] = False
            particle_state["drag"] = 1.0
        for particle in self._boxes:
            x = (np.random.rand()-0.5) * (self._map_width - 1) * 2.0
            y = (np.random.rand()-0.5) * (self._map_width - 1) * 2.0
            pybullet.resetBasePositionAndOrientation(particle, [x,y,0.5], pybullet.getQuaternionFromEuler([0.,0,0]))
            # pybullet.resetBaseVelocity(particle, [0,0,0], [0,0,0])
        
        # Reset obstacles
        """
        for i in range(len(self._blocks)):
            x = (np.random.rand()-0.5) * self._map_width * 2.0
            y = (np.random.rand()-0.5) * self._map_width * 2.0
            p.resetBasePositionAndOrientation(self._blocks[i], [x,y,0.5], p.getQuaternionFromEuler([0.,0,0]))
            p.resetBaseVelocity(self._blocks[i], [0,0,0], [0,0,0]) 
        """
        return self.getObservation()
    
    def getObservation(self):
        
        if (self._flat_obs):
            obs = np.array([np.array(self._obs_stack).flatten()]) / 255.0 ## Normalize to [0,1]
        else:
            obs = np.array(self.getlocalMapObservation()) / 255.0 ## Normalize to [0,1]
        # print ("obs 1: ", obs)
        # obs = np.array(self.getlocalMapObservation()).flatten()
        # obs = np.array(self.getlocalMapObservation()).flatten()
        # print ("obs 2: ", obs)
        # print ("self._obs_stack: ", self._obs_stack)
        return obs
    
    def getState(self):
        return self.getObservation()

    def _get_target(self):
        return self._particles[0]
    
    def computeReward(self, state=None):
        
        pos = np.array(pybullet.getBasePositionAndOrientation(self._demon)[0])

        particle = self._get_target()
        posT = np.array(pybullet.getBasePositionAndOrientation(particle)[0])
        
        goalDirection = posT-pos
        goalDistance = np.sqrt((goalDirection*goalDirection).sum(axis=0))
        goalDir = self.getTargetDirection()
        agentVel = np.array(pybullet.getBaseVelocity(self._demon)[0])
        velDiff = goalDir - agentVel
        diffMag = np.sqrt((velDiff*velDiff).sum(axis=0))
        # heading towards goal
        reward = np.exp((diffMag*diffMag) * -2.0) + np.exp((goalDistance*goalDistance) * -2.0)
        """
        if (goalDistance < 1.5):
            # Reached goal
            reward = reward + self._map_width
        """
        # Check contacts with obstacles
        """
        for box_id in self._blocks:
            contacts = p.getContactPoints(self._agent, box_id)
            # print ("contacts: ", contacts)
            if len(contacts) > 0:
                reward = reward + -self._map_width
                break
        """
        return reward
        
    def getTargetDirection(self):
        # raycast around the area of the agent
        
        pos = np.array(pybullet.getBasePositionAndOrientation(self._demon)[0])
        particle = self._get_target()
        posT = np.array(pybullet.getBasePositionAndOrientation(particle)[0])
        goalDirection = posT-pos
        goalDirection = goalDirection / np.sqrt((goalDirection*goalDirection).sum(axis=0))
        return goalDirection
    
    def getlocalMapObservation(self):
        # raycast around the area of the agent
        """
            For now this includes the agent in the center of the computation
        """
        
        com_p, com_o = pybullet.getBasePositionAndOrientation(self._demon)
        rot_matrix = pybullet.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        # init_camera_vector = (0, 0, -1) # z-axis
        # init_up_vector = (0, 1, 0) # y-axis
        # Rotated vectors
        # camera_vector = rot_matrix.dot(init_camera_vector)
        # up_vector = rot_matrix.dot(init_up_vector)
        # view_matrix = p.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
        if (self._fixed_view):
            x = 1.5; y = 0
        else:
            x = com_p[0]; y = com_p[1]
        view_matrix = pybullet.computeViewMatrix(cameraEyePosition=[x, y, self._observation_height],
                                                 cameraTargetPosition=[x, y, 0],
                                                 cameraUpVector=[0, 1, 0])
        projection_matrix = pybullet.computeProjectionMatrixFOV(
            self._obs_fov, self._obs_aspect, self._obs_nearplane, self._obs_farplane)
        # img = pybullet.getCameraImage(1000, 1000, view_matrix)
        (w,y,img,depth,segment) = pybullet.getCameraImage(
            width=self._observation_shape[0],
            height=self._observation_shape[1], 
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix)
        # print (img)
        # Don't want alpha channel
        img = img[..., :3]
        
        if (self._grayscale): ### Convert to Gray scale
            img = np.mean(img, axis=-1)
        return img
    
    def updateAction(self, action):
        # action[-1] = min(max(action[-1], -0.01), self._wall_heights[-1] - (self._wall_heights[-1] - self._wall_heights[-2])/2.)
        # pdb.set_trace()
        action = np.asarray(action)
        action = np.minimum(np.maximum(action, self.action_space.low), self.action_space.high).tolist()
        pos = np.array(pybullet.getBasePositionAndOrientation(self._demon)[0])
        # vel = np.array(pybullet.getBaseVelocity(self._demon)[0])
        
        # Box update if agent is close enough.
        # Moving the closest box
        for particle, particle_state in zip(self._particles, self._particle_states):
            pos_d = np.array(pybullet.getBasePositionAndOrientation(particle)[0])
            vel_d = np.array(pybullet.getBaseVelocity(particle)[0])
            # Fast growth rate
            diff = pos - pos_d
            ### Don't need the square root if you just square the threshold
            dist = (diff*diff).sum(axis=0)
            if (dist < (1.0*1.0)):
                ## [0,1]
                sig_action = mathu.genlogistic_function(action[2], b=1, a=-1.0, k=0.0) + 1
                # pybullet.resetBasePositionAndOrientation(particle, pos_d, pybullet.getQuaternionFromEuler([0.,0,0]))
                ## Only move the box proportional to how strongly the agent grabs it.
                particle_state["drag"] = sig_action
                if (sig_action < 0.2):
                    # print ("Agent fixed")
                    particle_state["fixed"] = True
                pybullet.resetBaseVelocity(particle, linearVelocity=vel_d*particle_state["drag"])
                    
            
        # apply delta position change.
        action = np.array([action[0], action[1], 0])
        base_vel = pybullet.getBaseVelocity(self._demon)[0]
        updated_vel = base_vel + action
        pybullet.resetBaseVelocity(self._demon, linearVelocity=updated_vel)
        # vel = pybullet.getBaseVelocity(self._agent)[0]
        
    def update(self):
        
        pos = np.array(pybullet.getBasePositionAndOrientation(self._demon)[0])
        vel = np.array(pybullet.getBaseVelocity(self._demon)[0])
        pos = pos + (vel*self._dt)
        pos[2] = 0.5

        # Need to do this so the intersections are computed
        for _ in range(self.sim_steps):
            for particle, particle_state in zip(self._particles, self._particle_states):
                pos, ori = [np.array(_) for _ in pybullet.getBasePositionAndOrientation(particle)]
                lin_vel, ang_vel = pybullet.getBaseVelocity(particle)
                pos[-1] = self._wall_heights[0]
                pybullet.resetBasePositionAndOrientation(particle, posObj=pos, ornObj=ori)
                # noise = np.random.normal(loc=0,scale=0.1,size=3)
                pybullet.resetBaseVelocity(particle, np.array(lin_vel)*particle_state["drag"], np.array(ang_vel)*particle_state["drag"])
            pybullet.stepSimulation()

        for particle, particle_state in zip(self._particles, self._particle_states):
            target_base_vel = pybullet.getBaseVelocity(particle)[0]
            updated_vel = target_base_vel + np.random.normal(loc=0., size=(3,), scale=1.0)
            updated_vel[-1] = 0.
            pybullet.resetBaseVelocity(particle, linearVelocity=updated_vel*particle_state["drag"])
        
        reward = self.computeReward(state=None)
        
        ### Update observation Stack
        self._obs_stack.popleft()
        self._obs_stack.append(np.array(self.getlocalMapObservation()).flatten())
        self.__reward = reward

    def put_on_ground(self, agent):
        agent_pos, agent_ori = map(np.array, pybullet.getBasePositionAndOrientation(agent))
        agent_pos[-1] = 0.6
        pybullet.resetBasePositionAndOrientation(agent, agent_pos, agent_ori)
        
    def finish(self):
        pass
        
    def calcReward(self):
        return self.__reward
        
    def agentHasFallen(self):
        if self._reset_upon_touch: return self.endOfEpoch()
        else: return False
    
    def endOfEpoch(self):
        pos = np.array(pybullet.getBasePositionAndOrientation(self._demon)[0])

        particle = self._get_target()
        posT = np.array(pybullet.getBasePositionAndOrientation(particle)[0])
        goalDirection = posT-pos
        goalDistance = np.sqrt((goalDirection*goalDirection).sum(axis=0))
        if (goalDistance < self._touch_thresh
            or (pos[0] > self._map_width)
            or (pos[1] > self._map_width)
            or (pos[0] < -self._map_width)
            or (pos[1] < -self._map_width)):
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
    
    def getFullViewData(self): 
        return self.render(mode="rgb_array")

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

class TagEnvPartiallyObserved(TagEnv):
    def __init__(self, **kwargs):
        kwargs.update({'render_shape':(128, 128, 3),
                       'observation_shape':(64, 64, 3),
                       'map_width':6,
                       'observation_height':10,
                       'reset_upon_touch': False})
        super().__init__(**kwargs)

class TagEnvBoxesFullyObserved(TagEnv):
    def __init__(self, **kwargs):
        kwargs.update({'render_shape':(128, 128, 3),
                       'observation_shape':(64, 64, 3),
                       'map_width':4,
                       'observation_height':15,
                       'reset_upon_touch': False})
        super().__init__(**kwargs)
