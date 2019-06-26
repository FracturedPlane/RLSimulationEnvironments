
import pybullet_data
import os
import time
from rlsimenv.EnvWrapper import ActionSpace
from rlsimenv.Environment import Environment, clampValue
from rlsimenv.PyBulletUtil import *

class BayesianSupriseDisk(PyBulletEnv):
    """
        The HLC is the first agent
        The LLC is the second agent
    """
    
    def __init__(self, settings):
        super(BayesianSupriseDisk,self).__init__(settings)
        
        self.parts = None
        self.objects = []
        self.jdict = None
        self.ordered_joints = None
        self.agent = None
    
        
        self._state_bounds = self._game_settings['state_bounds']
        self._action_bounds = self._game_settings['action_bounds']
        self._action_length = len(self._action_bounds[0])
        
        self._llc_target = [1.0, 0, 0]
        
        
        # ob_low = [0] * len(self.getEnv().getObservationSpaceSize()
        # ob_high = [1] * self.getEnv().getObservationSpaceSize() 
        # observation_space = [ob_low, ob_high]
        # self._observation_space = ActionSpace(observation_space)
        self._action_space = ActionSpace(self._game_settings['action_bounds'])
        self._map_area = self._game_settings['map_size']
        self._reach_goal_threshold = 1.0
        
        self._vel_bounds = [[-2.0, -2.0, -0.00001],
                            [ 2.0,  2.0,  0.00001]]
        
        self._pos_bounds = [[-self._map_area, -self._map_area,  0.499999],
                            [ self._map_area,  self._map_area,  0.50001]]
        
        self._ran = 0.0
        
    def getActionSpaceSize(self):
        return self._action_length
    
    def getObservationSpaceSize(self):
        return self._state_length

    def display(self):
        pass
    
    def init(self):
        
        super(BayesianSupriseDisk,self).init()
        
        RLSIMENV_PATH = os.environ['RLSIMENV_PATH']
        cubeStartPos = [0,0,0.5]
        cubeStartOrientation = self._p.getQuaternionFromEuler([0.,0,0])
        ### For colissions to work one object should not be fixed
        self._agent = self._p.loadURDF(RLSIMENV_PATH + "/rlsimenv/data/disk.urdf",
        # self._agent = self._p.loadURDF("sphere2red.urdf",
                    [0.0,0.0,0.5],
                    cubeStartOrientation,
                    useFixedBase=0) 
        RLSIMENV_PATH = os.environ['RLSIMENV_PATH']
        
        lo = [-1.0 , -1.0]
        hi = [ 1.0 , 1.0]
        self._action_bounds = np.array([lo, hi])
        
        self._blocks = []
        self._num_blocks=0
        if ("num_blocks" in self._game_settings):
            self._num_blocks = self._game_settings["num_blocks"] 
        for i in range(self._num_blocks):
            blockId = self._p.loadURDF(RLSIMENV_PATH + "/rlsimenv/data/cube2.urdf",
                    [2.0,2.0,0.5],
                    cubeStartOrientation,
                    useFixedBase=1) 
            self._blocks.append(blockId)
        
        self._jointIds = []
        
        lo = [0.0 for l in self.getObservation()[0]]
        hi = [1.0 for l in self.getObservation()[0]]
        state_bounds_llc = [lo, hi]
        state_bounds = state_bounds_llc
        
        print ("NavGameHRL2D state bounds: ", state_bounds)
        self._game_settings['state_bounds'] = [lo, hi]
        self._state_length = len(self._game_settings['state_bounds'][0])
        print ("self._state_length: ", self._state_length)
        self._observation_space = ActionSpace(self._game_settings['state_bounds'])
        
        self._last_state = self.getState()
        self._last_pose = self._p.getBasePositionAndOrientation(self._agent)[0]
        

    def reset(self):
        self.initEpoch()
        return self.getObservation()
    
    def initEpoch(self):
        import numpy as np
        x = (np.random.rand()-0.5) * self._map_area * 2.0
        y = (np.random.rand()-0.5) * self._map_area * 2.0
        self._p.resetBasePositionAndOrientation(self._agent, [x,y,0.5], self._p.getQuaternionFromEuler([0.,0,0]))
        x = (np.random.rand()-0.5) *  2.0
        y = (np.random.rand()-0.5) *  2.0
        self._p.resetBaseVelocity(self._agent, [x,y,0], [0,0,0])
        
        if ("ignore_hlc_actions" in self._game_settings
            and (self._game_settings["ignore_hlc_actions"] == True)):
            self._ran = 0.6 ## Ignore HLC action and have env generate them if > 0.5.
        else:
            self._ran = 0.4 ## Ignore HLC action and have env generate them if > 0.5.
        
        ### Reset obstacles
        for i in range(self._num_blocks):
            x = (np.random.rand()-0.5) * self._map_area * 2.0
            y = (np.random.rand()-0.5) * self._map_area * 2.0
            self._p.resetBasePositionAndOrientation(self._blocks[i], [x,y,0.5], self._p.getQuaternionFromEuler([0.,0,0]))
            self._p.resetBaseVelocity(self._blocks[i], [0,0,0], [0,0,0]) 
            
    def setLLC(self, llc):
        self._llc = llc
        
    def getObservation(self):
        import numpy as np
        out = []
        localMap = self.getlocalMapObservation(pos=(0,0,0))
        out.extend(localMap)
        data = self.getRobotPose()
        out.extend(data)
        ### linear vel
        out = [np.array(out)]
        self._last_state = np.array(out)
        self._last_pose = np.array(self._p.getBasePositionAndOrientation(self._agent)[0])
        return self._last_state
    
    def getState(self):
        return self.getObservation()
    
    def computeReward(self, state=None):
        """
            
        """
        import numpy as np
        active_blocks = 0
        for box_id in self._blocks:
            pos = np.array(self._p.getBasePositionAndOrientation(box_id)[0])
            if pos[2] > 0.0: ### Above ground
                active_blocks = active_blocks + 1
        
        rewards = [-active_blocks]
        return rewards
        
        
        
    def getTargetDirection(self):
        ### raycast around the area of the agent
        import numpy as np
        
        pos = np.array(self._p.getBasePositionAndOrientation(self._agent)[0])
        posT = np.array(self._p.getBasePositionAndOrientation(self._target)[0])
        goalDirection = posT-pos
        goalDirection = goalDirection / np.sqrt((goalDirection*goalDirection).sum(axis=0))
        return goalDirection
    
    def getlocalMapObservation(self, pos=None):
        ### raycast around the area of the agent
        """
            For now this includes the agent in the center of the computation
        """
        import numpy as np
        if (pos is None):
            pos = self._p.getBasePositionAndOrientation(self._agent)[0]
        ### number of samples
        size = 64
        ### width of box
        dimensions = 25.0
        toRays = []
        for i in range(0,size):
            for j in range(0,size):
                toRays.append([(1.0/(size * 1.0))*i*dimensions,(1.0/(size * 1.0))*j*dimensions,-1])
        assert (len(toRays) == (size*size))
        toRays = np.array(toRays)
        ### Adjust to put agent in middle of map
        toRays = toRays + pos - np.array([dimensions/2.0, dimensions/2.0, 0])
        fromRays = toRays + np.array([0,0,5])
        rayResults = self._p.rayTestBatch(fromRays, toRays)
        intersections = [ray[0] for ray in rayResults]
        # print ("intersections: ", intersections)
        
        for ray in range(len(intersections)):
            if (intersections[ray] in [self._agent]):
                # print ("bad index: ", ray)
                ### Remove agent from vision
                # print ("Hit Agent")
                
                intersections[ray] = 1
        
        intersections = np.array(np.greater(intersections, 0.1), dtype="int")
        return intersections
    
    def getVisualState(self):
        return self.getlocalMapObservation(pos=(0,0,0))
    
    def updateAction(self, action):
        import numpy as np
        """
            action[0] == hlc action
            action[1] == llc action
        """
        # print ("action: ", action[0])
        self._p.resetBaseVelocity(self._agent, linearVelocity=[action[0][0], action[0][1], 0], angularVelocity=[0,0,0])
        # super(BayesianSupriseDisk,self).updateAction(action_)
        vel = self._p.getBaseVelocity(self._agent)[0]
        # if (self._ran > 0.5): ### Only Do HLC training half the time.
        # print ("New vel: ", vel)
        
    def update(self):
        import numpy as np
        pos = np.array(self._p.getBasePositionAndOrientation(self._agent)[0])
        vel = np.array(self._p.getBaseVelocity(self._agent)[0])
        # print ("vel: ", vel)
        pos =  pos + (vel*self._dt)
        pos[2] = 0.5
        ### Need to do this so the intersections are updated
        self._p.stepSimulation()
        self._p.resetBasePositionAndOrientation(self._agent, pos, self._p.getQuaternionFromEuler([0.,0,0]))
        ### This must be after setting position, because setting the position removes the velocity
        self._p.resetBaseVelocity(self._agent, linearVelocity=vel, angularVelocity=[0,0,0])
        
        for box_id in self._blocks:
            contacts = self._p.getContactPoints(self._agent, box_id)
            # print ("box ", box_id, " contacts: ", contacts)
            if len(contacts) > 0:
                ### Push block under ground
                pos = np.array(self._p.getBasePositionAndOrientation(box_id)[0])
                pos[2] = -5
                self._p.resetBasePositionAndOrientation(box_id, pos, self._p.getQuaternionFromEuler([0.,0,0]))
        
        reward = self.computeReward(state=None)
        # print("reward: ", reward)
        self._reward = reward
        
    def agentHasFallen(self):
        return self.endOfEpoch()
    
    def endOfEpoch(self):
        import numpy as np
        
        return False
        
    def setRandomSeed(self, seed):
        import numpy as np
        """
            Set the random seed for the simulator
            This is helpful if you are running many simulations in parallel you don't
            want them to be producing the same results if they all init their random number 
            generator the same.
        """
        # random.seed(seed)
        np.random.seed(seed)

if __name__ == "__main__":
    import numpy as np
    settings = {"state_bounds": [[0],[1]],
                "action_bounds": [[0],[1]],
                "render": True,
                "map_size": 10.0,
                "control_substeps": 1,
                "physics_timestep": 0.02,
                "num_blocks": 15}
    
    sim = BayesianSupriseDisk(settings)
    sim.init()
    
    # action = np.array([[0.5, 0.5], [-0.5, -0.5]])
    print ("sim._action_bounds: ", sim._action_bounds)
    action = np.mean(sim._action_bounds, axis=0)
    import time
    for i in range(10000):
        if (i % 100 == 0):
            sim.reset()
        sim.updateAction([[-1.1232,1.534534]])
        sim.update()
        ob = sim.getObservation()
        reward = sim.computeReward()
        time.sleep(1/30.0)
        print ("Reward: ", reward)
        print ("od: ", ob)
        
