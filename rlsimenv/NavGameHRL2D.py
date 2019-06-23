
import pybullet as p
import pybullet_data
import os
import time
from rlsimenv.EnvWrapper import ActionSpace
from rlsimenv.Environment import Environment, clampValue

class NavGameHRL2D(Environment):
    """
        The HLC is the first agent
        The LLC is the second agent
    """
    
    def __init__(self, settings):
        super(NavGameHRL2D,self).__init__(settings)
        self._GRAVITY = -9.8
        self._dt = 1/25.0
        self._iters=2000 
        
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

    def getNumAgents(self):
        return 2
    
    def display(self):
        pass
    
    def init(self):
        
        if (self._game_settings['render']):
            # self._object.setPosition([self._x[self._step], self._y[self._step], 0.0] )
            self._physicsClient = p.connect(p.GUI)
        else:
            self._physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        #p.setRealTimeSimulation(True)
        p.setGravity(0,0,self._GRAVITY)
        p.setTimeStep(self._dt)
        planeId = p.loadURDF("plane.urdf")
        
        cubeStartPos = [0,0,0.5]
        cubeStartOrientation = p.getQuaternionFromEuler([0.,0,0])
        self._agent = p.loadURDF("sphere2.urdf",
                cubeStartPos,
                cubeStartOrientation) 
        
        self._blocks = []
        self._num_blocks=0
        # if ("num_blocks" in self._game_settings):
        #     self._num_blocks = self._game_settings["num_blocks"] 
        for i in range(self._num_blocks):
            blockId = p.loadURDF("cube2.urdf",
                    [2.0,2.0,0.5],
                    cubeStartOrientation,
                    useFixedBase=1) 
            self._blocks.append(blockId)
        
        
        self._target = p.loadURDF("sphere2red.urdf",
                cubeStartPos,
                cubeStartOrientation,
                useFixedBase=1)
        
         
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
        p.setRealTimeSimulation(0)
        
        lo = [0.0 for l in self.getObservation()[0]]
        hi = [1.0 for l in self.getObservation()[0]]
        state_bounds_llc = [lo, hi]
        lo = [0.0 for l in self.getObservation()[1]]
        hi = [1.0 for l in self.getObservation()[1]]
        state_bounds_hlc = [lo, hi]
        state_bounds = [state_bounds_llc, state_bounds_hlc]
        
        print ("NavGameHRL2D state bounds: ", state_bounds)
        self._game_settings['state_bounds'] = [lo, hi]
        self._state_length = len(self._game_settings['state_bounds'][0])
        print ("self._state_length: ", self._state_length)
        self._observation_space = ActionSpace(self._game_settings['state_bounds'])
        
        self._last_state = self.getState()
        self._last_pose = p.getBasePositionAndOrientation(self._agent)[0]
        
    def reset(self):
        self.initEpoch()
        return self.getObservation()
    
    def initEpoch(self):
        import numpy as np
        x = (np.random.rand()-0.5) * self._map_area * 2.0
        y = (np.random.rand()-0.5) * self._map_area * 2.0
        p.resetBasePositionAndOrientation(self._agent, [x,y,0.5], p.getQuaternionFromEuler([0.,0,0]))
        x = (np.random.rand()-0.5) *  2.0
        y = (np.random.rand()-0.5) *  2.0
        p.resetBaseVelocity(self._agent, [x,y,0], [0,0,0])
        
        x = (np.random.rand()-0.5) * self._map_area * 2.0
        y = (np.random.rand()-0.5) * self._map_area * 2.0
        # x = (np.random.rand()-0.5) * 2.0 * 2.0
        # y = (np.random.rand()-0.5) * 2.0 * 2.0
        p.resetBasePositionAndOrientation(self._target, [x,y,0.5], p.getQuaternionFromEuler([0.,0,0]))
        p.resetBaseVelocity(self._target, [0,0,0], [0,0,0])
        
        # self._ran = np.random.rand(1)[0]
        if ("ignore_hlc_actions" in self._game_settings
            and (self._game_settings["ignore_hlc_actions"] == True)):
            self._ran = 0.6 ## Ignore HLC action and have env generate them if > 0.5.
        else:
            self._ran = 0.4 ## Ignore HLC action and have env generate them if > 0.5.
        self._llc_target = [x/self._map_area, y/self._map_area, 0]
        ### Make sure to apply HLC action right away
        self._hlc_timestep = 1000000
        self._hlc_skip = 10
        if ("hlc_timestep" in self._game_settings):
            self._hlc_skip = self._game_settings["hlc_timestep"]
        
        ### Reset obstacles
        for i in range(len(self._blocks)):
            x = (np.random.rand()-0.5) * self._map_area * 2.0
            y = (np.random.rand()-0.5) * self._map_area * 2.0
            p.resetBasePositionAndOrientation(self._blocks[i], [x,y,0.5], p.getQuaternionFromEuler([0.,0,0]))
            p.resetBaseVelocity(self._blocks[i], [0,0,0], [0,0,0]) 
            
    def setLLC(self, llc):
        self._llc = llc
        
    def getObservation(self):
        import numpy as np
        out = []
        out_hlc = []
        if ("include_egocentric_vision" in self._game_settings
            and (self._game_settings["include_egocentric_vision"] == True)):
            localMap = self.getlocalMapObservation()
            out_hlc.extend(localMap)
        data = p.getBaseVelocity(self._agent)
        ### linear vel
        out_hlc.extend([data[0][0], data[0][1]])
        ### angular vel
        # out.extend(data[1])
        pos = np.array(p.getBasePositionAndOrientation(self._agent)[0])
        posT = np.array(p.getBasePositionAndOrientation(self._target)[0])
        goalDirection = posT-pos
        out_hlc.extend([goalDirection[0], goalDirection[1]])
        # out = [np.array([np.array(out)])]
        # out = np.array([np.array(out)])
        # print ("obs: ", np.array(out))
        out_llc = []
        out_llc.extend([data[0][0], data[0][1]])
        ### Relative distance from current LLC state
        # if (self._ran < 0.5):
        # out_llc.extend(np.array(self._llc_target) - np.array(data[0]))
        out_llc.extend(np.array([self._llc_target[0], self._llc_target[1]]))
        # else:
        #     out_llc.extend(np.array(self._llc_target) - pos)
        # print ("out_llc: ", out_llc)
        out.append(np.array(out_hlc))
        out.append(np.array(out_llc))
        self._last_state = np.array(out)
        self._last_pose = np.array(p.getBasePositionAndOrientation(self._agent)[0])
        return self._last_state
    
    def getState(self):
        return self.getObservation()
    
    def computeReward(self, state=None):
        """
            
        """
        import numpy as np
        pos = np.array(p.getBasePositionAndOrientation(self._agent)[0])
        posT = np.array(p.getBasePositionAndOrientation(self._target)[0])
        rewards = []
        # print ("self._llc_target: ", self._llc_target)
        
        goalDirection = posT-pos
        goalDistance = np.sqrt((goalDirection*goalDirection).sum(axis=0))
        goalDir = self.getTargetDirection()
        # goalDir = goalDir / np.sqrt((goalDir*goalDir).sum(axis=0))
        # print ("goalDir: ", goalDir)
        agentVel = np.array(p.getBaseVelocity(self._agent)[0])
        agentDir = agentVel / np.sqrt((agentVel*agentVel).sum(axis=0))
        velDiff = goalDir - agentVel
        diffMag = np.sqrt((velDiff*velDiff).sum(axis=0))
        # agentVel = agentVel / agentSpeed
        # print ("agentVel: ", agentVel)
        # agentSpeedDiff = (1 - agentSpeed)
        ### heading towards goal
        # reward = np.dot(goalDir, agentVel) + np.exp(agentSpeedDiff*agentSpeedDiff * -2.0)
        
        if ( goalDistance < self._reach_goal_threshold ):
            hlc_reward = 2.0
        else:
            hlc_reward = -goalDistance/((self._map_area - -self._map_area)/2.0)
            # hlc_reward = 0
        hlc_reward = np.exp((goalDistance*goalDistance) * -1.0) + np.exp((diffMag*diffMag) * -2.0)
        # hlc_reward = np.exp((diffMag*diffMag) * -2.0)
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
                hlc_reward = hlc_reward + -1
                break
        """
        # print ("self._llc_target: ", self._llc_target)
        # print ("pos: ", pos, " agentVel: ", agentVel)
        """
        if (self._ran < 0.5):
            llc_dir = np.array([self._llc_target[0], self._llc_target[1], 0])
            ### normalize
            # llc_dir = llc_dir / np.sqrt((llc_dir*llc_dir).sum(axis=0))
            relative_llc_goal_state = (llc_dir-self._last_state[1][:3])
            des_change = (self._last_state[1][:3] + relative_llc_goal_state) - p.getBaseVelocity(self._agent)[0]
            # des_change = (self._last_pose + llc_dir) - np.array(p.getBasePositionAndOrientation(self._agent)[0])
            # print ("self._last_state[1][:3] - p.getBaseVelocity(self._agent)[0]: ", self._last_state[1][:3] - p.getBaseVelocity(self._agent)[0])
            # llc_reward = np.dot(agentDir, llc_dir) - 1
            # llc_reward = -(agentDir*llc_dir).sum(axis=0)
            # llc_reward = np.exp((llc_reward*llc_reward) * -2.0)
            llc_reward = -(des_change*des_change).sum(axis=0)
        else:
        """
        llc_dir = np.array([self._llc_target[0], self._llc_target[1], 0])
        des_change = llc_dir - agentVel
        llc_reward = -(des_change*des_change).sum(axis=0)
            
        rewards = [[hlc_reward], [llc_reward]]
        # print ("rewards: ", rewards)
        return rewards
        
        
        
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
        
        # bad_indecies = np.where(intersections == self._target)[0]
        # print ("bad_indecies: ", bad_indecies)
        # bad_indecies = np.where(intersections == int(self._agent))
        # print ("bad_indecies: ", bad_indecies)
        # print ("self._agent: ", self._agent)
        """
        if ( len(bad_indecies) > 0):
            # print ("self._target: ", self._target)
            intersections[bad_indecies] = -1
        """
        # intersections_ = np.reshape(intersections, (size, size))
        # print ("intersections", intersections_)
        intersections = np.array(np.greater(intersections, 0), dtype="int")
        return intersections
    
    def updateAction(self, action):
        import numpy as np
        """
            action[0] == hlc action
            action[1] == llc action
        """
        self._hlc_timestep = self._hlc_timestep + 1
        if (self._hlc_timestep >= self._hlc_skip 
            and (self._ran < 0.5)):
            # print ("Updating llc target from HLC")
            self._llc_target = clampValue([action[0][0], action[0][1], 0], self._vel_bounds)
            ### Need to store this target in the sim as a gobal location to allow for computing local distance state.
            pos = np.array(p.getBasePositionAndOrientation(self._agent)[0])
            # self._llc_target = self._llc_target + action_
            self._hlc_timestep = 0
            ### Update llc action
            llc_obs = self.getObservation()[1]
            ### crazy hack to get proper state size...
            # llc_obs = np.concatenate([llc_obs,[0,0,0,0,0,0]])
            action[1] = self._llc.predict([llc_obs])
            # action[1] = [0.03, -0.023]
            # print ("self._llc_target: ", self._llc_target)
        ### apply delta position change.
        action_ = np.array([action[1][0], action[1][1], 0])
        agentVel = np.array(p.getBaseVelocity(self._agent)[0])
        # print ("action_: ", action_, " agentVel: ", agentVel) 
        action_ = agentVel + action_
        action_ = clampValue(action_, self._vel_bounds)
        if ("use_hlc_action_directly" in self._game_settings
            and (self._game_settings["use_hlc_action_directly"] == True)):
            action_ = self._llc_target
        # print ("New action: ", action_)
        p.resetBaseVelocity(self._agent, linearVelocity=action_, angularVelocity=[0,0,0])
        # vel = p.getBaseVelocity(self._agent)[0]
        # if (self._ran > 0.5): ### Only Do HLC training half the time.
        # print ("New vel: ", vel)
        
    def update(self):
        import numpy as np
        pos = np.array(p.getBasePositionAndOrientation(self._agent)[0])
        vel = np.array(p.getBaseVelocity(self._agent)[0])
        pos =  pos + (vel*self._dt)
        # print ("vel: ", vel)
        # pos = clampValue(pos, self._pos_bounds) ### Don't do this allow epoch to reset instead.
        pos[2] = 0.5
        ### Need to do this so the intersections are updated
        p.stepSimulation()
        p.resetBasePositionAndOrientation(self._agent, pos, p.getQuaternionFromEuler([0.,0,0]))
        ### This must be after setting position, because setting the position removes the velocity
        p.resetBaseVelocity(self._agent, linearVelocity=vel, angularVelocity=[0,0,0])
        
        reward = self.computeReward(state=None)
        # print("reward: ", reward)
        self.__reward = reward
        
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
        if ((goalDistance < self._reach_goal_threshold)
            or (pos[0] > self._map_area)
            or (pos[1] > self._map_area)
            or (pos[0] < -self._map_area)
            or (pos[1] < -self._map_area)):
            return True
        
        else:
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

class LLC(object):

    def __init__(self):
        pass
    
    def predict(self, state):
        
        return [0.5,0.5]
if __name__ == "__main__":
    import numpy as np
    settings = {"state_bounds": [[0],[1]],
                "action_bounds": [[0],[1]],
                "render": True,
                "map_size": 10.0}
    
    sim = NavGameHRL2D(settings)
    sim.init()
    
    llc = LLC()
    sim.setLLC(llc)

    action = np.array([[0.5, 0.5], [-0.5, -0.5]])
    import time
    for i in range(10000):
        if (i % 100 == 0):
            sim.reset()
        # p.stepSimulation()
        # p.setJointMotorControl2(botId, 1, p.TORQUE_CONTROL, force=1098.0)
        # p.setGravity(0,0,sim._GRAVITY)
        time.sleep(1/240.)
        sim.updateAction(action)
        sim.update()
        ob = sim.getObservation()
        reward = sim.computeReward()
        time.sleep(1/25.0)
        print ("Reward: ", reward)
        print ("od: ", ob)
        
