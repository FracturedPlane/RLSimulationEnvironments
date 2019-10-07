
import pybullet as p
import pybullet
import pybullet_data
import os
import time
from rlsimenv.EnvWrapper import ActionSpace
from rlsimenv.Environment import Environment, clampValue
from rlsimenv.PyBulletUtil import *

class NavGameAntHRL(PyBulletEnv):
    """
        The HLC is the first agent
        The LLC is the second agent
    """
    
    def __init__(self, settings):
        super(NavGameAntHRL,self).__init__(settings)
        
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

    def getNumAgents(self):
        return 2
    
    def display(self):
        pass
    
    
    def init(self):
        
        super(NavGameAntHRL,self).init()
        
        cubeStartPos = [0,0,0.5]
        cubeStartOrientation = self._p.getQuaternionFromEuler([0.,0,0])
        self.objects = self._p.loadMJCF(
            os.path.join(pybullet_data.getDataPath(), "mjcf", "ant.xml"))
        self.parts, self.jdict, self._jointIds, self.robot_body = self.addToScene(
            self._p, self.objects)
        # self._agent = self.robot_body
        self._agent = self.objects[0]
        
        lo = [-1.0 for l in range(len(self._jointIds))]
        hi = [ 1.0 for l in range(len(self._jointIds))]
        self._action_bounds = np.array([lo, hi])
        self.computeActionBounds()
        
        self._blocks = []
        self._num_blocks=0
        if ("num_blocks" in self._game_settings):
            self._num_blocks = self._game_settings["num_blocks"] 
        for i in range(self._num_blocks):
            blockId = self._p.loadURDF("cube2.urdf",
                    [2.0,2.0,0.5],
                    cubeStartOrientation,
                    useFixedBase=1) 
            self._blocks.append(blockId)
        
        
        self._target = self._p.loadURDF("sphere2red.urdf",
                cubeStartPos,
                cubeStartOrientation,
                useFixedBase=1)
        
         
        lo = [0.0 for l in self.getObservation()[0]]
        hi = [1.0 for l in self.getObservation()[0]]
        state_bounds_llc = [lo, hi]
        state_bounds = state_bounds_llc
        if ( "use_MARL_HRL" in self._game_settings
             and (self._game_settings["use_MARL_HRL"] == True)):
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
        
        x = (np.random.rand()-0.5) * self._map_area * 2.0
        y = (np.random.rand()-0.5) * self._map_area * 2.0
        self._p.resetBasePositionAndOrientation(self._target, [x,y,0.5], self._p.getQuaternionFromEuler([0.,0,0]))
        self._p.resetBaseVelocity(self._target, [0,0,0], [0,0,0])
        
        # self._ran = np.random.rand(1)[0]
        if ("ignore_hlc_actions" in self._game_settings
            and (self._game_settings["ignore_hlc_actions"] == True)):
            self._ran = 0.6 ## Ignore HLC action and have env generate them if > 0.5.
        else:
            self._ran = 0.4 ## Ignore HLC action and have env generate them if > 0.5.
        ### By default init this to direction towards goal
        if ( "use_full_pose_goal" in self._game_settings
         and (self._game_settings["use_full_pose_goal"] == True)):
            self._llc_target = self.genRandomPose()
        else:
            x = (np.random.rand()-0.5) * 2.0
            y = (np.random.rand()-0.5) * 2.0
            self._llc_target = [x, y]
        ### Make sure to apply HLC action right away
        self._hlc_timestep = 1000000
        self._hlc_skip = 10
        if ("hlc_timestep" in self._game_settings):
            self._hlc_skip = self._game_settings["hlc_timestep"]
        
        ### Reset obstacles
        for i in range(len(self._blocks)):
            x = (np.random.rand()-0.5) * self._map_area * 2.0
            y = (np.random.rand()-0.5) * self._map_area * 2.0
            self._p.resetBasePositionAndOrientation(self._blocks[i], [x,y,0.5], self._p.getQuaternionFromEuler([0.,0,0]))
            self._p.resetBaseVelocity(self._blocks[i], [0,0,0], [0,0,0]) 
            
    def setLLC(self, llc):
        self._llc = llc

    def setHLP(self, hlp):
        pass
        
    def getObservation(self):
        import numpy as np
        out = []
        out_hlc = []
        if ("include_egocentric_vision" in self._game_settings
            and (self._game_settings["include_egocentric_vision"] == True)):
            localMap = self.getlocalMapObservation()
            out_hlc.extend(localMap)
        data = self.getRobotPose()
        ### linear vel
        out_hlc.extend(data)
        ### angular vel
        pos = np.array(self._p.getBasePositionAndOrientation(self._agent)[0])
        posT = np.array(self._p.getBasePositionAndOrientation(self._target)[0])
        goalDirection = posT-pos
        out_hlc.extend([goalDirection[0], goalDirection[1]])
        out_llc = []
        out_llc.extend(data)
        ### Relative distance from current LLC state
        # if (self._ran < 0.5):
        # out_llc.extend(np.array(self._llc_target) - np.array(data[0]))
        
        if ( "use_MARL_HRL" in self._game_settings
             and (self._game_settings["use_MARL_HRL"] == True)):
            out_llc.extend(np.array(self._llc_target))
        else:
            out_llc.extend(np.array([goalDirection[0], goalDirection[1]]))
        # else:
        #     out_llc.extend(np.array(self._llc_target) - pos)
        # print ("out_llc: ", out_llc)
        if ( "use_MARL_HRL" in self._game_settings
             and (self._game_settings["use_MARL_HRL"] == True)):
            out.append(np.array(out_hlc))
        out.append(np.array(out_llc))
        self._last_state = np.array(out)
        self._last_pose = np.array(self._p.getBasePositionAndOrientation(self._agent)[0])
        return self._last_state
    
    def getState(self):
        return self.getObservation()
    
    def computeReward(self, state=None):
        """
            
        """
        import numpy as np
        pos = np.array(self._p.getBasePositionAndOrientation(self._agent)[0])
        posT = np.array(self._p.getBasePositionAndOrientation(self._target)[0])
        rewards = []
        
        goalDirection = posT-pos
        goalDistance = np.sqrt((goalDirection*goalDirection).sum(axis=0))
        goalDir = self.getTargetDirection()
        agentVel = np.array(self._p.getBaseVelocity(self._agent)[0])
        agentDir = agentVel / np.sqrt((agentVel*agentVel).sum(axis=0))
        velDiff = goalDir - agentVel
        diffMag = np.sqrt((velDiff*velDiff).sum(axis=0))
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
        
        for box_id in self._blocks:
            contacts = self._p.getContactPoints(self._agent, box_id)
            # print ("contacts: ", contacts)
            if len(contacts) > 0:
                hlc_reward = hlc_reward + -1.0
                break
        if ( "use_full_pose_goal" in self._game_settings
             and (self._game_settings["use_full_pose_goal"] == True)):
            llc_dir = self.getRobotPose()
            des_change = llc_dir - self._llc_target
        else:
            llc_dir = np.array([self._llc_target[0], self._llc_target[1], 0])
            des_change = llc_dir - agentVel
        # llc_reward = -(des_change*des_change).sum(axis=0)
        llc_reward = -(np.fabs(des_change)).sum(axis=0)
        if ( "use_MARL_HRL" in self._game_settings
             and (self._game_settings["use_MARL_HRL"] == True)):
            rewards = [[hlc_reward], [llc_reward]]
        else:
            ### Use Simple HLC reward in this case
            rewards = [ [hlc_reward]]
        # print ("rewards: ", rewards)
        return rewards
        
        
    def getTargetDirection(self):
        ### raycast around the area of the agent
        import numpy as np
        
        pos = np.array(self._p.getBasePositionAndOrientation(self._agent)[0])
        posT = np.array(self._p.getBasePositionAndOrientation(self._target)[0])
        goalDirection = posT-pos
        goalDirection = goalDirection / np.sqrt((goalDirection*goalDirection).sum(axis=0))
        return goalDirection
    
    def getlocalMapObservation(self):
        ### raycast around the area of the agent
        """
            For now this includes the agent in the center of the computation
        """
        import numpy as np
        
        pos = self._p.getBasePositionAndOrientation(self._agent)[0]
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
        fromRays = toRays + np.array([0,0,5])
        rayResults = self._p.rayTestBatch(fromRays, toRays)
        intersections = [ray[0] for ray in rayResults]
        
        for ray in range(len(intersections)):
            if (intersections[ray] in [self._target, self._agent]):
                # print ("bad index: ", ray)
                intersections[ray] = -1
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
            and (self._ran < 0.5) and 
            (( "use_MARL_HRL" in self._game_settings
             and (self._game_settings["use_MARL_HRL"] == True)))):
            # print ("Updating llc target from HLC")
            if ("use_hardCoded_LLC_goals" in self._game_settings
             and (self._game_settings["use_hardCoded_LLC_goals"] == True)):
                # pass
                if ( "use_full_pose_goal" in self._game_settings
                 and (self._game_settings["use_full_pose_goal"] == True)):
                    self._llc_target = self.genRandomPose()
                else:
                    x = (np.random.rand()-0.5) * 2.0
                    y = (np.random.rand()-0.5) * 2.0
                    self._llc_target = [x, y]
            else:
                if ( "use_full_pose_goal" in self._game_settings
                 and (self._game_settings["use_full_pose_goal"] == True)):
                    self._llc_target = action[0]
                else:
                    self._llc_target = clampValue([action[0][0], action[0][1]], self._vel_bounds)
            ### Need to store this target in the sim as a gobal location to allow for computing local distance state.
            pos = np.array(self._p.getBasePositionAndOrientation(self._agent)[0])
            self._hlc_timestep = 0
            ### Update llc action
            llc_obs = self.getObservation()[1]
            ### crazy hack to get proper state size...
            if ("append_centralized_state_hack" in self._game_settings
                and (self._game_settings["append_centralized_state_hack"] == True)):
                llc_obs = np.concatenate([llc_obs,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
            action[1] = self._llc.predict([llc_obs])[0]
            # action[1] = [0.03, -0.023]
            # print ("self._llc_target: ", self._llc_target)
        ### apply delta position change.
        if ( "use_MARL_HRL" in self._game_settings
             and (self._game_settings["use_MARL_HRL"] == False)):
            action_ = np.array(action[0])
        else:
            action_ = np.array(action[1])
        """
        if ("use_hlc_action_directly" in self._game_settings
            and (self._game_settings["use_hlc_action_directly"] == True)):
            action_ = self._llc_target
        """
        # print ("New action: ", action_)
        # self._p.resetBaseVelocity(self._agent, linearVelocity=action_, angularVelocity=[0,0,0])
        super(NavGameAntHRL,self).updateAction(action_)
        # vel = self._p.getBaseVelocity(self._agent)[0]
        # if (self._ran > 0.5): ### Only Do HLC training half the time.
        # print ("New vel: ", vel)
        
    def agentHasFallen(self):
        return self.endOfEpoch()
    
    def endOfEpoch(self):
        import numpy as np
        
        pos = np.array(self._p.getBasePositionAndOrientation(self._agent)[0])
        posT = np.array(self._p.getBasePositionAndOrientation(self._target)[0])
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
        
        return np.mean(self._llc_action_bounds, axis=0)
    
    def setActionBounds(self, bounds):
        self._llc_action_bounds = bounds
        
if __name__ == "__main__":
    import numpy as np
    settings = {"state_bounds": [[0],[1]],
                "action_bounds": [[0],[1]],
                "render": True,
                "map_size": 10.0,
                "control_substeps": 10,
                "physics_timestep": 0.00333333333,}
    
    sim = NavGameAntHRL(settings)
    sim.init()
    
    llc = LLC()
    llc.setActionBounds(sim._action_bounds)
    sim.setLLC(llc)

    # action = np.array([[0.5, 0.5], [-0.5, -0.5]])
    print ("sim._action_bounds: ", sim._action_bounds)
    action = np.mean(sim._action_bounds, axis=0)
    import time
    for i in range(10000):
        if (i % 100 == 0):
            sim.reset()
        time.sleep(1/240.)
        print ("llc action bounds: ", llc._llc_action_bounds)
        sim.updateAction([[-1.1232,1.534534],np.random.normal(action, sim._action_bounds[1] - sim._action_bounds[0])])
        sim.update()
        ob = sim.getObservation()
        reward = sim.computeReward()
        time.sleep(1/30.0)
        print ("Reward: ", reward)
        print ("od: ", ob)
        
