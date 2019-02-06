
import pybullet as p
import pybullet_data
import os
import time
from rlsimenv.EnvWrapper import ActionSpace
from rlsimenv.Environment import Environment


class NavGame2DDirect(Environment):
    
    def __init__(self, settings):
        super(NavGame2DDirect,self).__init__(settings)
        self._game_settings = settings
        self._GRAVITY = 0.0
        self._dt = 1/50.0
        self._iters=2000 
        
        self._state_bounds = self._game_settings['state_bounds']
        self._action_bounds = self._game_settings['action_bounds']
        self._action_length = len(self._action_bounds[0])
        
        # ob_low = [0] * len(self.getEnv().getObservationSpaceSize()
        # ob_high = [1] * self.getEnv().getObservationSpaceSize() 
        # observation_space = [ob_low, ob_high]
        # self._observation_space = ActionSpace(observation_space)
        self._action_space = ActionSpace(self._game_settings['action_bounds'])
        
        self._map_area = 5
        
    def getActionSpaceSize(self):
        return self._action_length
    
    def getObservationSpaceSize(self):
        return self._state_length

    def getNumAgents(self):
        return 1
    
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
        """
        blockId = p.loadURDF("cube2.urdf",
                [2.0,2.0,0.5],
                cubeStartOrientation,
                useFixedBase=1) 
        """
        self._target = p.loadURDF("sphere2red.urdf",
                cubeStartPos,
                cubeStartOrientation)
        
         
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
        # p.setRealTimeSimulation(1)
        
        lo = self.getObservation()[0] * 0.0
        hi = lo + 1.0
        self._game_settings['state_bounds'] = [lo, hi]
        self._state_length = len(self._game_settings['state_bounds'][0])
        print ("self._state_length: ", self._state_length)
        self._observation_space = ActionSpace(self._game_settings['state_bounds'])
        
    def initEpoch(self):
        import numpy as np
        
        x = (np.random.rand()-0.5) * self._map_area * 2.0
        y = (np.random.rand()-0.5) * self._map_area * 2.0
        p.resetBasePositionAndOrientation(self._agent, [x,y,0.5], p.getQuaternionFromEuler([0.,0,0]))
        p.resetBaseVelocity(self._agent, [0,0,0], [0,0,0])
        
        x = (np.random.rand()-0.5) * self._map_area * 2.0
        y = (np.random.rand()-0.5) * self._map_area * 2.0
        p.resetBasePositionAndOrientation(self._target, [x,y,0.5], p.getQuaternionFromEuler([0.,0,0]))
        p.resetBaseVelocity(self._target, [0,0,0], [0,0,0])
        
    def getObservation(self):
        import numpy as np
        out = []
        # localMap = self.getlocalMapObservation()
        # out.extend(localMap)
        data = p.getBaseVelocity(self._agent)
        ### linear vel
        out.extend(data[0])
        ### angular vel
        # out.extend(data[1])
        # print (out)
        goalDir = self.getTargetDirection()
        out.extend(goalDir)
        # out = [np.array([np.array(out)])]
        out = np.array([np.array(out)])
        # print ("obs: ", np.array(out))
        return out
    
    def getState(self):
        return self.getObservation()
    
    def computeReward(self, state=None):
        import numpy as np
        goalDir = self.getTargetDirection()
        # goalDir = goalDir / np.sqrt((goalDir*goalDir).sum(axis=0))
        # print ("goalDir: ", goalDir)
        agentVel = np.array(p.getBaseVelocity(self._agent)[0])
        # agentSpeed = np.sqrt((agentVel*agentVel).sum(axis=0))
        velDiff = goalDir - agentVel
        diffMag = np.sqrt((velDiff*velDiff).sum(axis=0))
        # agentVel = agentVel / agentSpeed
        # print ("agentVel: ", agentVel)
        # agentSpeedDiff = (1 - agentSpeed)
        ### heading towards goal
        # reward = np.dot(goalDir, agentVel) + np.exp(agentSpeedDiff*agentSpeedDiff * -2.0)
        reward = np.exp((diffMag*diffMag) * -2.0)
        
        return reward
        
        
        
    def getTargetDirection(self):
        ### raycast around the area of the agent
        import numpy as np
        
        pos = np.array(p.getBasePositionAndOrientation(self._agent)[0])
        posT = np.array(p.getBasePositionAndOrientation(self._target)[0])
        goalDirection = posT-pos
        goalDirection = goalDirection / np.sqrt((goalDirection*goalDirection).sum(axis=0))
        return goalDirection
    
    def updateAction(self, action):
        import numpy as np
        ### apply delta position change.
        action = np.array([action[0], action[1], 0])
        # action = np.array([1, 0, 0])
        # print ("New action: ", action)
        p.resetBaseVelocity(self._agent, linearVelocity=action, angularVelocity=[0,0,0])
        vel = p.getBaseVelocity(self._agent)[0]
        # print ("New vel: ", vel)
        
    def update(self):
        import numpy as np
        pos = np.array(p.getBasePositionAndOrientation(self._agent)[0])
        vel = np.array(p.getBaseVelocity(self._agent)[0])
        pos_a = pos + (vel*self._dt)
        pos_a[2] = 0.5
        p.resetBasePositionAndOrientation(self._agent, pos_a, p.getQuaternionFromEuler([0.,0,0]))
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
        # print ("goalDistance: ", goalDistance)
        if (goalDistance < 1.2):
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
        

if __name__ == "__main__":
    
    sim = NavGame2D()
    sim.init()

    import time
    for i in range(10000):
        if (i % 100 == 0):
            sim.reset()
        # p.stepSimulation()
        # p.setJointMotorControl2(botId, 1, p.TORQUE_CONTROL, force=1098.0)
        # p.setGravity(0,0,sim._GRAVITY)
        time.sleep(1/240.)
        sim.updateAction([0.1,0.1])
        ob = sim.getObservation()
        reward = sim.computeReward()
        print ("Reward: ", reward)
        print ("od: ", ob)
        
    time.sleep(1000)