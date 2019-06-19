
import pybullet as p
import pybullet_data
import os
import time
from rlsimenv.EnvWrapper import ActionSpace
from rlsimenv.Environment import Environment, clampValue

class CassieWalk(Environment):
    """
        Cassie walk env
    """
    
    def __init__(self, settings):
        super(CassieWalk,self).__init__(settings)
        self._GRAVITY = -9.8
        self._dt = 1/240.0
        self._iters=2000 
        
        
        self._state_bounds = self._game_settings['state_bounds']
        self._action_bounds = self._game_settings['action_bounds']
        self._action_length = len([0,0,1.0204,-1.97,-0.084,2.06,-1.9,0,0,1.0204,-1.97,-0.084,2.06,-1.9,0])
        
        
        
        # ob_low = [0] * len(self.getEnv().getObservationSpaceSize()
        # ob_high = [1] * self.getEnv().getObservationSpaceSize() 
        # observation_space = [ob_low, ob_high]
        # self._observation_space = ActionSpace(observation_space)
        self._action_space = ActionSpace(self._game_settings['action_bounds'])
        
        
    def resetAgent(self):
        activeJoint = 0
        for j in range (p.getNumJoints(self._agent)):
            p.changeDynamics(self._agent,j,linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self._agent,j)
            #print(info)
            jointName = info[1]
            jointType = info[2]
            if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
                # self._jointIds.append(j)
                # paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"),-4,4,jointAngles[activeJoint]))
                p.resetJointState(self._agent, j, self._jointAngles[activeJoint])
                activeJoint+=1
        
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
        # p.connect(p.GUI)
        
        p.loadURDF("plane.urdf")
        self._agent = p.loadURDF("rlsimenv/data/cassie/urdf/cassie_collide.urdf",[0,0,0.8], useFixedBase=False)
        # gravId = p.addUserDebugParameter("gravity",-10,10,-10)
        self._jointIds=[]
        paramIds=[]
        
        p.setPhysicsEngineParameter(numSolverIterations=100)
        p.changeDynamics(self._agent,-1,linearDamping=0, angularDamping=0)
        
        self._jointAngles=[0,0,1.0204,-1.97,-0.084,2.06,-1.9,0,0,1.0204,-1.97,-0.084,2.06,-1.9,0]
        activeJoint=0
        for j in range (p.getNumJoints(self._agent)):
            p.changeDynamics(self._agent,j,linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self._agent,j)
            #print(info)
            jointName = info[1]
            jointType = info[2]
            if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
                self._jointIds.append(j)
                # paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"),-4,4,jointAngles[activeJoint]))
                # self._paramIds.append()
                p.resetJointState(self._agent, j, self._jointAngles[activeJoint])
                activeJoint+=1
                
        p.setRealTimeSimulation(0)
        
        lo = [0.0 for l in self.getObservation()[0]]
        hi = [1.0 for l in self.getObservation()[0]]
        state_bounds = [lo, hi]
        state_bounds = [state_bounds]
        
        self._game_settings['state_bounds'] = [lo, hi]
        self._state_length = len(self._game_settings['state_bounds'][0])
        print ("self._state_length: ", self._state_length)
        self._observation_space = ActionSpace(self._game_settings['state_bounds'])
        
        self._last_state = self.getState()
        self._last_pose = p.getBasePositionAndOrientation(self._agent)[0]
        
    def initEpoch(self):
        import numpy as np
        self.resetAgent()
        
    def reset(self):
        self.initEpoch()
        return self.getObservation()
            
    def setLLC(self, llc):
        self._llc = llc
        
    def getObservation(self):
        import numpy as np
        out_hlc = []
        data = p.getBaseVelocity(self._agent)
        ### linear vel
        out_hlc.extend(data[0])
        ### angular vel
        out_hlc.extend(data[1])
        pos = np.array(p.getBasePositionAndOrientation(self._agent)[0])

        self._last_state = [np.array(out_hlc)]
        self._last_pose = np.array(p.getBasePositionAndOrientation(self._agent)[0])
        return self._last_state
    
    def getState(self):
        return self.getObservation()
    
    def computeReward(self, state=None):
        """
            
        """
        import numpy as np
        data = p.getBaseVelocity(self._agent)
        rewards = data[0][0]
        # print ("rewards: ", rewards)
        return rewards
        
    def updateAction(self, action):
        import numpy as np
        # while(1):
        # p.getCameraImage(320,200)
        # p.setGravity(0,0,p.readUserDebugParameter(gravId))
        for i in range(len(self._jointIds)):
            c = self._jointIds[i]
            targetPos = action[i]
            p.setJointMotorControl2(self._agent,c,p.POSITION_CONTROL,targetPos, force=140.)    
        # time.sleep(0.01)
        
    def update(self):
        import numpy as np
        ### Need to do this so the intersections are updated
        p.stepSimulation()
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
        

if __name__ == "__main__":
    import numpy as np
    settings = {"state_bounds": [[0],[1]],
                "action_bounds": [[0],[1]],
                "render": True}
    sim = CassieWalk(settings)
    sim.init()
    action = np.array([0,0,1.0204,-1.97,-0.084,2.06,-1.9,0,0,1.0204,-1.97,-0.084,2.06,-1.9,0])

    import time
    
    for i in range(1000):
        if (i % 100 == 0):
            sim.reset()
        # p.stepSimulation()
        # p.setJointMotorControl2(botId, 1, p.TORQUE_CONTROL, force=1098.0)
        # p.setGravity(0,0,sim._GRAVITY)
        sim.updateAction(action + np.random.normal(0,0.1, len(action)))
        sim.update()
        # sim.updateAction(action)
        ob = sim.getObservation()
        reward = sim.computeReward()
        time.sleep(1/240.)
        print ("Reward: ", reward)
        print ("od: ", ob)
        
    