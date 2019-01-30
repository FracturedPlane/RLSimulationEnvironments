
import pybullet as p
import pybullet_data
import os
import time


class NavGame2D(object):
    
    def __init__(self):
        self._GRAVITY = -9.8
        self._dt = 1/50.0
        self._iters=2000 

    def init(self):
        
        self._physicsClient = p.connect(p.GUI)
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
        
        blockId = p.loadURDF("cube2.urdf",
                [2.0,2.0,0.5],
                cubeStartOrientation,
                useFixedBase=1) 
        
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
        p.setRealTimeSimulation(1)
        
    def reset(self):
        import numpy as np
        map_area = 10
        x = (np.random.rand()-0.5) * map_area
        y = (np.random.rand()-0.5) * map_area
        p.resetBasePositionAndOrientation(self._agent, [x,y,0.5], p.getQuaternionFromEuler([0.,0,0]))
        
        x = (np.random.rand()-0.5) * map_area
        y = (np.random.rand()-0.5) * map_area
        p.resetBasePositionAndOrientation(self._target, [x,y,0.5], p.getQuaternionFromEuler([0.,0,0]))
        
    def getObservation(self):
        out = []
        localMap = self.getlocalMapObservation()
        out.extend(localMap)
        data = p.getBaseVelocity(self._agent)
        ### linear vel
        out.extend(data[0])
        ### angular vel
        # out.extend(data[1])
        # print (out)
        goalDir = self.getTargetDirection()
        out.extend(goalDir)
        
        return out
    
    def computeReward(self):
        import numpy as np
        goalDir = self.getTargetDirection()
        # goalDir = goalDir / np.sqrt((goalDir*goalDir).sum(axis=0))
        # print ("goalDir: ", goalDir)
        agentVel = np.array(p.getBaseVelocity(self._agent)[0])
        agentVel = agentVel / np.sqrt((agentVel*agentVel).sum(axis=0))
        # print ("agentVel: ", agentVel)
        reward = np.dot(goalDir, agentVel)
        
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
        # print (rayResults)
        intersections = np.array(np.greater(intersections, 0), dtype="int")
        # intersections = np.reshape(intersections, (size, size))
        # print ("intersections", intersections)
        return intersections
    
    def act(self, action):
        import numpy as np
        ### apply delta position change.
        action = np.array([action[0], action[1], 0])
        
        pos = p.getBasePositionAndOrientation(self._agent)[0]
        p.resetBasePositionAndOrientation(self._agent, pos + action, p.getQuaternionFromEuler([0.,0,0]))
        p.resetBaseVelocity(self._agent, action, p.getQuaternionFromEuler([0.,0,0]))
        
        
    def endOfEpoch(self):
        import numpy as np
        
        pos = np.array(p.getBasePositionAndOrientation(self._agent)[0])
        posT = np.array(p.getBasePositionAndOrientation(self._target)[0])
        goalDirection = posT-pos
        goalDistance = np.sqrt((goalDirection*goalDirection).sum(axis=0))
        if (goalDistance < 1.0):
            return True
        else:
            return False
        
        
        

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
        sim.act([0.1,0.1])
        ob = sim.getObservation()
        reward = sim.computeReward()
        print ("Reward: ", reward)
        print ("od: ", ob)
        
    time.sleep(1000)