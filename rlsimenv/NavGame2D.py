
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
        
    def reset(self):
        import numpy as np
        x = (np.random.rand()-0.5) * 10
        y = (np.random.rand()-0.5) * 10
        p.resetBasePositionAndOrientation(self._agent, [x,y,0.5], p.getQuaternionFromEuler([0.,0,0]))
        
        
    def run(self):
        import time
        p.setRealTimeSimulation(1)
        for i in range(10000):
            if (i % 10 == 0):
                self.reset()
            # p.stepSimulation()
            # p.setJointMotorControl2(botId, 1, p.TORQUE_CONTROL, force=1098.0)
            p.setGravity(0,0,self._GRAVITY)
            time.sleep(1/240.)
            # self.act([0.1,0.1])
            self.getObservation()
            
        time.sleep(1000)
        
    def getObservation(self):
        out = []
        data = p.getBaseVelocity(self._agent)
        ### linear vel
        out.extend(data[0])
        ### angular vel
        out.extend(data[1])
        # print (out)
        self.getlocalMapObservation()
        return out
        
    def getlocalMapObservation(self):
        ### raycast around the area of the agent
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
        intersections = np.reshape(intersections, (size, size))
        print ("intersections", intersections)
        return intersections
    
    def act(self, action):
        import numpy as np
        ### apply delta position change.
        action = np.array([action[0], action[1], 0])
        pos = p.getBasePositionAndOrientation(self._agent)[0]
        p.resetBasePositionAndOrientation(self._agent, pos + action, p.getQuaternionFromEuler([0.,0,0]))
        
        
        

if __name__ == "__main__":
    
    sim = NavGame2D()
    sim.init()
    sim.run()