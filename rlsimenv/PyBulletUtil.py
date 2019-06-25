
import pybullet
import pybullet_data
import numpy as np

from rlsimenv.EnvWrapper import ActionSpace
from rlsimenv.Environment import Environment, clampValue

class PyBulletEnv(Environment):
    
    """
        The HLC is the first agent
        The LLC is the second agent
    """
    
    def __init__(self, settings):
        super(PyBulletEnv,self).__init__(settings)
        import pybullet as p
        self._p = p
        self._dt = self._game_settings["physics_timestep"]
        self._GRAVITY = -9.8
        self._iters=2000 
        
    def init(self):
        
        if (self._game_settings['render']):
            # self._object.setPosition([self._x[self._step], self._y[self._step], 0.0] )
            self._physicsClient = self._p.connect(self._p.GUI)
        else:
            self._physicsClient = self._p.connect(self._p.DIRECT)
            
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.resetSimulation()
        #p.setRealTimeSimulation(True)
        self._p.setGravity(0,0,self._GRAVITY)
        self._p.setTimeStep(self._dt)
        
        self._p.setRealTimeSimulation(0)
        
        self._p.loadURDF("plane.urdf")
        
    def getActionSpaceSize(self):
        return self._action_length
    
    def getObservationSpaceSize(self):
        return self._state_length
    
    def getNumAgents(self):
        return 1
    
    def display(self):
        pass
    
    def resetAgent(self):
        
        # Set Initial pose
        self._p.resetBasePositionAndOrientation(
            self._agent, posObj=self._init_root_pos[0], ornObj=self._init_root_pos[1])
        self._p.resetBaseVelocity(
            self._agent, linearVelocity=self._init_root_vel[0], angularVelocity=[0,0,0])           
        
        activeJoint = 0
        for j in range (self._p.getNumJoints(self._agent)):
            self._p.changeDynamics(self._agent,j,linearDamping=0, angularDamping=0)
            info = self._p.getJointInfo(self._agent,j)
            #print(info)
            jointName = info[1]
            jointType = info[2]
            if (jointType==self._p.JOINT_PRISMATIC or jointType==self._p.JOINT_REVOLUTE):
                # paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"),-4,4,jointAngles[activeJoint]))
                self._p.resetJointState(self._agent, j, self._jointAngles[activeJoint])
                activeJoint+=1

    def getRobotPose(self):
        import numpy as np
        out_hlc = []
        data = self._p.getBaseVelocity(self._agent)
        ### root linear vel
        out_hlc.extend(data[0])
        ### root angular vel
        out_hlc.extend(data[1])
        pos = np.array(self._p.getBasePositionAndOrientation(self._agent))
        ### root height
        out_hlc.append(pos[0][1])
        ### root rotation
        out_hlc.extend(pos[1])
        
        ### Add pose state as relative rotations from parent frame 
        ### Works for revolute joints
        for j in range(len(self._jointIds)):
            # info = self._p.getJointState(self._agent,self._jointIds[j])
            #print(info)
              
            info = self._jointIds[j].get_state()  
            out_hlc.append(info[0]) ### Position
            out_hlc.append(info[1]) ### Velocity
                
        _state = [np.array(out_hlc)]
        self._last_pose = np.array(self._p.getBasePositionAndOrientation(self._agent)[0])
        return _state
    
    def getObservation(self):
        import numpy as np
        self._last_state = self.getRobotPose()
        return self._last_state
    
    
    def updateAction(self, action):
        import numpy as np
        # while(1):
        # p.getCameraImage(320,200)
        # p.setGravity(0,0,p.readUserDebugParameter(gravId))
        for i in range(len(self._jointIds)):
            # c = self._jointIds[i]
            targetPos = action[i]
            self._jointIds[i].set_position(targetPos)
            # self._p.setJointMotorControl2(self._agent,c,p.POSITION_CONTROL,targetPos, force=140.)    
        # time.sleep(0.01)
    
    def update(self):
        import numpy as np
        ### Need to do this so the intersections are updated
        for i in range(self._game_settings["control_substeps"]):
            self._p.stepSimulation()
        reward = self.computeReward(state=None)
        # print("reward: ", reward)
        self.__reward = reward
    
class Pose_Helper:  # dummy class to comply to original interface

  def __init__(self, body_part):
    self.body_part = body_part

  def xyz(self):
    return self.body_part.current_position()

  def rpy(self):
    return pybullet.getEulerFromQuaternion(self.body_part.current_orientation())

  def orientation(self):
      return self.body_part.current_orientation()

class BodyPart:

  def __init__(self, bullet_client, body_name, bodies, bodyIndex, bodyPartIndex):
    self.bodies = bodies
    self._p = bullet_client
    self.bodyIndex = bodyIndex
    self.bodyPartIndex = bodyPartIndex
    self.initialPosition = self.current_position()
    self.initialOrientation = self.current_orientation()
    self.bp_pose = Pose_Helper(self)

  def state_fields_of_pose_of(
      self, body_id,
      link_id=-1):  # a method you will most probably need a lot to get pose and orientation
    if link_id == -1:
      (x, y, z), (a, b, c, d) = self._p.getBasePositionAndOrientation(body_id)
    else:
      (x, y, z), (a, b, c, d), _, _, _, _ = self._p.getLinkState(body_id, link_id)
    return np.array([x, y, z, a, b, c, d])

  def get_position(self):
    return self.current_position()

  def get_pose(self):
    return self.state_fields_of_pose_of(self.bodies[self.bodyIndex], self.bodyPartIndex)

  def speed(self):
    if self.bodyPartIndex == -1:
      (vx, vy, vz), _ = self._p.getBaseVelocity(self.bodies[self.bodyIndex])
    else:
      (x, y, z), (a, b, c, d), _, _, _, _, (vx, vy, vz), (vr, vp, vy) = self._p.getLinkState(
          self.bodies[self.bodyIndex], self.bodyPartIndex, computeLinkVelocity=1)
    return np.array([vx, vy, vz])

  def current_position(self):
    return self.get_pose()[:3]

  def current_orientation(self):
    return self.get_pose()[3:]

  def get_orientation(self):
    return self.current_orientation()

  def reset_position(self, position):
    self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position,
                                            self.get_orientation())

  def reset_orientation(self, orientation):
    self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], self.get_position(),
                                            orientation)

  def reset_velocity(self, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0]):
    self._p.resetBaseVelocity(self.bodies[self.bodyIndex], linearVelocity, angularVelocity)

  def reset_pose(self, position, orientation):
    self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position, orientation)

  def pose(self):
    return self.bp_pose

  def contact_list(self):
    return self._p.getContactPoints(self.bodies[self.bodyIndex], -1, self.bodyPartIndex, -1)



class Joint:

  def __init__(self, bullet_client, joint_name, bodies, bodyIndex, jointIndex):
    self.bodies = bodies
    self._p = bullet_client
    self.bodyIndex = bodyIndex
    self.jointIndex = jointIndex
    self.joint_name = joint_name

    jointInfo = self._p.getJointInfo(self.bodies[self.bodyIndex], self.jointIndex)
    self.lowerLimit = jointInfo[8]
    self.upperLimit = jointInfo[9]

    self.power_coeff = 0

  def set_state(self, x, vx):
    self._p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, x, vx)

  def current_position(self):  # just some synonyme method
    return self.get_state()

  def current_relative_position(self):
    pos, vel = self.get_state()
    pos_mid = 0.5 * (self.lowerLimit + self.upperLimit)
    return (2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit), 0.1 * vel)

  def get_state(self):
    x, vx, _, _ = self._p.getJointState(self.bodies[self.bodyIndex], self.jointIndex)
    return x, vx

  def get_position(self):
    x, _ = self.get_state()
    return x

  def get_orientation(self):
    _, r = self.get_state()
    return r

  def get_velocity(self):
    _, vx = self.get_state()
    return vx

  def set_position(self, position):
    self._p.setJointMotorControl2(self.bodies[self.bodyIndex],
                                  self.jointIndex,
                                  pybullet.POSITION_CONTROL,
                                  targetPosition=position)

  def set_velocity(self, velocity):
    self._p.setJointMotorControl2(self.bodies[self.bodyIndex],
                                  self.jointIndex,
                                  pybullet.VELOCITY_CONTROL,
                                  targetVelocity=velocity)

  def set_motor_torque(self, torque):  # just some synonyme method
    self.set_torque(torque)

  def set_torque(self, torque):
    self._p.setJointMotorControl2(bodyIndex=self.bodies[self.bodyIndex],
                                  jointIndex=self.jointIndex,
                                  controlMode=pybullet.TORQUE_CONTROL,
                                  force=torque)  #, positionGain=0.1, velocityGain=0.1)

  def reset_current_position(self, position, velocity):  # just some synonyme method
    self.reset_position(position, velocity)

  def reset_position(self, position, velocity):
    self._p.resetJointState(self.bodies[self.bodyIndex],
                            self.jointIndex,
                            targetValue=position,
                            targetVelocity=velocity)
    self.disable_motor()

  def disable_motor(self):
    self._p.setJointMotorControl2(self.bodies[self.bodyIndex],
                                  self.jointIndex,
                                  controlMode=pybullet.POSITION_CONTROL,
                                  targetPosition=0,
                                  targetVelocity=0,
                                  positionGain=0.1,
                                  velocityGain=0.1,
                                  force=0)