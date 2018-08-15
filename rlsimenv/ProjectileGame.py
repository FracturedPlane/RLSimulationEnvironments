"""
A 2D projectile environment

"""

import sys, os, random, time
from math import *
import numpy as np
# from twisted.protocols import stateful
import copy
import math
from rlsimenv.EnvWrapper import ActionSpace

def clampAction(actionV, bounds):
    """
    bounds[0] is lower bounds
    bounds[1] is upper bounds
    """
    for i in range(len(actionV)):
        if actionV[i] < bounds[0][i]:
            actionV[i] = bounds[0][i]
        elif actionV[i] > bounds[1][i]:
            actionV[i] = bounds[1][i]
    return actionV 

def sign(x):
    """Returns 1.0 if x is positive, -1.0 if x is negative or zero."""
    if x > 0.0: return 1.0
    else: return -1.0

def len3(v):
    """Returns the length of 3-vector v."""
    return sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def neg3(v):
    """Returns the negation of 3-vector v."""
    return (-v[0], -v[1], -v[2])

def add3(a, b):
    """Returns the sum of 3-vectors a and b."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def sub3(a, b):
    """Returns the difference between 3-vectors a and b."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def mul3(v, s):
    """Returns 3-vector v multiplied by scalar s."""
    return (v[0] * s, v[1] * s, v[2] * s)

def div3(v, s):
    """Returns 3-vector v divided by scalar s."""
    return (v[0] / s, v[1] / s, v[2] / s)

def dist3(a, b):
    """Returns the distance between point 3-vectors a and b."""
    return len3(sub3(a, b))

def norm3(v):
    """Returns the unit length 3-vector parallel to 3-vector v."""
    l = len3(v)
    if (l > 0.0): return (v[0] / l, v[1] / l, v[2] / l)
    else: return (0.0, 0.0, 0.0)

def dot3(a, b):
    """Returns the dot product of 3-vectors a and b."""
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])

def cross(a, b):
    """Returns the cross product of 3-vectors a and b."""
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0])

def project3(v, d):
    """Returns projection of 3-vector v onto unit 3-vector d."""
    return mul3(v, dot3(norm3(v), d))

def acosdot3(a, b):
    """Returns the angle between unit 3-vectors a and b."""
    x = dot3(a, b)
    if x < -1.0: return pi
    elif x > 1.0: return 0.0
    else: return acos(x)

def rotate3(m, v):
    """Returns the rotation of 3-vector v by 3x3 (row major) matrix m."""
    return (v[0] * m[0] + v[1] * m[1] + v[2] * m[2],
        v[0] * m[3] + v[1] * m[4] + v[2] * m[5],
        v[0] * m[6] + v[1] * m[7] + v[2] * m[8])

def invert3x3(m):
    """Returns the inversion (transpose) of 3x3 rotation matrix m."""
    return (m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8])

def zaxis(m):
    """Returns the z-axis vector from 3x3 (row major) rotation matrix m."""
    return (m[2], m[5], m[8])

def calcRotMatrix(axis, angle):
    """
    Returns the row-major 3x3 rotation matrix defining a rotation around axis by
    angle.
    """
    cosTheta = cos(angle)
    sinTheta = sin(angle)
    t = 1.0 - cosTheta
    return (
        t * axis[0]**2 + cosTheta,
        t * axis[0] * axis[1] - sinTheta * axis[2],
        t * axis[0] * axis[2] + sinTheta * axis[1],
        t * axis[0] * axis[1] + sinTheta * axis[2],
        t * axis[1]**2 + cosTheta,
        t * axis[1] * axis[2] - sinTheta * axis[0],
        t * axis[0] * axis[2] - sinTheta * axis[1],
        t * axis[1] * axis[2] + sinTheta * axis[0],
        t * axis[2]**2 + cosTheta)

def makeOpenGLMatrix(r, p):
    """
    Returns an OpenGL compatible (column-major, 4x4 homogeneous) transformation
    matrix from ODE compatible (row-major, 3x3) rotation matrix r and position
    vector p.
    """
    return (
        r[0], r[3], r[6], 0.0,
        r[1], r[4], r[7], 0.0,
        r[2], r[5], r[8], 0.0,
        p[0], p[1], p[2], 1.0)

def getBodyRelVec(b, v):
    """
    Returns the 3-vector v transformed into the local coordinate system of ODE
    body b.
    """
    return rotate3(invert3x3(b.getRotation()), v)

def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

    
    
class Obstacle(object):
    
    def __init__(self):
        self._pos = np.array([0,0,0])
        self._vel = np.array([0,0,0])
        self.shape = "arrow"
        self.radius = 0.1
        self._dir = 1.0
        self._colour = np.array([0.8, 0.3, 0.3])
        
    def setPosition(self, pos):
        self._pos = pos
        
    def getPosition(self):
        return copy.deepcopy(self._pos)
    
    def setLinearVel(self, vel):
        self._vel = vel
        
    def getLinearVel(self):
        return copy.deepcopy(self._vel)

    def setRotation(self, balh):
        pass
    
    def getRotation(self):
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    
    def getDir(self):
        return self._dir
    def setDir(self, dir):
        self._dir = dir
    def setColour(self, r, g, b):
        self._colour[0] = r
        self._colour[1] = g
        self._colour[2] = b
    def getColour(self):
        return self._colour
    

class ProjectileGame(object):
    def __init__(self, settings):
        """Creates a ragdoll of standard size at the given offset."""
        self._game_settings=settings
        # print("self._game_settings: ", self._game_settings)
        # initialize view
        from rendering import eglRenderer
        self.eglRenderer = eglRenderer.EGLRender()
        self.eglRenderer._init()
        
        if self._game_settings['render']:
            import matplotlib.pyplot as plt

            plt.ion()
            self._fig, (self._bellman_error_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
            img_ = self.eglRenderer.getPixels(0, 0, 1000, 1000)
            img_ = np.reshape(img_, (1000, 1000, 3))
            self._bellman_error_ax.imshow(img_, origin='lower')
            self._bellman_error_ax.set_title("visual Data: ")
            # self._bellman_error_ax = plt.imshow(img_, origin='lower')
            # plt.title("visual Data: ")
            plt.grid(b=True, which='major', color='black', linestyle='--')
            plt.grid(b=True, which='minor', color='g', linestyle='--')
            self._fig.set_size_inches(8.0, 8.0, forward=True)
            plt.show()
        
        self._gravity = -9.81
        # create an infinite plane geom to simulate a floor

        # create a list to store any ODE bodies which are not part of the ragdoll (this
        #   is needed to avoid Python garbage collecting these bodies)
        self._bodies = []
        
        # create a joint group for the contact joints generated during collisions
        #   between two bodies collide
        
        # set the initial simulation loop parameters
        self._fps = self._game_settings['action_fps'] * self._game_settings["timestep_subsampling"]
        self._dt = 1.0 / self._fps
        self._SloMo = 1.0
        self._Paused = False
        self._lasttime = time.time()
        self._numiter = 0
        
        self._ballRadius=0.1
        self._ballEpsilon=0.02 # Must be less than _ballRadius * 0.5
        self._state_num=0
        self._state_num_max=10

        self._agent = Obstacle()
        self._agent.shape = "sphere"
        self._agent.radius = self._ballRadius
        pos = (0.0, 0.0, 0.0)
        #pos = (0.27396178783269359, 0.20000000000000001, 0.17531818795388002)
        self._agent.setPosition(pos)
        # self._object.setRotation(rightRot)
        self._bodies.append(self._agent)
        print ("obstacle created at %s" % (str(pos)))
        # print ("total mass is %.4f kg" % (self._object.getMass().mass))
        
        ## debug visualization stuff
        self._object = Obstacle()
        self._object.setColour(0.2,0.2,0.8)
        self._object.shape = "sphere"
        self._object.radius = self._ballRadius
        pos = (0.0, 0.0, 0.0)
            #pos = (0.27396178783269359, 0.20000000000000001, 0.17531818795388002)
        self._object.setPosition(pos)
        # self._agent.setRotation(rightRot)
        self._bodies.append(self._object)
        
        self._target_velocity = 1.0 # self._game_settings['target_velocity']
        self._target_vel_weight = -10.0
        self__reward = 0
        self._time_legth = 0
        self._sim_time = 0
                
        ### Stuff related to drawing the env
        self._lookAt = (0.0, 0.0, 0) 
        self._drawAgent = True
        self._drawObject = True
        
        if ("process_visual_data" in self._game_settings
            and (self._game_settings["process_visual_data"] == True)):
            self._visual_state = [1] * self._game_settings["timestep_subsampling"]
            self._imitation_visual_state = [0.5] * self._game_settings["timestep_subsampling"]
        ### To properly compute the visual state
        self.initEpoch()
        
        self._action_bounds = self._game_settings['action_bounds']
        
        if ("use_dual_viz_state_representations" in self._game_settings
                 and (self._game_settings["use_dual_viz_state_representations"] == True)):
            ob_low = (np.prod(self._visual_state[0].shape) * len(self._visual_state)) * [0]
            ob_high = (np.prod(self._visual_state[0].shape) * len(self._visual_state)) * [1]
            observation_space = [ob_low, ob_high]
            # print ("observation_space shape: ", np.array(observation_space).shape)
            self._state_bounds = observation_space 
            self._observation_space = ActionSpace(observation_space)
            # self._state_length = np.array(self.getState()).size
        elif ("use_dual_state_representations" in self._game_settings
                 and (self._game_settings["use_dual_state_representations"] == True)):
            self._state_bounds = self._game_settings['state_bounds']
            # self._state_length = np.array(self.getState()).size
        elif ("process_visual_data" in self._game_settings
            and (self._game_settings["process_visual_data"] == True)):
            ob_low = (np.prod(self._visual_state[0].shape) * len(self._visual_state)) * [0]
            ob_high = (np.prod(self._visual_state[0].shape) * len(self._visual_state)) * [1]
            observation_space = [ob_low, ob_high]
            # print ("observation_space shape: ", np.array(observation_space).shape)
            self._state_bounds = observation_space 
            self._observation_space = ActionSpace(observation_space)
        else:
            ob_low = [-1] * self.getEnv().getObservationSpaceSize()
            ob_high = [1] * self.getEnv().getObservationSpaceSize() 
            observation_space = [ob_low, ob_high]
            self._observation_space = ActionSpace(observation_space)
            self._action_bounds = self._game_settings['action_bounds']
            self._state_bounds = self._game_settings['state_bounds']
            # self._state_length = np.array(self.getState()).size
        
        
        self._state_length = len(self._state_bounds[0])
        self._action_length = len(self._action_bounds[0])
        
        
        
        
    def getActionSpaceSize(self):
        return self._action_length
    
    def getObservationSpaceSize(self):
        return self._state_length
    
    def getNumAgents(self):
        return 1
    
    def finish(self):
        pass
    
    def init(self):
        pass
    
    def initEpoch(self):
        self._validating=False
        self.__reward = 0
        
        pos = (0.0, 0.0, 0.0)
        #pos = (0.27396178783269359, 0.20000000000000001, 0.17531818795388002)
        self._object.setPosition(pos)
        self._agent.setPosition(pos)
        ### Generate random initial velocity for particle
        vel_x = np.random.rand(1)[0] * 4
        vel_y = (np.random.rand(1)[0] * 4) + 2.0
        vel_ = (vel_x, vel_y, 0)
        self._object.setLinearVel(vel_)
        self._agent.setLinearVel(vel_)
        self._time_legth = math.fabs((vel_[1]/self._gravity)*2) # time for rise and fall
        self._sim_time = 0
        # print ("episode velocity: ", vel_)
        # print ("episode self._time_legth: ", self._time_legth)
        """
        rotation_ = list(np.reshape(rand_rotation_matrix(), (1,9))[0])
        self._object.setRotation(rotation_)
        angularVel = rand_rotation_matrix()[0] # use first row
        self._object.setAngularVel(angularVel)
        
        tmp_vel = ( (np.random.random([1]) * (self._game_settings["velocity_bounds"][1]- self._game_settings["velocity_bounds"][0]))
                    + self._game_settings["velocity_bounds"][0])[0]
        # print ("New Initial Velocity is: ", tmp_vel)
        # vel = self._object.getLinearVel() 
        self._object.setLinearVel((tmp_vel,4.0,0.0))
        """
        
        self._state_num=0
        self._end_of_Epoch_Flag=False
        
        self._validating=False
        
        if ( "timestep_subsampling" in self._game_settings ):
            for i in range(self._game_settings["timestep_subsampling"]):
                if ("process_visual_data" in self._game_settings
                    and (self._game_settings["process_visual_data"] == True)):
                    self._visual_state[i] = self._getVisualState()
                    if (self._game_settings["also_imitation_visual_data"]):
                        self._imitation_visual_state[i] = self._getImitationVisualState()
        
    
    def getEvaluationData(self):
        """
            The best measure of improvement for this environment is the distance the 
            ball reaches.
        """
        pos = self._object.getPosition()
        return [pos[0]]
    
    def clear(self):
        pass
    
    def calcReward(self):
        return self.__reward
    
    def addAnchor(self, _anchor0, _anchor1, _anchor2):
        pass 
    
    
    def generateValidationEnvironmentSample(self, seed):
        # Hacky McHack
        self._validating=False
        
    def generateEnvironmentSample(self):
        # Hacky McHack
        pass
    
    def updateAction(self, action):
        
        vel = np.array(self._object.getLinearVel())
        new_vel = np.array([vel[0] + action[0], vel[1] + action[1]])
        new_vel = clampAction(new_vel, self._game_settings["velocity_bounds"])
        self._object.setLinearVel((new_vel[0], new_vel[1], 0))
        """
        # print ("Position Before action: ", pos)
        new_vel = np.array([vel[0] + action[0], 4.0])
        new_vel = clampAction(new_vel, self._game_settings["velocity_bounds"])
        # print("New action: ", new_vel)
        time = (new_vel[1]/9.81)*2 # time for rise and fall
        self._object.setLinearVel((new_vel[0], new_vel[1], 0))
        
        self._x = []
        self._y = []
        self._step = 0
        
        steps=16
        # hopTime=1.0
        vel = self._object.getLinearVel()
        time_ = (vel[1]/9.81)*2 # time for rise and fall
        dist = vel[0] * time_
        x = np.array(np.linspace(-0.5, 0.5, steps))
        y = np.array(list(map(self._computeHeight, x)))
        y = (y + math.fabs(float(np.amin(y)))) * action[1]
        x = np.array(np.linspace(0.0, 1.0, steps)) * dist
        # x = (x + 0.5) * action[0]
        x_ = (x + pos[0])
        self._x = x_
        self._y = y
        """
        """
        dist = new_vel[0] * time
        self._object.setPosition(pos + np.array([dist, 0.0, 0.0]))
        """
        pass
        
    def update(self):
        if ( self._end_of_Epoch_Flag ) :
            self.__reward = 0
            return self.__reward
        ### Integrate kinematic agent
        updates__ = 1
        if ("process_visual_data" in self._game_settings
            and (self._game_settings["process_visual_data"] == True)):
            updates__= self._game_settings["timestep_subsampling"]
            agent_pos = self._agent.getPosition()
            imitate_pos = self._object.getPosition()
            save_lookat = self._lookAt
        for i in range(updates__):
            pos = self._agent.getPosition()
            vel = np.array(self._agent.getLinearVel())
            vel[1] = (vel[1] + (self._gravity * self._dt))
            self._agent.setLinearVel(vel)
            pos = pos + (vel * self._dt)
            self._agent.setPosition(pos)
            
            ### integrate agent
            pos_ = self._object.getPosition()
            vel_ = np.array(self._object.getLinearVel())
            pos_ = pos_ + (vel_ * self._dt)
            # self._object.setLinearVel(vel)
            self._object.setPosition(pos_)
            # print (pos)
            self._sim_time = self._sim_time + self._dt
            # print ("Sime time: ", self._sim_time)
    
            self._state_num=self._state_num+1
            if ("process_visual_data" in self._game_settings
            and (self._game_settings["process_visual_data"] == True)):
                self._lookAt = imitate_pos
                self._visual_state[i] = self._getVisualState()
                self._lookAt = agent_pos
                self._imitation_visual_state[i] = self._getImitationVisualState()
                self._lookAt = save_lookat
            # state = self.getState()
            # print ("state length: " + str(len(state)))
            # print (state)
        reward = self.computeReward(state=None)
        # print("reward: ", reward)
        self.__reward = reward
                
    def computeReward(self, state=None, next_state=None):
        if ( state is None ):
            pos_ = np.array(self._agent.getPosition())
            pos = np.array(self._object.getPosition())
        else:
            pos_ = np.array([state[0], state[1], 0])
            pos = np.array([state[4], state[5], 0])
        d = dist3(pos, pos_)
        vel_dif = np.abs(pos - pos_)
        reward = math.exp((d*d)*self._target_vel_weight)
        return reward
    
    def getSimState(self):
        state = [self._sim_time, self._time_legth]
        # state.append(self._sim_time)
        pos1 = self._object.getPosition()
        state.extend(pos1)
        vel1 = self._object.getLinearVel()
        state.extend(vel1)
        pos2 = self._agent.getPosition()
        state.extend(pos2)
        vel2 = self._agent.getLinearVel()
        state.extend(vel2)
        # print ("get sim State: " , state)
        return state
        
    def setSimState(self, state_):
        # print ("set sim State: " , state_)
        self._sim_time = state_[0]
        self._time_legth = state_[1]
        start_index = 2
        self._object.setPosition(state_[start_index:start_index+3])
        start_index = 5
        self._object.setLinearVel(state_[start_index:start_index+3])
        start_index = 8
        self._agent.setPosition(state_[start_index:start_index+3])
        start_index = 11
        self._agent.setLinearVel(state_[start_index:start_index+3])
        
    def display(self, redraw=True):
        pos = self._agent.getPosition()
        pos2 = self._object.getPosition()
        
        self.eglRenderer.setPosition(pos[0], pos[1], 0)
        self.eglRenderer.setPosition2(pos2[0], pos2[1], 0)
        self.eglRenderer.setCameraPosition(self._lookAt[0], self._lookAt[1], self._lookAt[2])
        self.eglRenderer.draw()
        if (self._game_settings['render'] and redraw):
            # self._object.setPosition([self._x[self._step], self._y[self._step], 0.0] )
            self.onDraw()
            
    def endOfEpoch(self):
        pos = self._object.getPosition()
        self.agentHasFallen()
        if (self.agentHasFallen()):
            return True 
        else:
            return False  
        
    def agentHasFallen(self):
        if ( (self._sim_time > self._time_legth)):
            self._end_of_Epoch_Flag=True # kind of hacky way to end Epoch after the ball falls in a hole.
            return True
    
        return False
        
    def onKey(c, x, y):
        """GLUT keyboard callback."""
    
        global SloMo, Paused
    
        # set simulation speed
        if c >= '0' and c <= '9':
            SloMo = 4 * int(c) + 1
        # pause/unpause simulation
        elif c == 'p' or c == 'P':
            Paused = not Paused
        # quit
        elif c == 'q' or c == 'Q':
            sys.exit(0)
    
    
    def onDraw(self):
        """GLUT render callback."""
        ### drawing using matplotlib...
        pos = self._agent.getPosition()
        pos2 = self._object.getPosition()
        self.eglRenderer.setPosition(pos[0], pos[1], 0)
        self.eglRenderer.setPosition2(pos2[0], pos2[1], 0)
        
        self.eglRenderer.setCameraPosition(self._lookAt[0], self._lookAt[1], self._lookAt[2])
        self.eglRenderer.setDrawAgent(True)
        self.eglRenderer.setDrawObject(True)
        self.eglRenderer.draw()
        img_ = self.eglRenderer.getPixels(0, 0, 1000, 1000)
        img_ = np.reshape(img_, (1000, 1000, 3))
        ax_img = self._bellman_error_ax.images[-1]
        ax_img.set_data(img_)
        # ax_img = self._bellman_error_ax.set_data(img_)
        # self._bellman_error_ax.canvas.draw()
        self._fig.canvas.draw()
        
    def _computeHeight(self, action_):
        init_v_squared = (action_*action_)
        # seconds_ = 2 * (-self._box.G)
        return (-init_v_squared)/1.0  
    
    def _computeTime(self, velocity_y):
        """
            Time till ball stops moving/ reaches apex
        """
        seconds_ = velocity_y/-self._gravity
        return seconds_  
    
    def simulateAction(self, action):
        """
            Returns True if a contact was detected
        
        """
        if self._Paused:
            return
        t = self._dt - (time.time() - self._lasttime)    
        if self._game_settings['render']:
            if (t > 0):
                time.sleep(t)
        ##
        """ 
        if self._game_settings['render']:
            pos = self._object.getPosition()
            steps=50
            # hopTime=1.0
            vel = self._object.getLinearVel()
            time_ = (vel[1]/9.81)*2 # time for rise and fall
            dist = vel[0] * time_
            x = np.array(np.linspace(-0.5, 0.5, steps))
            y = np.array(list(map(self._computeHeight, x)))
            y = (y + math.fabs(float(np.amin(y)))) * action[1]
            x = np.array(np.linspace(0.0, 1.0, steps)) * dist
            # x = (x + 0.5) * action[0]
            x_ = (x + pos[0])
            for i in range(steps):
                ## Draw the ball arch
                self._object.setPosition([x_[i], y[i], 0.0] )
                pos_ = self._object.getPosition()
                # print ("New obstacle position: ", pos_)
                
                self.onDraw()
        """
        return True
        
        
    def visualizeNextState(self, terrain, action, terrain_dx):
        # self._object.setLinearVel((action[0],4.0,0.0))
        time_ = (4.0/9.81)*2 # time for rise and fall
        
    def visualizeState(self, terrain, action, terrain_dx):
        pos = self._object.getPosition() 
        # self._object.setLinearVel((action[0],4.0,0.0))
        time_ = 0
        
    def getCharacterState(self):
        # add velocity
        state_ = []
        pos = self._agent.getPosition()
        state_.append(pos[0])
        state_.append(pos[1])
        vel = self._agent.getLinearVel()
        state_.append(vel[0])
        state_.append(vel[1])
        return state_
    
    def getKinCharacterState(self):
        # add velocity
        state_ = []
        pos = self._object.getPosition()
        state_.append(pos[0])
        state_.append(pos[1])
        vel = self._object.getLinearVel()
        state_.append(vel[0])
        state_.append(vel[1])
        return state_
    
    def getDiffState(self):
        state_ = []
        pos = self._object.getPosition()
        pos2 = self._agent.getPosition()
        state_.append(pos[0]-pos2[0])
        state_.append(pos[1]-pos2[1])
        vel = self._object.getLinearVel()
        vel2 = self._agent.getLinearVel()
        state_.append(vel[0]-vel2[0])
        state_.append(vel[1]-vel2[0])
        return state_
    
    
    def getState(self):
        """ get the next self._num_points points"""
        state = []
        if ("process_visual_data" in self._game_settings
            and (self._game_settings["process_visual_data"] == True)):
            
            if (("use_dual_viz_state_representations" in self._game_settings
                 and (self._game_settings["use_dual_viz_state_representations"] == True))):

                state = []
                            
                ob = np.array(self.getVisualState())
                ob = ob.flatten()
                state.append(ob)
                
                ob = np.array(self.getImitationVisualState())
                ob = ob.flatten()
                state.append(ob)
                
                return [state]
            elif ( "use_dual_state_representations" in self._game_settings
                 and (self._game_settings["use_dual_state_representations"] == True)):
            
                charState = self.getCharacterState()
                kincharState = self.getKinCharacterState()
                diffState = self.getDiffState()
                state_ = []
                state_.extend(charState)
                state_.extend(kincharState)
                state_.extend(diffState)
                state.append(np.array(state_))
                
                ob = np.array(self.getVisualState())
                ob = np.reshape(np.array(ob), (-1, ob.size))
                state.append(ob)
                
                return [state]
        

        if ("process_visual_data" in self._game_settings
            and (self._game_settings["process_visual_data"] == True)):
            # print("Getting visual state")
            ob = np.array(self.getVisualState())
            ob = np.reshape(np.array(ob), (-1, ob.size))
            # print("ob shape: ", ob.shape)
            return ob
        pos = self._object.getPosition()
        charState = self.getCharacterState()
        kincharState = self.getKinCharacterState()
        diffState = self.getDiffState()
        state = []
        state.extend(charState)
        state.extend(kincharState)
        state.extend(diffState)
        return state
    
    def setRandomSeed(self, seed):
        """
            Set the random seed for the simulator
            This is helpful if you are running many simulations in parallel you don't
            want them to be producing the same results if they all init their random number 
            generator the same.
        """
        random.seed(seed)
        np.random.seed(seed)

    def getVisualState(self):
        return self._visual_state

    def getImitationVisualState(self):
        return self._imitation_visual_state
    
    def getImitationState(self):
        return self.getKinCharacterState()
        
    def getViewData(self):
        from skimage.measure import block_reduce
        ### Get pixel data from view
        img = self.eglRenderer.getPixels(self._game_settings["image_clipping_area"][0],
                           self._game_settings["image_clipping_area"][1], 
                           self._game_settings["image_clipping_area"][2], 
                           self._game_settings["image_clipping_area"][3])
        # print (img)
        # img = np.asarray(img)/255
        # print ("img shape:", np.array(img).shape)
        # assert(np.sum(img) > 0.0)
        ### reshape into image, colour last
        img = np.reshape(img, (self._game_settings["image_clipping_area"][3], 
                           self._game_settings["image_clipping_area"][2], 3)) / 255
        ### downsample image
        img = block_reduce(img, block_size=(self._game_settings["downsample_image"][0], 
                                            self._game_settings["downsample_image"][1], 
                                            self._game_settings["downsample_image"][2]), func=np.mean)
        ### convert to greyscale
        if (self._game_settings["convert_to_greyscale"]):
            img = np.mean(img, axis=2)
        return img
    
    def _getVisualState(self):
        ### toggle things that we don't want in the rendered image
        ### Yes the Agent is the red triangle (the object)
        self.eglRenderer.setDrawAgent(False)
        self.eglRenderer.setDrawObject(True)
        self.display(redraw=False)
        img = self.getViewData()
        # self.render()
        return img
    
    def _getImitationVisualState(self):
        self.eglRenderer.setDrawAgent(True)
        self.eglRenderer.setDrawObject(False)
        self.display(redraw=False)
        img = self.getViewData()
        return img
    