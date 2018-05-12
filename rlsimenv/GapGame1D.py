"""
A 2D bouncing ball environment

"""


import sys, os, random, time
from math import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
# from twisted.protocols import stateful
import copy
import math

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


# rotation directions are named by the third (z-axis) row of the 3x3 matrix,
#   because ODE capsules are oriented along the z-axis
rightRot = (0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
leftRot = (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
upRot = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
downRot = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
bkwdRot = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

# axes used to determine constrained joint rotations
rightAxis = (1.0, 0.0, 0.0)
leftAxis = (-1.0, 0.0, 0.0)
upAxis = (0.0, 1.0, 0.0)
downAxis = (0.0, -1.0, 0.0)
bkwdAxis = (0.0, 0.0, 1.0)
fwdAxis = (0.0, 0.0, -1.0)


# polygon resolution for capsule bodies
CAPSULE_SLICES = 16
CAPSULE_STACKS = 12

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

def draw_body(body):
    """Draw an ODE body."""
    glColor3f(0.8, 0.3, 0.3)
    # rot = makeOpenGLMatrix(body.getRotation(), body.getPosition())
    glPushMatrix()
    # glMultMatrixd(rot)
    if body.shape == "capsule":
        cylHalfHeight = body.length / 2.0
        glBegin(GL_QUAD_STRIP)
        for i in range(0, CAPSULE_SLICES + 1):
            angle = i / float(CAPSULE_SLICES) * 2.0 * pi
            ca = cos(angle)
            sa = sin(angle)
            glNormal3f(ca, sa, 0)
            glVertex3f(body.radius * ca, body.radius * sa, cylHalfHeight)
            glVertex3f(body.radius * ca, body.radius * sa, -cylHalfHeight)
        glEnd()
        glTranslated(0, 0, cylHalfHeight)
        glutSolidSphere(body.radius, CAPSULE_SLICES, CAPSULE_STACKS)
        glTranslated(0, 0, -2.0 * cylHalfHeight)
        glutSolidSphere(body.radius, CAPSULE_SLICES, CAPSULE_STACKS)
    elif body.shape == "sphere":
        pos = body.getPosition()
        glTranslated(pos[0], pos[1], pos[2])
        glutSolidSphere(body.radius, CAPSULE_SLICES, CAPSULE_STACKS)
    elif body.shape == "rectangle":
        sx,sy,sz = body.boxsize
        glScalef(sx, sy, sz)
        glutSolidCube(1)
    elif body.shape == "arrow":
        glColor3f(body.getColour()[0], body.getColour()[1], body.getColour()[2])
        glTranslatef(0, -0.1, 1.1)
        pos = body.getPosition()
        glTranslated(pos[0], pos[1], pos[2])
        glRotatef(90 * body.getDir(), 0, 1, 0)
        glutSolidCone(body.radius, body.radius, CAPSULE_SLICES, CAPSULE_STACKS )
    else:
        print( "Don't know how to draw ", body.shape, " bodies.")
    glPopMatrix()
    
def generateTerrainVerts(terrainData_, translateX):
    
    terrainScale=0.1
    verts=[]
    i=0
    verts.append([(i*terrainScale)+translateX, terrainData_[i], -1.0])
    verts.append([(i*terrainScale)+translateX, terrainData_[i], 1.0])
    faces=[]
    for i in range(1, len(terrainData_)):
        verts.append([(i*terrainScale)+translateX, terrainData_[i], -1.0])
        verts.append([(i*terrainScale)+translateX, terrainData_[i], 1.0])
        j=2*i
        face=[]
        face.append(j-2)
        face.append(j-1)
        face.append(j)
        faces.append(face)
        
        face=[]
        face.append(j)
        face.append(j-1)
        face.append(j+1)
        faces.append(face) 
        
    return verts, faces


def drawTerrain(terrainData, translateX, translateY=0.0, translateZ=0.0, colour=(0.4, 0.4, 0.8, 0.0), wirefame=False):
    
    terrainScale=0.1
    glColor4f(colour[0], colour[1], colour[2], colour[3])
    verts, faces = generateTerrainVerts(terrainData, translateX)
    if (wirefame):
        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    glBegin(GL_TRIANGLES)
    for face in faces:
        # j=i*3
        # glNormal3f(0, 1.0, 0)
        v0 = verts[face[0]]
        glVertex3f(v0[0],v0[1]+translateY,v0[2]+translateZ) #;//triangle one first vertex
        v0 = verts[face[1]]
        glVertex3f(v0[0],v0[1]+translateY,v0[2]+translateZ) #;//triangle one first vertex
        v0 = verts[face[2]]
        glVertex3f(v0[0],v0[1]+translateY,v0[2]+translateZ) #;//triangle one first vertex
    glEnd()
    
    # draw side of terrain
    glColor4f(colour[0], colour[1], colour[2], colour[3])
    glBegin(GL_QUAD_STRIP)
    for i in range(0, len(terrainData)):

        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f((i*terrainScale)+translateX, terrainData[i]+translateY, 1.0+translateZ)
        glVertex3f((i*terrainScale)+translateX, -10+translateY, 1.0+translateZ)
    glEnd()
    
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
    
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
    

class GapGame1D(object):
    def __init__(self, settings):
        """Creates a ragdoll of standard size at the given offset."""
        self._game_settings=settings
        # initialize GLUT
        if self._game_settings['render']:
            glutInit([])
            glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE)
        
        self._terrainScale=self._game_settings["terrain_scale"]
        self._terrainParameters=self._game_settings['terrain_parameters']
        
        self._action_bounds = self._game_settings['action_bounds']
        self._state_bounds = self._game_settings['state_bounds']
        self._state_length = len(self._state_bounds[0])
        self._action_length = len(self._action_bounds[0])
        
        # create the program window
        if self._game_settings['render']:
            x = 0
            y = 0
            self._window_width = 1200
            self._window_height = 400
            glutInitWindowPosition(x, y);
            glutInitWindowSize(self._window_width, self._window_height);
            glutCreateWindow(None)
        
        self._gravity = -9.81
        # create an infinite plane geom to simulate a floor
        """
        self._terrainStartX=0.0
        self._terrainStripIndex=-1
        self._terrainData = self.generateValidationTerrain(12)
        verts, faces = generateTerrainVerts(self._terrainData)
        self._terrainMeshData = ode.TriMeshData()
        self._terrainMeshData.build(verts, faces)
        """
        self._terrainMeshData = None
        self._floor = None
        self._terrainData = []
        self._nextTerrainData = []
        
        # create a list to store any ODE bodies which are not part of the ragdoll (this
        #   is needed to avoid Python garbage collecting these bodies)
        self._bodies = []
        
        # create a joint group for the contact joints generated during collisions
        #   between two bodies collide
        
        # set the initial simulation loop parameters
        self._fps = 60
        self._dt = 1.0 / self._fps
        self._stepsPerFrame = 16
        self._SloMo = 1.0
        self._Paused = False
        self._lasttime = time.time()
        self._numiter = 0
        
        # create the ragdoll
        # ragdoll = RagDoll(world, space, 500, (0.0, 0.9, 0.0))
        # print ("total mass is %.1f kg (%.1f lbs)" % (ragdoll.totalMass, ragdoll.totalMass )* 2.2)
        
        
        # set GLUT callbacks
        # glutKeyboardFunc(onKey)
        # glutDisplayFunc(onDraw)
        # glutIdleFunc(onIdle)
        
        # enter the GLUT event loop
        # Without this there can be no control over the key inputs
        # glutMainLoop()
        
        self._ballRadius=0.05
        self._ballEpsilon=0.02 # Must be less than _ballRadius * 0.5
        self._state_num=0
        self._state_num_max=10
        self._num_points=self._game_settings['num_terrain_samples']
        

        self._obstacle = Obstacle()
        self._obstacle.shape = "sphere"
        self._obstacle.radius = self._ballRadius
        pos = (0.0, 0.0, 0.0)
        #pos = (0.27396178783269359, 0.20000000000000001, 0.17531818795388002)
        self._obstacle.setPosition(pos)
        # self._obstacle.setRotation(rightRot)
        self._bodies.append(self._obstacle)
        print ("obstacle created at %s" % (str(pos)))
        # print ("total mass is %.4f kg" % (self._obstacle.getMass().mass))
        
        ## debug visualization stuff
        self._obstacle2 = Obstacle()
        self._obstacle2.setColour(0.2,0.2,0.8)
        pos = (0.0, self._ballRadius+self._ballEpsilon, 0.0)
            #pos = (0.27396178783269359, 0.20000000000000001, 0.17531818795388002)
        self._obstacle2.setPosition(pos)
        self._obstacle2.setRotation(rightRot)
        self._bodies.append(self._obstacle2)
        
        self._obstacles = []
        num_obstacles = 10
        for n in range(num_obstacles):
            obs_ = Obstacle()

            pos = (0.0, self._ballRadius+self._ballEpsilon, 0.0)
            #pos = (0.27396178783269359, 0.20000000000000001, 0.17531818795388002)
            obs_.setPosition(pos)
            obs_.setRotation(rightRot)
            self._bodies.append(obs_)
            self._obstacles.append(obs_)
            
        self._target_velocity = 1.0 # self._game_settings['target_velocity']
        self._target_vel_weight = -2.0
        self__reward = 0
        
    def setTargetVelocity(self, target_vel):
        self._target_velocity = target_vel
        
    def getActionSpaceSize(self):
        return self._action_length
    
    def getObservationSpaceSize(self):
        return self._state_length
    
    def getNumAgents(self):
        return 0
    
    def finish(self):
        pass
    
    def init(self):
        pass
    
    def initEpoch(self):
        self._terrainStartX=0.0
        self._terrainStripIndex=0
        self._validating=False
        # self.generateValidationTerrain()
        self.generateTerrain()
        self.__reward = 0
        
        pos = (0.0, 0.0, 0.0)
        #pos = (0.27396178783269359, 0.20000000000000001, 0.17531818795388002)
        self._obstacle.setPosition(pos)
        self._obstacle.setLinearVel(pos)
        """
        rotation_ = list(np.reshape(rand_rotation_matrix(), (1,9))[0])
        self._obstacle.setRotation(rotation_)
        angularVel = rand_rotation_matrix()[0] # use first row
        self._obstacle.setAngularVel(angularVel)
        # self._terrainData = []
        # self._terrainStartX=0.0
        # self._terrainStripIndex=0
        
        tmp_vel = ( (np.random.random([1]) * (self._game_settings["velocity_bounds"][1]- self._game_settings["velocity_bounds"][0]))
                    + self._game_settings["velocity_bounds"][0])[0]
        # print ("New Initial Velocity is: ", tmp_vel)
        # vel = self._obstacle.getLinearVel() 
        self._obstacle.setLinearVel((tmp_vel,4.0,0.0))
        """
        
        self._state_num=0
        self._end_of_Epoch_Flag=False
        
        self._validating=False
        
        # self.generateTerrain()
    
    def getEvaluationData(self):
        """
            The best measure of improvement for this environment is the distance the 
            ball reaches.
        """
        pos = self._obstacle.getPosition()
        return [pos[0]]
    
    def clear(self):
        pass
    
    def calcReward(self):
        return self.__reward
    
    def addAnchor(self, _anchor0, _anchor1, _anchor2):
        pass 
    
    
    def generateValidationEnvironmentSample(self, seed):
        # Hacky McHack
        self._terrainStartX=0.0
        self._terrainStripIndex=0
        self._validating=False
        self.generateValidationTerrain(seed)
        
    def generateEnvironmentSample(self):
        # Hacky McHack
        self._terrainStartX=0.0
        self._terrainStripIndex=0
        
        self.generateTerrain()
    
        
    def endOfEpoch(self):
        pos = self._obstacle.getPosition()
        self.agentHasFallen()
        # start = (pos[0]/self._terrainScale)+1 
        # assert start+self._num_points+1 < (len(self._terrainData)), "Ball is exceeding terrain length %r after %r actions" % (start+self._num_points+1, self._state_num)
        if (self.agentHasFallen()):
            return True 
        else:
            return False  
        
    def glut_print(self, x,  y,  font,  text, r,  g , b , a):

        blending = False 
        if glIsEnabled(GL_BLEND) :
            blending = True
    
        #glEnable(GL_BLEND)
        glColor3f(r,g,b)
        glWindowPos2f(x,y)
        for ch in text :
            glutBitmapCharacter( font , ctypes.c_int( ord(ch) ) )
    
        if not blending :
            glDisable(GL_BLEND) 
                    
    def prepare_GL(self):
        """Setup basic OpenGL rendering with smooth shading and a single light."""
    
        glClearColor(0.8, 0.8, 0.9, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)
    
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-6, 6.0, -2.0, 2.0, 1.0, 100);
        # gluPerspective (45.0, 1.3333, 0.2, 20.0)
    
        glViewport(0, 0, self._window_width, self._window_height)
    
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
    
        glLightfv(GL_LIGHT0,GL_POSITION,[0, 0, 1, 0])
        glLightfv(GL_LIGHT0,GL_DIFFUSE,[1, 1, 1, 1])
        glLightfv(GL_LIGHT0,GL_SPECULAR,[1, 1, 1, 1])
        glEnable(GL_LIGHT0)
    
        glEnable(GL_COLOR_MATERIAL)
        glColor3f(0.8, 0.8, 0.8)
    
        pos = self._obstacle.getPosition()
        x_adjust=4.5
        gluLookAt(pos[0]+x_adjust, 0.0, 8.0, pos[0]+x_adjust, 0.0, -10.0, 0.0, 1.0, 0.0)
        
        y_adjust=15
        y_start = 5
        vel = self._obstacle.getLinearVel()
        self.glut_print( 5 , y_start , GLUT_BITMAP_9_BY_15 , "Vel: " + str(vel) , 0.0 , 0.0 , 0.0 , 1.0 )
        y_start += y_adjust
        self.glut_print( 5 , y_start , GLUT_BITMAP_9_BY_15 , "Target Vel: " + str(self._target_velocity) , 0.0 , 0.0 , 0.0 , 1.0 )
        
    def actContinuous(self, action, bootstrapping=False):
        """
            
        """
        if ( self._end_of_Epoch_Flag ) :
            return 0
        # print ("Action: ", action)
        pos = self._obstacle.getPosition()
        vel = self._obstacle.getLinearVel()
        # print ("Position Before action: ", pos)
        new_vel = np.array([action[0], 4.0])
        new_vel = clampAction(new_vel, self._game_settings["velocity_bounds"])
        # print("New action: ", new_vel)
        time = (new_vel[1]/9.81)*2 # time for rise and fall
        self._obstacle.setLinearVel((new_vel[0], new_vel[1], 0))
        ## Move forward along X
        self.simulateAction(new_vel)
        dist = new_vel[0] * time
        self._obstacle.setPosition(pos + np.array([dist, 0.0, 0.0]))
        # print ("Position After action: ", pos + np.array([dist, 0.0, 0.0]))

        # print (pos)

        # self._terrainData = self.generateTerrain()
        self._state_num=self._state_num+1
        # state = self.getState()
        # print ("state length: " + str(len(state)))
        # print (state)
        vel_dif  = np.abs(self._target_velocity - new_vel[0])
        reward = math.exp((vel_dif*vel_dif)*self._target_vel_weight)
        self.__reward = reward
        return reward
        # obstacle.addForce((0.0,100.0,0.0))
        
    def agentHasFallen(self):
        start = self.getTerrainIndex()
        # print ("Terrain start index: ", start)
        # print ("Terrain Data: ", self._terrainData)
        if ( (self._terrainData[start-1] < -0.1) or 
             (self._terrainData[start] < -0.1 )):
            self._end_of_Epoch_Flag=True # kind of hacky way to end Epoch after the ball falls in a hole.
            return True
    
        return False
        
    def calcDist(self, bootstrapping=False):
        pos = self._obstacle.getPosition()
        vel = self._obstacle.getLinearVel()
        # contacts = ode.collide(self._floor, self._obsgeom)
        # print ("Ball velocity:", vel, " Ball position: ", pos)
        if (self.agentHasFallen() ): # fell in a hole
        # if (pos[1] < (0.0)): # fell in a hole
            # print ("Ball Fell in hole: ", pos[1])
            # if (not bootstrapping):
            self._end_of_Epoch_Flag=True # kind of hacky to end Epoch after the ball falls in a hole.
            # return 0
        
        vel = vel[0] 
        
        # reward = 1.0 - (fabs(vel[0] - targetVel)/targetVel)
        return vel
        
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
        self.prepare_GL()
    
        for b in self._bodies:
            draw_body(b)
        # for b in ragdoll.bodies:
        #     draw_body(b)
        drawTerrain(self._terrainData, self._terrainStartX)
        if ( len(self._nextTerrainData) > 0):
            drawTerrain(self._nextTerrainData, self._nextTerrainStartX, translateY=0.02, translateZ=0.02, colour=(0.9, 0.2, 0.9, 0.0), wirefame=True)
        
        # state = self.getState()
        # pos = self._obstacle.getPosition()
        # drawTerrain(state, pos[0]+state[len(state)-1], translateY=0.1, colour=(0.1, 0.8, 0.1, 0.1))
        # drawTerrain(state, pos[0], translateY=0.0, colour=(0.1, 0.8, 0.1, 0.1))
    
        glutSwapBuffers()
    
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
        if self._game_settings['render']:
            pos = self._obstacle.getPosition()
            steps=50
            # hopTime=1.0
            vel = self._obstacle.getLinearVel()
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
                self._obstacle.setPosition([x_[i], y[i], 0.0] )
                pos_ = self._obstacle.getPosition()
                # print ("New obstacle position: ", pos_)
                
                glutPostRedisplay()
                self.onDraw()
        return True
        
        
    def visualizeNextState(self, terrain, action, terrain_dx):
        self._nextTerrainData = terrain
        pos = self._obstacle.getPosition() 
        # self._obstacle.setLinearVel((action[0],4.0,0.0))
        time_ = (4.0/9.81)*2 # time for rise and fall
        self._nextTerrainStartX = pos[0] + (time_ * action[0]) + terrain_dx
        # self._nextTerrainStartX = pos[0] + terrain_dx
        # drawTerrain(terrain, translateX, translateY=0.0, colour=(0.4, 0.4, 0.8, 0.0), wirefame=False):
        
    def visualizeState(self, terrain, action, terrain_dx):
        self._nextTerrainData = terrain
        pos = self._obstacle.getPosition() 
        # self._obstacle.setLinearVel((action[0],4.0,0.0))
        time_ = 0
        self._nextTerrainStartX = pos[0] + (time_ * action[0]) + terrain_dx
        # self._nextTerrainStartX = pos[0] + terrain_dx
        # drawTerrain(terrain, translateX, translateY=0.0, colour=(0.4, 0.4, 0.8, 0.0), wirefame=False):
    
    def generateTerrain(self):
        """
            If this is the first time this is called generate a new strip of terrain and use it.
            The second time this is called and onward generate a new strip and add to the end of the old strip.
            Also remove the beginning half of the old strip
        """
        # print ("Generating more terrain")
        terrainData_=[]
        if (self._terrainStripIndex == 0):
            # print ("Generating NEW terrain, translateX: ", self._terrainStartX)
            terrainData_ = self._generateTerrain(self._terrainParameters['terrain_length'])
            self._terrainStartX=0
        elif (self._terrainStripIndex > 0):
            # print ("Generating more terrain, translateX: ", self._terrainStartX)
            terrainData_ = self._generateTerrain(self._terrainParameters['terrain_length']/2)
            self._terrainStartX=self._terrainStartX+((len(self._terrainData)*self._terrainScale)/2.0)
            terrainData_ = np.append(self._terrainData[int(len(self._terrainData)/2):], terrainData_)
        else:
            print ("Why is the strip index < 0???")
            sys.exit()    
        
        self.setTerrainData(terrainData_)
        self._terrainStripIndex=self._terrainStripIndex+1
        
    def _generateTerrain(self, length):
        """
            Generate a single strip of terrain
        """
        terrainLength=length
        terrainData=np.zeros(int(terrainLength))
        # gap_size=random.randint(2,7)
        gap_size=self._terrainParameters['gap_size']
        # gap_start=random.randint(2,7)
        gap_start=self._terrainParameters['gap_start']
        next_gap=self._terrainParameters['distance_till_next_gap']
        for i in range(int(terrainLength/next_gap)):
            gap_start= gap_start+np.random.random_integers(self._terrainParameters['random_gap_start_range'][0],
                                                self._terrainParameters['random_gap_start_range'][1])
            gap_size=np.random.random_integers(self._terrainParameters['random_gap_width_range'][0],
                                    self._terrainParameters['random_gap_width_range'][1])
            terrainData[gap_start:gap_start+gap_size] = self._terrainParameters['terrain_change']
            gap_start=gap_start+next_gap
            
        return terrainData
    
    def generateValidationTerrain(self, seed):
        """
            If this is the first time this is called generate a new strip of terrain and use it.
            The second time this is called and onward generate a new strip and add to the end of the old strip.
            Also remove the beginning half of the old strip
        """
        """
        if (not self._validating):
            self._validating=True
            random.seed(seed)
            np.random.seed(seed)
        """
        # print ("Generating more terrain")
        terrainData_=[]
        if (self._terrainStripIndex == 0):
            # print ("Generating NEW validation terrain, translateX: ", self._terrainStartX)
            terrainData_ = self._generateValidationTerrain(self._terrainParameters['terrain_length'], seed)
            self._terrainStartX=0
        elif (self._terrainStripIndex > 0):
            # print ("Generating more validation terrain, translateX: ", self._terrainStartX)
            terrainData_ = self._generateValidationTerrain(self._terrainParameters['terrain_length']/2, seed)
            self._terrainStartX=self._terrainStartX+((len(self._terrainData)*self._terrainScale)/2.0)
            terrainData_ = np.append(self._terrainData[len(self._terrainData)/2:], terrainData_)
        else:
            print ("Why is the strip index < 0???")
            sys.exit()    
        
        
        self.setTerrainData(terrainData_)
        self._terrainStripIndex=self._terrainStripIndex+1
        
    def _generateValidationTerrain(self, length, seed):
        terrainLength=length
        terrainData=np.zeros((terrainLength))
        # gap_size=random.randint(2,7)
        gap_size=self._terrainParameters['gap_size']
        # gap_start=random.randint(2,7)
        gap_start=self._terrainParameters['gap_start']
        next_gap=self._terrainParameters['distance_till_next_gap']
        for i in range(int(terrainLength/next_gap)):
            gap_start= gap_start+np.random.random_integers(self._terrainParameters['random_gap_start_range'][0],
                                                self._terrainParameters['random_gap_start_range'][1])
            gap_size=np.random.random_integers(self._terrainParameters['random_gap_width_range'][0],
                                    self._terrainParameters['random_gap_width_range'][1])
            terrainData[gap_start:gap_start+gap_size] = self._terrainParameters['terrain_change']
            gap_start=gap_start+next_gap
        return terrainData
    
    def setTerrainData(self, data_):
        self._terrainData = data_
        # verts, faces = generateTerrainVerts(self._terrainData, translateX=self._terrainStartX)
        # del self._terrainMeshData
        # self._terrainMeshData = ode.TriMeshData()
        # self._terrainMeshData.build(verts, faces)
        # del self._floor
        # self._floor = ode.GeomTriMesh(self._terrainMeshData, self._space)
        
    def getCharacterState(self):
        # add velocity
        angularVel = []
        vel = self._obstacle.getLinearVel()
        angularVel.append(vel[0])
        return angularVel
    
    def getTerrainIndex(self):
        pos = self._obstacle.getPosition()
        return int(math.floor( (pos[0]-(self._terrainStartX) )/self._terrainScale)+1)
    
    def getState(self):
        """ get the next self._num_points points"""
        pos = self._obstacle.getPosition()
        charState = self.getCharacterState()
        num_extra_feature=1
        ## Get the index of the next terrain sample
        start = self.getTerrainIndex()
        ## Is that sample + terrain_state_size beyond the current terrain extent
        if (start+self._num_points+num_extra_feature >= (len(self._terrainData))):
            # print ("State not big enough ", len(self._terrainData))
            if (self._validating):
                self.generateValidationTerrain(0)
            else:
                self.generateTerrain()
        start = self.getTerrainIndex()
        assert start+self._num_points+num_extra_feature < (len(self._terrainData)), "Ball is exceeding terrain length %r after %r actions" % (start+self._num_points+num_extra_feature-1, self._state_num)
        # print ("Terrain Data: ", self._terrainData)
        state=np.zeros((self._num_points+num_extra_feature+len(charState)))
        if pos[0] < 0: #something bad happened...
            return state
        else: # good things are going on...
            # state[0:self._num_points] = copy.deepcopy(pos[1]-self._terrainData[start:start+self._num_points])
            state[0:self._num_points] = pos[1] - copy.deepcopy(self._terrainData[start:start+self._num_points])
            # state = copy.deepcopy(self._terrainData[start:start+self._num_points+1])
            # print ("Start: ", start, " State Data: ", state)
            ### Distance to first point in terrain features
            state[self._num_points] = fabs(float(math.floor(start)*self._terrainScale)-(pos[0]-self._terrainStartX)) # X distance to first sample
            # state[self._num_points+1] = (pos[1]) # current height of character, This was returning huge nagative values... -1.5x+14
            # print ("Dist to next point: ", state[len(state)-1])
            
            # add character State
            state[self._num_points+num_extra_feature:self._num_points+num_extra_feature+len(charState)] = charState
        
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
