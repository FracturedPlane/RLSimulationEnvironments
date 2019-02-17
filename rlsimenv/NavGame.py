# from matplotlib import mpl
import numpy as np
# import matplotlib.animation as animation
import random
import math
import sys
import time


def loadMap():
    dataset="map.json"
    import json
    dataFile = open(dataset)
    s = dataFile.read()
    data = json.loads(s)
    dataFile.close()
    return data["map"]
    
# make values from -5 to 5, for this example
# zvals = loadMap()

# print (zvals)

def resetExtent(data,im):
    """
    Using the data and axes from an AxesImage, im, force the extent and 
    axis values to match shape of data.
    """
    ax = im.get_axes()
    dataShape = data.shape

    if im.origin == 'upper':
        im.set_extent((-0.5,dataShape[0]-.5,dataShape[1]-.5,-.5))
        ax.set_xlim((-0.5,dataShape[0]-.5))
        ax.set_ylim((dataShape[1]-.5,-.5))
    else:
        im.set_extent((-0.5,dataShape[0]-.5,-.5,dataShape[1]-.5))
        ax.set_xlim((-0.5,dataShape[0]-.5))
        ax.set_ylim((-.5,dataShape[1]-.5))


class NavGame(object):
    """
        An n-d continuous grid world like navigation game
        The dimension of the world is determined from the length of state bounds
    """
    
    def __init__(self, settings):
        self._settings = settings
        # print ("Game settings: ", self._settings)
        self._action_bounds = self._settings['action_bounds']
        self._state_bounds = self._settings['state_bounds']
        self._state_length = len(self._state_bounds[0])
        # self._state_bounds = np.array([[self._state_bounds[0][0]]*self._state_length, [self._state_bounds[1][0]]*self._state_length])
        ## For plotting objects
        self._markerSize = 25
        self._map = np.zeros((int(self._state_bounds[1][0]-self._state_bounds[0][0]),
                              int(self._state_bounds[1][0]-self._state_bounds[0][0])))
        
        self._agent = np.array([2]* self._state_length) ## Somewhat random initial spot
        self._target = np.array([0]* self._state_length) ## goal location
        
        ## Some obstacles
        obstacles = []
        obstacles.append([self._state_bounds[0][0]]*self._state_length)
        obstacles.append([self._state_bounds[1][0]]*self._state_length)
        obstacles.append([4]*self._state_length)
        obstacles.append([-4]*self._state_length)
        obstacles.append([4] + ([-4]*(self._state_length-1)))
        obstacles.append([-4] + ([4]*(self._state_length-1)))
        """
        num_random_obstacles=5
        for i in range(num_random_obstacles):
            obstacles.append(np.random.random_integers(self._state_bounds[0][0], self._state_bounds[1][0], self._state_length))
        """
        self._obstacles = np.array(obstacles)
        
        # if self._settings['render'] == True:
        X,Y = self.getStateSamples()
        U = np.zeros((np.array(X).size))
        V = np.ones((np.array(X).size))
        Q = np.random.rand((np.array(X).size))
        if self._settings['render']:
            self.initRender(U, V, Q)
            
    def getActionSpaceSize(self):
        return self._state_length
    
    def getObservationSpaceSize(self):
        return self._state_length
    
    def setRandomSeed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        
    def getNumAgents(self):
        return 1
    
    def updateAction(self, action):
        self.__action = action
        
    def init(self):
        new_loc = (np.random.rand(self._state_length) - 0.5) * (8 + 8)
        self._agent = np.array(new_loc) ## Somewhat random initial spot
        self._target = np.array([0]* self._state_length) ## goal location
        # self._map[self._target[0]][self._target[1]] = 1
        
        
      
    def initEpoch(self):
        """
            Reset agent location
        """
        # new_loc = np.random.random_integers(self._state_bounds[0][0], self._state_bounds[1][0], self._state_length)
        new_loc = np.random.random_integers(-8, 8, self._state_length)
        self._agent = new_loc
        
    def generateValidationEnvironmentSample(self, seed):
        self.initEpoch()
    
    def generateEnvironmentSample(self):
        self.initEpoch()

    """        
    def getAgentLocalState(self):
        st_ = self._map[(self._agent[0]-1)::2, (self._agent[1]-1)::2] # submatrix around agent location
        st_ = np.reshape(st_, (1,)) # Make vector
        return st_
    """ 
    
    def move(self, action):
        """
        action in [0,1,2,3,4,5,6,7]
        Will only work for 2D environment
        """
        return {
            0: [-1,0],
            1: [-1,1],
            2: [0,1],
            3: [1,1],
            4: [1,0],
            5: [1,-1],
            6: [0,-1],
            7: [-1,-1],
            }.get(action, [-1,0]) 
            
    """
    def act(self, action):
        print ("Trying discrete action: ", action)
        move = np.array(self.move(action))
        # loc = self._agent + (move * random.uniform(0.5,1.0))
        loc = self._agent + (move)
        
        if (((loc[0] < self._state_bounds[0][0]) or (loc[0] > self._state_bounds[1][0]) or 
            (loc[1] < self._state_bounds[0][1]) or (loc[1] > self._state_bounds[1][1])) or
            self.collision(loc) or
            self.fall(loc)):
            # Can't move out of map
            return self.reward() + -8
            
        # if self._map[loc[0]-1][loc[1]-1] == 1:
            # Can't walk onto obstacles
        #     return self.reward() +-5
        self._agent = loc
        return self.reward()
    """
    def actContinuous(self, action, bootstrapping):
        # print ("Trying action: ", action)
        move = np.array(action)
        # loc = self._agent + (move * random.uniform(0.5,1.0))
        loc = self._agent + (move)
        
        if(
           (
            # np.any(np.less(loc, self._state_bounds[0])) or np.any(np.greater(loc, self._state_bounds[1]))) or
            np.any(np.less(loc, -8.0)) or np.any(np.greater(loc, 8.0))) or
            self.collision(loc) or
            self.fall(loc)
            ):
            ### can't overlap an obstacle or be outside working area
            self.__reward = (self._state_bounds[0][0] - self._state_bounds[1][0])/8.0
            return self.__reward
            
        # if self._map[loc[0]-1][loc[1]-1] == 1:
            # Can't walk onto obstacles
        #     return self.reward() +-5
        self._agent = loc
        
        if ( self._settings['render'] == True ):
            self.display()
        self.__reward = self.reward()
        return self.__reward
    
    def fall(self, loc):
        # Check to see if collision at loc with any obstacles
        # print (int(math.floor(loc[0])), int(math.floor(loc[1])))
        # if self._map[int(math.floor(loc[0]))][ int(math.floor(loc[1]))] < 0:
        #     return True
        return False
    
    def agentHasFallen(self):
        loc = self._agent
        return self.fall(loc)
    
    def collision(self, loc):
        # Check to see if collision at loc with any obstacles
        for obs in self._obstacles:
            a=(loc - obs)
            d = np.sqrt((a*a).sum(axis=0))
            # print ("d: ", d)
            if d < 0.2:
                # print ("Found collision")
                return True
        return False
    
    def reward(self):
        # More like a cost function for distance away from target
        # print ("Agent Loc: " + str(self._agent))
        # print ("Target Loc: " + str(self._target))
        a=(self._agent - self._target)
        d = np.sqrt((a*a).sum(axis=0))
        # print ("Dist Vector: " + str(a) + " Distance: " + str(d))
        if d < 0.3:
            return 2.0
        return -d/((self._state_bounds[1][0]- self._state_bounds[0][0])/2.0)
    
    def calcReward(self):
        return self.__reward
    
    def computeReward(self, state, next_state=None):
        a=(state - self._target)
        d = np.sqrt((a*a).sum(axis=0))
        # print ("Dist Vector: " + str(a) + " Distance: " + str(d))
        if d < 0.3:
            return 2.0
        return -d/((self._state_bounds[1][0]- self._state_bounds[0][0])/2.0)
        return reward
    
    def getSimState(self):
        state = self._agent
        # state.append(self._sim_time)
        # print ("get sim State: " , state)
        return state
        
    def setSimState(self, state_):
        # print ("set sim State: " , state_)
        self._agent = state_
    
    def getState(self):
        return self._agent
    
    def getStateForAgent(self, i):
        return self.getState()
    
    def setState(self, st):
        self._agent = st
        
    def getStateSamples(self):
        X,Y = np.mgrid[self._state_bounds[0][0]:self._state_bounds[1][0]+1,
                       self._state_bounds[0][1]:self._state_bounds[1][1]+1]
        return (X,Y)
    
    def initRender(self, U, V, Q):
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.patches as patches
        from matplotlib import cm as CM
        from matplotlib import mlab as ML
        colours = ['gray','black','blue']
        cmap = matplotlib.colors.ListedColormap(['gray','black','blue'])
        bounds=[-1,-1,1,1]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        

        plt.ion()
        
        # Two subplots, unpack the axes array immediately
        self._fig, (self._map_ax, self._policy_ax) = plt.subplots(1, 2, sharey=False)
        self._fig.set_size_inches(18.5, 8.5, forward=True)
        self._map_ax.set_title('Map')
        self._particles, = self._map_ax.plot([self._agent[0]], [self._agent[1]], 'bo', ms=self._markerSize)
        
        self._map_ax.plot([self._target[0]], [self._target[1]], 'ro', ms=self._markerSize)
        
        self._map_ax.plot(self._obstacles[:,0], self._obstacles[:,1], 'gs', ms=28)
        # self._line1, = self._ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma        
        
        cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap',
                                                  colours,
                                                 256)
        """
        img1 = self._map_ax.imshow(self._map,interpolation='nearest',
                            cmap = cmap2,
                            origin='lower')
        """
        p = patches.Rectangle(
            (self._state_bounds[0][0], self._state_bounds[0][0]),
            (self._state_bounds[1][0]-self._state_bounds[0][0]),
            (self._state_bounds[1][0]-self._state_bounds[0][0]),
            alpha=0.25,
            facecolor="#999999"
            # fill=False      # remove background
        )
        self._map_ax.add_patch(p)
        """
        img2 = self._policy_ax.imshow(np.array(self._map)*0.0,interpolation='nearest',
                            cmap = cmap2,
                            origin='lower')
        """
        p = patches.Rectangle(
            (self._state_bounds[0][0], self._state_bounds[0][0]),
            (self._state_bounds[1][0]-self._state_bounds[0][0]),
            (self._state_bounds[1][0]-self._state_bounds[0][0]),
            alpha=0.25,
            facecolor="#999999"
            # fill=False      # remove background
        )
        self._policy_ax.add_patch(p)
        
        # make a color bar
        # self._map_ax.colorbar(img2,cmap=cmap,
        #               norm=norm,boundaries=bounds,ticks=[-1,0,1])
        self._map_ax.grid(True,color='white')
        # self._policy_ax.grid(True,color='white')
        
        # fig = plt.figure()
        #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        #ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
        #                    xlim=(-3.2, 3.2), ylim=(-2.4, 2.4))
        
        # particles holds the locations of the particles
        
        self._policy_ax.set_title('Policy')
        
        X,Y = self.getStateSamples()
        print (X,Y)
        # self._policy = self._policy_ax.quiver(X[::2, ::2],Y[::2, ::2],U[::2, ::2],V[::2, ::2], linewidth=0.5, pivot='mid', edgecolor='k', headaxislength=5, facecolor='None')
        textstr = "$\max V(s,a)=%.2f$\n$\min V(s,a)=%.2f$"%(np.max(Q), np.min(Q))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
        
        # place a text box in upper left in axes coords
        self._policyText = self._policy_ax.text(0.05, 0.95, textstr, transform=self._policy_ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        q_max = np.max(Q)
        q_min = np.min(Q)
        Q = (Q - q_min)/ (q_max-q_min)
        self._policy2 = self._policy_ax.quiver(X,Y,U,V,Q, alpha=.75, linewidth=0.5, width=0.005, pivot='mid', angles='xy', linestyles='-', scale=50.0)
        # self._policy = self._policy_ax.quiver(X,Y,U,V, linewidth=0.5, pivot='mid', edgecolor='k', headaxislength=3, facecolor='None', angles='xy', linestyles='-', scale=25.0)
        
        ### Add value visulization
        # self._value_function = self._policy_ax.hexbin(X.ravel(), Y.ravel(), C=(Q.ravel()*10), gridsize=30, cmap=CM.jet, bins=None)
        # self._value_function = self._policy_ax.scatter(X, Y, c=Q)
        # PLT.axis([x.min(), x.max(), y.min(), y.max()])
        
        # Two subplots, unpack the axes array immediately
        self._fig2, (self._policy_mbae) = plt.subplots(1, 1, sharey=False)
        self._fig2.set_size_inches(8.5, 8.5, forward=True)
        """
        img2 = self._policy_mbae.imshow(np.array(self._map)*0.0,interpolation='nearest',
                            cmap = cmap2,
                            origin='lower')
        """
        p = patches.Rectangle(
            (self._state_bounds[0][0], self._state_bounds[0][0]),
            (self._state_bounds[1][0]-self._state_bounds[0][0]),
            (self._state_bounds[1][0]-self._state_bounds[0][0]),
            alpha=0.25,
            facecolor="#999999"
            # fill=False      # remove background
        )
        self._policy_mbae.add_patch(p)
        
        self._policy_mbae.set_title('MBAE Action and Advantage')
        
        X,Y = self.getStateSamples()
        print (X,Y)
        # self._policy = self._policy_ax.quiver(X[::2, ::2],Y[::2, ::2],U[::2, ::2],V[::2, ::2], linewidth=0.5, pivot='mid', edgecolor='k', headaxislength=5, facecolor='None')
        textstr2 = "$\max A(s,a)=%.2f$\n$\min A(s,a)=%.2f$"%(np.max(Q), np.min(Q))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
        
        # place a text box in upper left in axes coords
        self._mbaeText = self._policy_mbae.text(0.05, 0.95, textstr2, transform=self._policy_mbae.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        q_max = np.max(Q)
        q_min = np.min(Q)
        Q = (Q - q_min)/ (q_max-q_min)
        self._mbae2 = self._policy_mbae.quiver(X,Y,U,V,Q, alpha=.75, linewidth=0.5, width=0.01, pivot='mid', angles='xy', linestyles='-', scale=25.0)
        # self._mbae = self._policy_mbae.quiver(X,Y,U,V, linewidth=0.5, width=0.01, pivot='mid', edgecolor='k', headaxislength=3, facecolor='None', angles='xy', linestyles='-', scale=25.0)
        
        # Two subplots, unpack the axes array immediately
        self._fig3, (self._fd_error) = plt.subplots(1, 1, sharey=False)
        self._fig3.set_size_inches(8.5, 8.5, forward=True)
        self._fd2 = self._fd_error.quiver(X,Y,U,V,Q, alpha=.75, linewidth=0.5, width=0.005, pivot='mid', angles='xy', linestyles='-', scale=25.0)
        # self._fd = self._fd_error.quiver(X,Y,U,V, linewidth=0.5, pivot='mid', edgecolor='k', headaxislength=3, facecolor='None', angles='xy', linestyles='-', scale=25.0)
        self._fd_error.set_title('MBAE FD error')
        p = patches.Rectangle(
            (self._state_bounds[0][0], self._state_bounds[0][0]),
            (self._state_bounds[1][0]-self._state_bounds[0][0]),
            (self._state_bounds[1][0]-self._state_bounds[0][0]),
            alpha=0.25,
            facecolor="#999999"
            # fill=False      # remove background
        )
        self._fd_error.add_patch(p)
        
        # self._policy_ax.set_aspect(1.)
    
    def update(self):
        """perform animation step"""
        # update pieces of the animation
        # self._agent = self._agent + np.array([0.1,0.1])
        # print ("Agent loc: " + str(self._agent))
        self.__reward = self.actContinuous(self.__action, bootstrapping=False)
        if self._settings['render']:
            self._particles.set_data(self._agent[0], self._agent[1] )
            self._particles.set_markersize(self._markerSize)
        # self._line1.set_ydata(np.sin(x + phase))
        # self._fig.canvas.draw()
        
    def display(self):
        if self._settings['render']:
            self._particles.set_data(self._agent[0], self._agent[1] )
            self._particles.set_markersize(self._markerSize)
        
    def updatePolicy(self, U, V, Q):
        # self._policy.set_UVC(U[::2, ::2],V[::2, ::2])
        if self._settings['render']:
            textstr = """$\max V=%.2f$\n$\min V=%.2f$"""%(np.max(Q), np.min(Q))
            self._policyText.set_text(textstr)
            q_max = np.max(Q)
            q_min = np.min(Q)
            Q = (Q - q_min)/ (q_max-q_min)
            self._policy2.set_UVC(U, V, Q)
            # self._policy2.set_vmin(1.0)
            """
            self._policy2.update_scalarmappable()
            print ("cmap " + str(self._policy2.cmap)  )
            print ("Face colours" + str(self._policy2.get_facecolor()))
            colours = ['gray','black','blue']
            cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                                       colours,
                                                       256)
            self._policy2.cmap._set_extremes()
            """
            # self._policy.set_UVC(U, V)
            self._fig.canvas.draw()
        
        # self._value_function.set_edgecolors(Q)
        # self._value_function.set_data(Q)
        
    def updateMBAE(self, U, V, Q):
        # self._policy.set_UVC(U[::2, ::2],V[::2, ::2])
        if self._settings['render']:
            textstr = """$\max A(s,a)=%.2f$\n$\min A(s,a)=%.2f$"""%(np.max(Q), np.min(Q))
            self._mbaeText.set_text(textstr)
            q_max = np.max(Q)
            q_min = np.min(Q)
            Q = (Q - q_min)/ (q_max-q_min)
            self._mbae2.set_UVC(U, V, Q)
            # self._policy2.set_vmin(1.0)
            """
            self._policy2.update_scalarmappable()
            print ("cmap " + str(self._policy2.cmap)  )
            print ("Face colours" + str(self._policy2.get_facecolor()))
            colours = ['gray','black','blue']
            cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                                       colours,
                                                       256)
            self._policy2.cmap._set_extremes()
            """
            # self._mbae.set_UVC(U, V)
            self._fig2.canvas.draw()
        
    def updateFD(self, U, V, Q):
        # self._policy.set_UVC(U[::2, ::2],V[::2, ::2])
        # textstr = """$\max A(s,a)=%.2f$\n$\min A(s,a)=%.2f$"""%(np.max(Q), np.min(Q))
        # self._mbaeText.set_text(textstr)
        if self._settings['render']:
            q_max = np.max(Q)
            q_min = np.min(Q)
            Q = (Q - q_min)/ (q_max-q_min)
            self._fd2.set_UVC(U, V, Q)
            # self._policy2.set_vmin(1.0)
            """
            self._policy2.update_scalarmappable()
            print ("cmap " + str(self._policy2.cmap)  )
            print ("Face colours" + str(self._policy2.get_facecolor()))
            colours = ['gray','black','blue']
            cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                                       colours,
                                                       256)
            self._policy2.cmap._set_extremes()
            """
            # self._fd.set_UVC(U, V)
            self._fig3.canvas.draw()
        
    def reachedTarget(self):
        # Might be a little touchy because floats are used
        a=(self._agent - self._target)
        d = np.sqrt((a*a).sum(axis=0))
        return d <= 0.3
    
    def endOfEpoch(self):
        return self.reachedTarget()

    def saveVisual(self, fileName):
        # plt.savefig(fileName+".svg")
        if self._settings['render']:
            self._fig.savefig(fileName+".svg")
            self._fig.savefig(fileName+".png")
            if (self._settings['train_forward_dynamics']):
                self._fig2.savefig(fileName+"_MBAE.svg")
                self._fig2.savefig(fileName+"_MBAE.png")
                self._fig3.savefig(fileName+"_FD_error.svg")
                self._fig3.savefig(fileName+"_FD_error.png")
        
    def finish(self):
        pass

if __name__ == '__main__':
    import json
    settings={}
    # game = BallGame2D(settings)
    if (len(sys.argv)) > 1:
        _settings=json.load(open(sys.argv[1]))
        print (_settings)
        _settings['render']=True
        game = NavGame(_settings)
    else:
        _settings['render']=True
        game = NavGame(_settings)
    game.init()
    for j in range(100):
        # game.generateEnvironmentSample()
        game.generateValidationEnvironmentSample(j)
        print ("Starting new epoch")
        game.initEpoch()
        i=0
        while not game.endOfEpoch():
        # for i in range(50):
            # state = game.getState()
            
            # action = model.predict(state)
            _action =  (np.random.random([1])[0] -0.5) * 2
            _action2 =  (np.random.random([1])[0] -0.5) * 2
            action = [_action,_action2]
            state = game.getState()
            # pos = game._obstacle.getPosition()
            # drawTerrain(state, pos[0], translateY=0.0, colour=(0.6, 0.6, 0.9, 1.0))
            print ("State: " + str(state))
            # print ("character State: " + str(game.getCharacterState()))
            # print ("rot Vel: " + str(game._obstacle.getQuaternion()))
            
            # print (state)
            
            # game.visualizeState(state[:len(state)-1], action, state[_settings['num_terrain_samples']])
            reward = game.actContinuous(action)
            game.update()
            
            if (game.agentHasFallen()):
                print (" *****Agent fell in a hole")
            
            if ( reward < 0.00001 ):
                print("******Agent has 0 reward?")
            
            print ("Reward: " + str(reward) + " on action: ", action, " actions: ", i)
            # print ("Number of geoms in space: ", game._space.getNumGeoms())
            # print ("Random rotation matrix", list(np.reshape(rand_rotation_matrix(), (1,9))[0]))
            i=i+1
            game._lasttime = time.time()
            
    game.finish()