
import numpy as np

class ActionSpace(object):
    """
        Wrapper for the action space of an env
    """
    
    def __init__(self, action_space):
        self._minimum = np.array(action_space[0])
        self._maximum = np.array(action_space[1])
        
    def getMinimum(self):
        return self._minimum

    def getMaximum(self):
        return self._maximum
    
    @property
    def low(self):
        return self._minimum
    
    @property
    def high(self):
        return self._maximum

class EnvWrapper(object):
    """
        Wrapper for the TerrainRLSim env to make function calls more simple
    """
    def __init__(self, sim, render=False, config=None):
        
        self._sim = sim
        self._render = render
        self._done = None
        
        act_low = [-1] * self.getEnv().getActionSpaceSize()
        act_high = [1] * self.getEnv().getActionSpaceSize() 
        action_space = [act_low, act_high]
        self._action_space = ActionSpace(action_space)
        ob_low = [-1] * self.getEnv().getObservationSpaceSize()
        ob_high = [1] * self.getEnv().getObservationSpaceSize() 
        observation_space = [ob_low, ob_high]
        self._observation_space = ActionSpace(observation_space)
        self._config = config
        
    def render(self):
        if (self._render):
            self._sim.display()
        
    def updateAction(self, action):
        # print ("step action: ", action)

        self._sim.updateAction(action)
            
        # self._sim.handleUpdatedAction()
        
    def update(self):
        self._sim.update()
        if (self._sim.getNumAgents() > 0): ### Multi Character simulation
            ### End of epoch when first agent falls
            """
            for a in range(self._sim.getNumAgents()):
                if (self._sim.endOfEpochForAgent(a)):
                    self._done = True
                    return
            self._done = False
            """
            ### End of epoch when last agent falls
            """
            fall_ = True
            for a in range(self._sim.getNumAgents()):
                fall_ = fall_ and self._sim.endOfEpochForAgent(a)
            self._done = fall_
            """
            ### End epoch when half of agents have fallen
            fall_s = 0
            for a in range(self._sim.getNumAgents()):
                if ( self._sim.endOfEpochForAgent(a) ):
                    fall_s = fall_s + 1
            if ( fall_s >= (self._sim.getNumAgents()/2.0)):
                self._done = True
                return
            else:
                self._done = False
        else:
            self._done = self._done or self._sim.agentHasFallen()
        # self.render()
        # print("Trying to render...")
        
    def getObservation(self):
        ob = []

        ob = self._sim.getState()
        ob = np.reshape(np.array(ob), (-1, self.getEnv().getObservationSpaceSize()))
            # ob = np.asarray(ob)
        return ob
    
    def step(self, action):
        """
            Adding multi character support
        """
        # action = action[0]
        action = np.array(action, dtype="float64")
        # print ("step action: ", action)
        self.updateAction(action)
        
        # for i in range(15):
        if ( "control_return" in self._config and (self._config["control_return"] == True) ):
            i=0
            while ( (not self._sim.needUpdatedAction()) and (i < 50 )):
                # print ("Controlling return")
                self._sim.update()
                self.render()
                i=i+1
        else:
            self._sim.update()
            self.render()
        # if ( self._render == True ):
        #    self._sim.display()
        # print("Num Agents: ", self._sim.getNumAgents())
        
        ob = self.getObservation()    
        reward = self.calcRewards()
            
        self._done = self._sim.agentHasFallen() or self._done
        # observation, reward, done, info
        # ob = np.array(ob)
        # print ("ob shape: ", ob.shape)
        return ob, reward, self._done, None
        
    def calcRewards(self):

        reward = self._sim.calcReward()
            
        return reward
        
    def reset(self):
        self._sim.initEpoch()
        self._done = False
        ob = self.getObservation()
        return ob
    
    def initEpoch(self):
        self.reset()
        
    def finish(self):
        """
            Unload simulation, free memory.
        """
        self._sim.finish()
        
    def getActionSpace(self):
        return self._action_space
    
    @property
    def action_space(self):
        return self._action_space
    
    @property
    def observation_space(self):
        return self._observation_space
    
    def endOfEpoch(self):
        return self._done
        
    def init(self):
        self._sim.init()
        
    def getEnv(self):
        return self._sim
    
    def onKeyEvent(self, c, x, y):
        self.getEnv().onKeyEvent(c, x, y)
        
    def setRandomSeed(self, seed):
        """
            Set the random seed for the simulator
            This is helpful if you are running many simulations in parallel you don't
            want them to be producing the same results if they all init their random number 
            generator the same.
        """
        # print ( "Setting random seed: ", seed )
        self.getEnv().setRandomSeed(seed)
    
    def getNumberofAgents(self):
        return self._sim.getNumAgents()
    
    
def getEnvsList():
    import os, sys, json
    from rlsimenv.config import SIMULATION_ENVIRONMENTS
    
    env_data = json.loads(SIMULATION_ENVIRONMENTS)
    
    return env_data

def getEnv(env_name, render=False):
    import os, sys, json
    
    env_data = getEnvsList()
    # print("Envs: ", json.dumps(env_data, indent=2))

    if (env_name in env_data):
        config_file = env_data[env_name]['config_file']
    else:
        print("Env: ", env_name, " not found. Check that you have the correct env name.")
        return None
    settings = env_data[env_name]
    settings['render'] = render
    if (env_data[env_name]['sim_name'] == 'NavGame'):
        from rlsimenv.NavGame import NavGame
        sim = NavGame(settings=env_data[env_name])
    elif (env_data[env_name]['sim_name'] == 'ParticleGame'):
        from rlsimenv.ParticleGame import ParticleGame
        sim = ParticleGame(settings=env_data[env_name])
    elif (env_data[env_name]['sim_name'] == 'GapGame2D'):
        from rlsimenv.GapGame2D import GapGame2D
        sim = GapGame2D(settings=env_data[env_name])
    elif ( env_data[env_name]['sim_name'] == 'NavGameMultiAgent'):
        from rlsimenv.NavGameMultiAgent import NavGameMultiAgent
        sim = NavGameMultiAgent(settings=env_data[env_name])
    else:
        print ("Env does not match a simulation environment type")
        return None
    
    ## place holder  
    # sim.setRender(render)
    sim.init()
    
    sim_ = EnvWrapper(sim, render=render, config=env_data[env_name])
    
    return sim_
