
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
        Wrapper for the RLSimEnv to make function calls simple
    """
    def __init__(self, sim, render=False, config=None):
        
        self._sim = sim
        self._render = render
        self._done = None
        
        self._action_space = self.getEnv().getActionSpace()
        self._observation_space = self.getEnv().getObservationSpace()
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
        if (self._sim.getNumAgents() > 1): ### Multi Character simulation
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
            if ( fall_s > (self._sim.getNumAgents()/2.0)):
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
        # print ("np.array(ob): ", np.array(ob).shape)
        if ("process_visual_data" in self._config
        and (self._config["process_visual_data"] == True)
        and ("use_dual_state_representations" in self._config
             and (self._config["use_dual_state_representations"] == True))
        and
            ("use_multimodal_state_representations" in self._config
             and (self._config["use_multimodal_state_representations"] == True))):
            ob_ = []
            ob = self._sim.getState()
            # ob = np.reshape(np.array(ob), (-1, len(ob)))
            ob_.append(ob)
            ob = np.array(self.getVisualState())
            ob = ob.flatten()
            ### Add pose state after pixel state, also me being lazy and duplicating pose data
            ob = np.concatenate((ob, self._sim.getState()), axis=0)
            # print ("vis ob shape: ", ob.shape)
            ob_.append(ob)
            return [ob_]
        elif (("use_dual_state_representations" in self._config 
            and (self._config["use_dual_state_representations"] == True))
            or ("use_dual_pose_state_representations" in self._config 
            and (self._config["use_dual_pose_state_representations"] == True))):
            # print ("np.array(ob): ", np.array(ob).shape)
            return ob
        ob = np.array(ob)
        # print ("np.array(ob): ", ob)
        # print ("self.getEnv().getObservationSpaceSize(): ", self.getEnv().getObservationSpaceSize())
        # ob = np.reshape(ob, (-1, int(self.getEnv().getObservationSpaceSize())))
        # ob = [ob.flatten()]
        # ob = np.asarray(ob)
        return ob
    
    def step(self, action):
        """
            Adding multi character support
        """
        # action = action[0]
        # print ("step action: ", action, " done: ", self._done)
        # action = np.array(action, dtype="float64")
        if ("openAIGym" in self._config
            and (self._config["openAIGym"] == True)):
            ob, reward, self._done, _ = self._sim.step(action)
            return ob, reward, self._done, None
            
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
    
    
    def computeReward(self, state, next_state):
        return self.getEnv().computeReward(state, next_state)
    
    def getSimState(self):
        return self.getEnv().getSimState()
    
    def setSimState(self, state_):
        self.getEnv().setSimState(state_)
    
    def getStateFromSimState(self, simState):
        state_ = self.getSimState()
        self.setSimState(simState)
        ob = self.getObservation()
        self.setSimState(state_) 
        return ob
    
    def getNumberofAgents(self):
        return self._sim.getNumAgents()
    
    def getVisualState(self):
        return self._sim.getVisualState()
        
    def getImitationState(self):
        return self._sim.getImitationState()
    
    def getCharacterState(self):
        return self._sim.getCharacterState()
    
    def getImitationVisualState(self):
        return self._sim.getImitationVisualState()
    
    def computeImitationReward(self, reward_func):
        """
            Uses a learned imitation based reward function to
            compute the reward in the simulation 
        """
        # print("self.getImitationState(): ", self.getVisualState())
        # print("self.getImitationVisualState(): ", self.getImitationVisualState())
        if ("use_multimodal_state_representations" in self._config
            and (self._config["use_multimodal_state_representations"] == True)):
            multi_state_ = self.getMultiModalRewardState()
            # dist = reward_func(np.reshape(self._sim.getState() ,newshape=(1, state_.size)),
            #                 np.reshape(viz_state_, newshape=(1, viz_state_.size)))
            # print ("multi_state_ shape: ", multi_state_.shape)
            # tmp_dist = reward_func(self.getMultiModalImitationState())
            # print ("tmp_dist for pure imitation data: ", tmp_dist)
            # pose_diff = self.getImitationState() - self._sim.getState()
            # print ("pose_diff: ", pose_diff)
            dist = reward_func(multi_state_)
        elif ("process_visual_data" in self._config 
            and (self._config["process_visual_data"] == True)):
            state_ = np.array(self.getVisualState())
            dist = reward_func(np.reshape(self.getVisualState() ,newshape=(1, state_.size)),
                                np.reshape(self.getImitationVisualState(), newshape=(1, state_.size)))
        else:
            state_ = np.array(self.getImitationState())
            dist = reward_func(np.reshape(self.getCharacterState() ,newshape=(1, state_.size)),
                                np.reshape(self.getImitationState(), newshape=(1, state_.size)))
        # print("reward dist: ", len(dist), dist)
        return -dist[0]
    
    def getFullViewData(self):
        return self._sim.getFullViewData()
    
    
def getEnvsList():
    import os, sys, json
    from rlsimenv.config import SIMULATION_ENVIRONMENTS
    
    env_data = json.loads(SIMULATION_ENVIRONMENTS)
    
    return env_data

def getEnv(env_name, render=False):
    import os, sys, json
    
    env_data = getEnvsList()
    # print("Envs: ", json.dumps(env_data, indent=2))
    RLSIMENV_PATH = os.environ['RLSIMENV_PATH']
    print ("RLSIMENV_PATH: ", RLSIMENV_PATH)
    sys.path.append(RLSIMENV_PATH+'/lib')

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
    elif ( env_data[env_name]['sim_name'] == 'CannonGame'):
        from rlsimenv.CannonGame import CannonGame
        sim = CannonGame(settings=env_data[env_name])
    elif ( env_data[env_name]['sim_name'] == 'CannonImitationGame'):
        from rlsimenv.CannonImitationGame import CannonImitationGame
        sim = CannonImitationGame(settings=env_data[env_name])
    elif ( env_data[env_name]['sim_name'] == 'ProjectileGame'):
        from rlsimenv.ProjectileGame import ProjectileGame
        sim = ProjectileGame(settings=env_data[env_name])
    elif ( env_data[env_name]['sim_name'] == 'ObjectCentricSawyer'):
        from rlsimenv.stackingv2.simpleworlds.envs.mujoco.sawyer_xyz.objectcentric_sawyer import ObjectCentricSawyer
        sim = ObjectCentricSawyer()
        sim.render_on = render
    else:
        from pydoc import locate
        from rlsimenv.Environment import Environment
        sys.path.append("./stackingv2/")
        sys.path.append("./rlsimenv/stackingv2/")
        modelClass = locate(env_data[env_name]['sim_name'])
        print ("modelClass: ", modelClass)
        if ( issubclass(modelClass, Environment)): ## Double check this load will work
            sim = modelClass(settings=env_data[env_name])
            print("Created sim: ", sim)
        else:
            # sys.exit(2)
            print ("Env ", env_data[env_name]['sim_name'], " does not match a simulation environment type")
            sys.exit()
            return None
    
    ## place holder  
    # sim.setRender(render)
    sim.init()
    
    sim_ = EnvWrapper(sim, render=render, config=env_data[env_name])
    
    return sim_
