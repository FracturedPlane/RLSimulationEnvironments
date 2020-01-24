

def clampValue(value, bounds):
    """
    bounds[0] is lower bounds
    bounds[1] is upper bounds
    """
    for i in range(len(value)):
        if value[i] < bounds[0][i]:
            value[i] = bounds[0][i]
        elif value[i] > bounds[1][i]:
            value[i] = bounds[1][i]
    return value

import gym

class Environment(gym.Env):
    
    def __init__(self,):
        self._config = {}
        self._done = False
        super(Environment,self).__init__()
            
    def getActionSpace(self):
        return self.action_space
    
    def getObservationSpace(self):
        return self.observation_space
    
    def step(self, action):
        """
            Adding multi character support
        """
        # action = action[0]
        # print ("step action: ", action, " done: ", self._done)
        # action = np.array(action, dtype="float64")
        if ("openAIGym" in self._config
            and (self._config["openAIGym"] == True)):
            ob, reward, self._done, _ = self.step(action)
            return [ob], reward, self._done, None
            
        self.updateAction(action)
        
        # for i in range(15):
        if ( "control_return" in self._config and (self._config["control_return"] == True) ):
            i=0
            while ( (not self.needUpdatedAction()) and (i < 50 )):
                # print ("Controlling return")
                self.update()
                # self.render()
                i=i+1
        else:
            self.update()
            # self.render()
        # if ( self._render == True ):
        #    self._sim.display()
        # print("Num Agents: ", self._sim.getNumAgents())
        
        ob = self.getObservation()    
        reward = self.calcReward()
            
        self._done = self.agentHasFallen() or self._done
        # observation, reward, done, info
        # ob = np.array(ob)
        # print ("ob shape: ", ob.shape)
        return ob, reward, self._done, None