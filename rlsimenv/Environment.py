

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

class Environment(object):
    
    def __init__(self, settings):
        self._game_settings = settings

    def getActionSpace(self):
        return self._action_space
    
    def getObservationSpace(self):
        return self._observation_space