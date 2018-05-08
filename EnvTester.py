

from rlsimenv.EnvWrapper import getEnv
import numpy as np

if __name__ == '__main__':
    
    
    # env = getEnv(env_name="ParticleGame_2D-v0", render=False)
    env = getEnv(env_name="GapGame_2D-v0", render=False)

    actionSpace = env.getActionSpace()
    env.setRandomSeed(1234)
    
    actions = []
    for i in range(1):
        action = ((actionSpace.getMaximum() - actionSpace.getMinimum()) * np.random.uniform(size=actionSpace.getMinimum().shape[0])  ) + actionSpace.getMinimum()
        actions.append(action)            
    print("Actions: ", actions)
    
    print("observation_space: ", env.observation_space.getMaximum())
    print("Actions space max: ", len(env.action_space.getMaximum()))
    print("Actions space min: ", env.action_space.getMinimum())
    print("Actions space max: ", env.action_space.getMaximum())
    
    env.reset()
    
    for epoch in range(10):
        env.reset()
        for state in range(100):
            actions = []
            for i in range(1):
                action = ((actionSpace.getMaximum() - actionSpace.getMinimum()) * np.random.uniform(size=actionSpace.getMinimum().shape[0])  ) + actionSpace.getMinimum()
                actions.append(action)
            observation, reward,  done, info = env.step(actions)
            print ("Reward: ", reward, "Action: ", actions, " observation: ", observation)
            print ("Done: ", done)
            if ( done ):
                break
            
            
    env.finish()
    
    
