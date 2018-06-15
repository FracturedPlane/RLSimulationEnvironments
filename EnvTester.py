

from rlsimenv.EnvWrapper import getEnv
import numpy as np

if __name__ == '__main__':
    
    
    # env = getEnv(env_name="ParticleGame_2D-v0", render=False)
    env = getEnv(env_name="CannonGame-v0", render=True)

    actionSpace = env.getActionSpace()
    env.setRandomSeed(1234)
    
    print("observation_space: ", env.observation_space.getMaximum())
    print("Actions space max: ", len(env.action_space.getMaximum()))
    print("Actions space min: ", env.action_space.getMinimum())
    print("Actions space max: ", env.action_space.getMaximum())
    
    env.reset()
    for epoch in range(10):
        env.reset()
        print ("New episode")
        while (True):
            actions = []
            for i in range(env.getNumberofAgents()):
                action = ((actionSpace.getMaximum() - actionSpace.getMinimum()) * np.random.uniform(size=actionSpace.getMinimum().shape[0])  ) + actionSpace.getMinimum()
                actions.append(action)
            if (env.getNumberofAgents() > 1):
                observation, reward,  done, info = env.step(actions)
            else:
                observation, reward,  done, info = env.step(actions[0])
            print ("Reward: ", reward, "Action: ", actions, " observation: ", observation)
            print ("Done: ", done)
            if ( done ):
                break
            
            
    env.finish()
    
    
