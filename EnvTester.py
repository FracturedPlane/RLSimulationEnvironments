
import matplotlib.pyplot as plt
import numpy as np
import pdb

import rlsimenv.EnvWrapper

def main():
    # env = getEnv(env_name="ParticleGame_2D-v0", render=False)
    # env = getEnv(env_name="CannonGameViz2-v0", render=True)
    env = rlsimenv.EnvWrapper.getEnv(env_name="MaxwellsDemon_v0", render=True)

    actionSpace = env.getActionSpace()
    env.setRandomSeed(1234)
    
    print("observation space min: ", env.observation_space.low)
    print("observation space max: ", env.observation_space.high)
    print("Actions space max: ", len(env.action_space.high))
    print("Actions space min: ", env.action_space.low)
    print("Actions space max: ", env.action_space.high)

    # plt.axis([0, 128, 128, 0])
    plt.ion()
    plt.show()
    
    env.reset()
    for epoch in range(10):
        env.reset()
        print ("New episode")
        # while (True):
        for i in range(100):
            actions = []
            for a in range(env.getNumberofAgents()):
                action = ((actionSpace.high - actionSpace.low) * np.random.uniform(size=actionSpace.low.shape[0])  ) + actionSpace.low
                actions.append(action)
            if (env.getNumberofAgents() > 1):
                observation, reward,  done, info = env.step(actions)
            else:
                observation, reward,  done, info = env.step(actions[0])
                # observation, reward,  done, info = env.step([0,0])
            # print ("Reward: ", reward, "Action: ", actions, " observation: ", observation)
            print ("observation size: ", np.array(observation).shape)
            print ("Done: ", done)
            # """

            rendering = env.getVisualState()
            vizImitateData = env.getImitationVisualState()
            
            plt.imshow(rendering, origin='lower')
            plt.draw()
            plt.pause(0.0001)
                    
            img = env.getEnv().getVisualState()
            # """
            if ( done ):
                break
            
            
    env.finish()
    
    
if __name__ == '__main__':
    main()
    
