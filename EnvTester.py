
import matplotlib.pyplot as plt
import numpy as np
import pdb

import rlsimenv.EnvWrapper

def main():
    # env = getEnv(env_name="ParticleGame_2D-v0", render=False)
    # env = getEnv(env_name="CannonGameViz2-v0", render=True)
    env = rlsimenv.EnvWrapper.getEnv(env_name="MaxwellsDemon_v0", render=True)
    env.setRandomSeed(1234)
    
    print("observation space min: ", env.observation_space.low)
    print("observation space max: ", env.observation_space.high)
    print("Actions space max: ", len(env.action_space.high))
    print("Actions space min: ", env.action_space.low)
    print("Actions space max: ", env.action_space.high)

    # plt.axis([0, 128, 128, 0])
    plt.ion()
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    fig.show()
    
    env.reset()
    repeat = 1
    for epoch in range(10):
        env.reset()
        print("New episode")
        # while (True):
        ims = None
        r = repeat
        for i in range(env._sim._max_steps):
            if r == repeat:
                r = 0
                actions = []
                # actions.append(np.random.choice([0, 2.5]))
                for a in range(env.getNumberofAgents()):
                    action = np.random.normal(size=(3,), scale=1.)
                    action[-1] = (np.random.random() - 0.5) * 10
                    # action[-1] = np.random.choice([-9, -1, 0., 1., 9])
                    actions.append(action)
            else:
                r += 1
            if (env.getNumberofAgents() > 1):
                observation, reward,  done, info = env.step(actions)
            else:
                observation, reward,  done, info = env.step(actions[0])
                # observation, reward,  done, info = env.step([0,0])
            # print ("Reward: ", reward, "Action: ", actions, " observation: ", observation)
            print("observation size: ", np.array(observation).shape)
            print("Done: ", done)
            # """

            rendering = env.getVisualState()
            observation = env.getObservation()

            if ims is not None:
                for im in ims: im.remove()
            im0 = axes[0].imshow(rendering, origin='lower')
            im1 = axes[1].imshow(observation, origin='lower')
            ims = [im0, im1]

            plt.draw()
            plt.pause(0.0001)
            if done: break
            
    env.finish()
    
if __name__ == '__main__':
    main()
