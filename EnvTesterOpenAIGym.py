
import gym
import numpy as np
import rlsimenv

if __name__ == '__main__':
    
#     env = gym.make("ContinuousMaxwellsDemonFullyObserved-v0")
    env = gym.make("TagEnvFullyObserved-1particle-flatobs-dualstate-16x16-render-v0")


    env.seed(1234)
    
    print("observation space min: ", env.observation_space.low)
    print("observation space max: ", env.observation_space.high)
    print("Actions space max: ", len(env.action_space.high))
    print("Actions space min: ", env.action_space.low)
    print("Actions space max: ", env.action_space.high)
    
    env.reset()
    for epoch in range(10):
        env.reset()
        print ("New episode")
        # while (True):
        for i in range(100):
            actions = env.action_space.sample()
            observation, reward,  done, info = env.step(actions)
            print ("Reward: ", reward, "Action: ", actions)
            print (info)
            print ("observation size: ", np.array(observation).shape)
            print ("Done: ", done)
            
            # viewData = env.render("rgb_array")
            viewData = observation
            ## Get and vis terrain data
            if (True):
                ## Don't use Xwindows backend for this
                import matplotlib
                # matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                img_ = viewData[1]
                img_ = np.reshape(img_, (16,16,3))
                print("img_ shape", img_.shape, " sum: ", np.sum(img_))
                fig1 = plt.figure(1)
                plt.imshow(img_, origin='lower')
                plt.title("agent visual Data: ")
                fig1.savefig("viz_state_"+str(i)+".png")
                    
            
            if ( done ):
                break
            
            
    env.finish()
    
    
