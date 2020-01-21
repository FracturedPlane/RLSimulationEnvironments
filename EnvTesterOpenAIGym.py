from rlsimenv.EnvWrapper import getEnv
import numpy as np

if __name__ == '__main__':
    
    
    # env = getEnv(env_name="ParticleGame_2D-v0", render=False)
    # env = getEnv(env_name="CannonGameViz2-v0", render=True)
    env = getEnv(env_name="MaxwellsDemon_v0", render=True)

    actionSpace = env.getActionSpace()
    env.setRandomSeed(1234)
    
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
            """
            vizData = env.getVisualState()
            # print("visual Data: " +  str(vizData))
            vizImitateData = env.getImitationVisualState()
            for vd in range(len(vizData)):
                # print("viewData: ", viewData)
                viewData = vizData[vd]
                viewImitateData = vizImitateData[vd]
                ## Get and vis terrain data
                if (False):
                    ## Don't use Xwindows backend for this
                    import matplotlib
                    # matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    # img_ = np.reshape(viewData, (150,158,3))
                    img_ = viewData
                    print("img_ shape", img_.shape, " sum: ", np.sum(viewData))
                    fig1 = plt.figure(1)
                    plt.imshow(img_, origin='lower')
                    plt.title("agent visual Data: " +  str(vd))
                    fig1.savefig("viz_state_"+str(i)+".svg")

                    if (True):                    
                        img_ = viewImitateData
                        fig2 = plt.figure(2)
                        plt.imshow(img_, origin='lower')
                        plt.title("imitation visual Data: " +  str(vd))
                        fig2.savefig("viz_imitation_state_"+str(i)+".svg")
                        
                    plt.show()
                    
            img = env.getEnv().getVisualState()
            """
            if ( done ):
                break
            
            
    env.finish()
    
    