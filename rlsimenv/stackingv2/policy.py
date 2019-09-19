from tree_search.stackingv2.simpleworlds.envs.mujoco.sawyer_xyz.sawyer import SawyerMultitaskXYZEnv
import numpy as np

def pickup0(obs):
    obs= obs[0]
    obj0 = obs[3*0:3*0+3][:2]
    obj1 = obs[3:3+3][:2]
    boxloc = np.array( [0. , 0.27921843])
    action =  np.concatenate([obj0, boxloc])
    obs= env.step(action)
    print("reward", obs[1])
    return obs

def pickup1(obs):
    obs= obs[0]
    obj0 = obs[3*0:3*0+3][:2]
    obj1 = obs[3:3+3][:2]
    boxloc = np.array( [0. , 0.27921843])
    action =  np.concatenate([obj1, boxloc])
    obs= env.step(action)
    print("reward", obs[1])
    return obs

def pickup01(obs):
    obs= obs[0]
    obj0 = obs[3*0:3*0+3][:2]
    obj1 = obs[3:3+3][:2]
    boxloc = np.array( [0. , 0.27921843])
    action =  np.concatenate([obj0, obj1])
    obs= env.step(action)
    print("reward", obs[1])
    return obs

def pickup10(obs):
    obs= obs[0]
    obj0 = obs[3*0:3*0+3][:2]
    obj1 = obs[3:3+3][:2]
    boxloc = np.array( [0. , 0.27921843])
    action =  np.concatenate([obj1, obj0])
    obs= env.step(action)
    print("reward", obs[1])
    return obs

env = SawyerMultitaskXYZEnv(render_on=True)
obs = env.reset()
obs = [obs]
obs = pickup0(obs)
obs = pickup10(obs)
env._earthquake()
# obs = env.step(np.zeros(4))[0]
# obj0 = obs[3*0:3*0+3][:2]
# obj1 = obs[3:3+3][:2]
# boxloc = np.array( [0. , 0.27921843])
# action =  np.concatenate([obj1, boxloc])
# obs2 = env.step(action)
# env.render()
# obs = obs2[0]
# obj0 = obs[3*0:3*0+3][:2]
# obj1 = obs[3:3+3][:2]
# #boxloc = np.array( [0.18796397 , 0.27921843])
# action =  np.concatenate([obj0, obj1])
#obs = env.step(action)
env.render()
#import pdb; pdb.set_trace()
#for i in range(10):
#    env._earthquake()
    #env.reset([[1]])
import pdb; pdb.set_trace()