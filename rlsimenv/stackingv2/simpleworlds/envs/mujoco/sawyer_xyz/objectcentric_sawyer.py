import numpy as np
import os.path as osp
from itertools import permutations, combinations, chain, product

from collections import OrderedDict
from gym.spaces import Box, Dict
import time
# Mujoco angle calculation
from pyquaternion import Quaternion
import sys
sys.path.append("./rlsimenv/stackingv2")

# Import library
from rlsimenv.stackingv2 import simpleworlds
from rlsimenv.stackingv2.simpleworlds.utils.points import generate_points
from rlsimenv.stackingv2.simpleworlds.utils.task_enumerators import enumerate_tasks, enumerate_tasks_corners
from rlsimenv.stackingv2.simpleworlds.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from rlsimenv.stackingv2.simpleworlds.envs.mujoco.util.create_xml import create_object_xml
from rlsimenv.stackingv2.simpleworlds.core.multitask_env import MultitaskEnv
from rlsimenv.stackingv2.simpleworlds.envs.mujoco.sawyer_xyz.sawyer import SawyerMultitaskXYZEnv

import mujoco_py

import pdb

## Constants
BASE_DIR = osp.dirname(osp.abspath(simpleworlds.__file__))
ASSETS_DIR = osp.join(BASE_DIR, 'simpleworlds', 'envs', 'assets')

## Convenience Functions
def quat_to_zangle(quat):
    angle = -(Quaternion(axis = [0,1,0], angle = np.pi).inverse 
            * Quaternion(quat)).angle
    
    if angle < 0: return angle + 2 * np.pi
    return angle

def zangle_to_quat(zangle):
    """
    :param zangle in rad
    :return: quaternion
    """
    return (Quaternion(axis=[0, 1,  0], angle=np.pi) * 
            Quaternion(axis=[0, 0, -1], angle= zangle)).elements
import time
## Environment
class ObjectCentricSawyer(SawyerMultitaskXYZEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = Box(np.concatenate([np.zeros(self.object_num), np.array([ -1, -1])]),
                                np.concatenate([np.ones(self.object_num), np.array([ 1, 1])]), dtype=np.float32)
        

    def step(self, action):
        obs = self._get_obs()
        action = action.astype(float)
        elements = np.arange(self.object_num)
        probabilities = np.exp(action[:self.object_num])
        probabilities = probabilities/np.sum(probabilities)
        
        obj_idx = np.random.choice(elements, 1, p=probabilities)[0]
        #print("ids", obj_idx)
        obj_entropy = np.sum(np.log(probabilities)*probabilities)
        
        obji = obs[3*obj_idx:3*obj_idx+3][:2]
        #import pdb; pdb.set_trace()
        action = np.concatenate((obji, action[self.object_num:]))
        self.obj_to_grasp = obj_idx
#         action[:2] = self.translate_action(action[:2])
#         action[2:] = self.translate_action(action[2:])
#         #obs     = self._get_obs()
        
#         # Perform grasp if action and gripper is opened
#         #if self.closed_grasper == -1:
#         # Open gripper before grasping (might be redundant...)
#         self.closed_grasper = -1
#         self.move_to_position(np.append(action[:2], self.raised_height))
#         self.execute_grasp()
#         # Perform placement if gripper is closed
#         #elif self.closed_grasper == 1:
#         # Close gripper before moving (might be redundant...)
#         self.closed_grasper = 1
#         self.move_to_position(np.append(action[2:], self.raised_height))
#         # Set gripper to opened and take a step
#         #self.closed_grasper = -1
#         self.execute_release()
#         # Let state settle
#         self.idle_actions(10)

#         self._set_goal_marker(self._state_goal)
        
# #         if 0.8 > np.random.rand():
# #             #print("EARTHQUAKE")
# #             self._earthquake()
        
#         obs     = self._get_obs()
#         reward  = self.compute_reward(action, obs)
#         #reward = int(self._get_done())
#         #info    = self._get_info()
#         obj0 = obs[3*0:3*0+3][:2]
#         obj1 = obs[3:3+3][:2]
#         boxloc = np.array( [0. , 0.27921843])
#         info    = {'obj0_box': np.linalg.norm(obj0-boxloc),
#                    'obj1_box': np.linalg.norm(obj1-boxloc),
#                    'obj_entropy': obj_entropy,
#                   }
#         for obj in range(self.object_num):
#             info['obj'+str(obj)+'_prob'] = probabilities[obj]
#         self.num_steps += 1
#         done    = self._get_done()
#         #reward += int(done) * 5
#         #print(reward)
        obs, reward, done, info = super().step(action)
        info['obj_entropy'] =  obj_entropy
        return obs, reward, done, info

    def execute_grasp(self):
        # Execute a grasp at the current location

        # Set grasp height (check if over lid)
        #lid_pos = self.get_object_positions()[-4:]
        grasp_height = self.grasp_height

        objs_pos = self.get_object_positions().reshape((-1, 4))[:self.object_num, :3]

        # This prevents grasper from grasping two
        # objects at once if they are stacked
        # Calc close objs and take max of height
        # and modify grasp height
        close_objs = np.linalg.norm(objs_pos[:, :2] - self.pos[:2], axis=1) < .03

        # If there is an object close by
        if np.sum(close_objs) > 0:
            height_diff = np.max(objs_pos[close_objs, 2]) - .026
            grasp_height += height_diff

        # Take L-infinity norm
        # Hardcoded .05 as lid half-width
#         if np.max(np.absolute(self.pos[:2] - lid_pos[:2])) < .08:   # Hardcoded
#             grasp_height = lid_pos[2] + .07     # Hardcoded
#             #grasp_height = .2

        down_goal = np.copy(self.pos)
        up_goal = np.copy(self.pos)
        down_goal[2] = grasp_height
        self.move_to_object(self.obj_to_grasp,  grasp_height)
        ctrl = np.zeros(len(self.sim.data.ctrl))
        ctrl[:8] = 1
        self.closed_grasper = 1
        self.do_simulation(ctrl)
        self.move_to_position(up_goal)
        
    def move_to_object(self, obj_index, grasp_height, step_size = .005):
        # We take steps of size `step_size` to the goal
        goal = self.get_object_positions().reshape((-1, 4))[obj_index, :3]
        goal[2] = grasp_height
        distance = np.linalg.norm(self.pos - goal)
        steps = int(distance / step_size)
        steps = 0
        while distance > 0.005 and steps < 100:
            goal = self.get_object_positions().reshape((-1, 4))[obj_index, :3]
            goal[2] = grasp_height
            steps = int(distance / step_size)
            transitions = np.array([np.linspace(s,e,steps+1) for s,e in zip(self.pos, goal)]).T
            trans_state =  transitions[1]
            self.pos = trans_state
            self.set_xyz_position(self.pos)
            distance = np.linalg.norm(self.pos - goal)

            ctrl = np.zeros(len(self.sim.data.ctrl))
            ctrl[:8] =self.closed_grasper
            self.do_simulation(ctrl)
            self.sim.step()
            self.render()
        self.idle_actions(10)