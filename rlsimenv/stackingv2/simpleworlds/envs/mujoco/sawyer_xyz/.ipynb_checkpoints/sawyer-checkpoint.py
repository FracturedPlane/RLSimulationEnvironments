import numpy as np
import os.path as osp
from itertools import permutations, combinations, chain, product

from collections import OrderedDict
from gym.spaces import Box, Dict
import time
# Mujoco angle calculation
from pyquaternion import Quaternion

# Import library
import simpleworlds
from simpleworlds.utils.points import generate_points
from simpleworlds.utils.task_enumerators import enumerate_tasks, enumerate_tasks_corners
from simpleworlds.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from simpleworlds.envs.mujoco.util.create_xml import create_object_xml
from simpleworlds.core.multitask_env import MultitaskEnv
from simpleworlds.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

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

## Environment
class SawyerMultitaskXYZEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            episode_length=10,
            render_on           = False,
            reward_type         = 'state_distance',
            norm_order          = 1,
            indicator_threshold = 0.06,

            hide_goal_markers   = False,

            # objects related keywords
            init_object_z       = 0.02,
            object_meshes       = None,
            object_mass         = 0.5,
            block_height        = 0.02, 
            block_width         = 0.02,
            object_num          = 2,
            object_maxlen       = 0.020,
            object_minlen       = 0.015,

            workspace_low       = (-.4, .2, -0.02, -0.5*np.pi),
            workspace_high      = ( .4, 1.,  0.40,  0.5*np.pi),
            object_radius       = .02,

            # hand related
            #init_hand_xyz       = (0, 0.4, 0.15),
            init_hand_xyz       = (0, 0.4, 0.3), # CHANGE THIS
            hand_low            = (-0.38, 0.2, 0.05),
            hand_high           = ( 0.38, 1.0, 0.4),
            
            # gripper related
            gripper_low         = (0.0),
            gripper_high        = (0.04),

            reset_free          = False,
            clamp_object_on_step= False,

            environment_seed    = 0,
            environment_xml     = 'sawyer_box.xml',        # Modify this instead
            objects_xml         = 'objects_with_box.xml', # Doesn't actually do anything...
            **kwargs
    ):
        self.render_on = render_on
        self.dont_put_in_box = -1   
        self.episode_length = episode_length
        self.quick_init(locals())
        self.seed(environment_seed)

        self.environment_filepath = get_asset_full_path(osp.join('sawyer_xyz', environment_xml))
        self.object_filepath      = get_asset_full_path(osp.join('sawyer_xyz', objects_xml))


        friction_params = (0.1, 0.1, 0.02)
        '''
        self.object_stats = create_object_xml(self.object_filepath, 
                                               object_num, 
                                               object_mass,
                                               friction_params,
                                               object_meshes,
                                               object_maxlen, 
                                               object_minlen, 
                                               None,
                                               block_height,
                                               block_width, 
                                               object_radius)
        '''

        MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(
            self,
            hand_low    = hand_low,
            hand_high   = hand_high,
            model_name  = self.environment_filepath,
            **kwargs)
        assert workspace_low  is not None
        assert workspace_high is not None

        # Constrain to only "graspable" locations (problem might be with sampling)
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        self.hand_low = np.array([-.2, .4, .05])
        self.hand_high = np.array([.2, .8, .3])
        #self.hand_low = np.array([-.17, .43, .05])
        #self.hand_high = np.array([.17, .77, .3])
        
        self.joint_num        = 8 #+2

        self.object_names     = ['object' + str(i) for i in range(object_num)]
        self.object_dim       = 4                           # (x, y, z, rotation)
        self.object_num       = object_num

        self.workspace_low    = np.array(workspace_low)
        self.workspace_high   = np.array(workspace_high)

        self.gripper_low      = np.array(gripper_low)
        self.gripper_high     = np.array(gripper_high)


        self.reward_type = reward_type
        self.norm_order  = norm_order
        self.indicator_threshold = indicator_threshold

        self._state_goal = None
        self.init_hand_xyz = init_hand_xyz

        self.hide_goal_markers = hide_goal_markers

        self.action_space = Box(np.array([-1, -1, -1, -1]),
                                np.array([1, 1, 1, 1]), dtype=np.float32)
        self.hand_space   = Box(self.hand_low,
                                self.hand_high, dtype=np.float32)

        # The observation space is the concatenation of 
        #       (bject_state)
#         self.obs_space = Box(
#             np.hstack(( self.workspace_low.tolist()  * self.object_num)),
#             np.hstack((self.workspace_high.tolist() * self.object_num)),
#             dtype=np.float32
#         )

#         self.observation_space = Dict([
#             ('observation',             self.obs_space),
#             ('desired_goal',            self.obs_space),
#             ('achieved_goal',           self.obs_space),
#             ('state_observation',       self.obs_space),
#             ('state_desired_goal',      self.obs_space),
#             ('state_achieved_goal',     self.obs_space),
#             ('proprio_observation',     self.hand_space),
#             ('proprio_desired_goal',    self.hand_space),
#             ('proprio_achieved_goal',   self.hand_space),
#         ])
#         self.observation_space = Box(
#             np.hstack((self.workspace_low.tolist()  * self.object_num)),
#             np.hstack((self.workspace_high.tolist() * self.object_num)),
#             dtype=np.float32
#         )
#         # Obs space for 4 dim (2 obj, 2 goal) obs space
#         self.observation_space = Box(
#             np.hstack(self.workspace_low.tolist() * 2),
#             np.hstack(self.workspace_high.tolist() * 2),
#             dtype=np.float32
#         )

        # Obs space for normalized 6 dim obs space (see ^^^)
        self.observation_space = Box(np.ones(3*self.object_num)*-1, np.ones(3*self.object_num), dtype=np.float32)

        self.init_object_z  = init_object_z
        self.init_hand_xyz  = np.array(init_hand_xyz)

        # Initialize object locationss
        self._set_object_positions(self._sample_object_positions(self.object_num))

        self.reset_free     = reset_free
        self.reset_counter  = 0
        self.object_space   = Box(self.workspace_low, 
                                  self.workspace_high, dtype=np.float32)
        self.clamp_object_on_step = clamp_object_on_step
        self.object_radius        = object_radius

        self.grasp_height = .07
        self.release_height = .15 +0.15
        self.raised_height = .3
        self.pos = np.array([0.0,0.4,self.raised_height])
        self.closed_grasper = -1

        self.task = None
        self.max_length = 3

        # TODO: Add option for corners
        self.task_list = enumerate_tasks()
        #self.task_list = enumerate_tasks_corners()

        self.reset()

    def get_config(self):
        return {'env_name': 'grasper'}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0]   = 0
        self.viewer.cam.lookat[1]   = 1.0
        #self.viewer.cam.lookat[1]   = .6    # CHANGE
        self.viewer.cam.lookat[2]   = 0.5
        self.viewer.cam.distance    = 1.0
        self.viewer.cam.elevation   = -45
        #self.viewer.cam.elevation   = -90   # CHANGE
        self.viewer.cam.azimuth     = 270
        self.viewer.cam.trackbodyid = -1

    def move_to_position(self, goal, step_size = .005):
        # We take steps of size `step_size` to the goal

        distance = np.linalg.norm(self.pos - goal)
        steps = int(distance / step_size)

        transitions = np.array([np.linspace(s,e,steps+1) for s,e in zip(self.pos, goal)]).T

        for trans_state in transitions:
            self.pos = trans_state
            self.set_xyz_position(self.pos)
            ctrl = np.zeros(len(self.sim.data.ctrl))
            ctrl[:8] =self.closed_grasper
            self.do_simulation(ctrl)
            self.sim.step()
            self.render()
        self.idle_actions(10)

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
        self.move_to_position(down_goal)
        ctrl = np.zeros(len(self.sim.data.ctrl))
        ctrl[:8] = 1
        self.closed_grasper = 1
        self.do_simulation(ctrl)
        self.move_to_position(up_goal)

    def execute_release(self):
        # Execute a release at the release height
        # Necessary b/c objs w/ too much speed breaks mujoco :(

        down_goal = np.copy(self.pos)
        up_goal = np.copy(self.pos)
        release_height = self.release_height

        # This prevents grasper from grasping two
        # objects at once if they are stacked
        # Calc close objs and take max of height
        # and modify grasp height
        
        objs_pos = self.get_object_positions().reshape((-1, 4))[:self.object_num, :3]
        close_objs = np.linalg.norm(objs_pos[:, :2] - self.pos[:2], axis=1) < .03
        # If there is an object close by
        if np.sum(close_objs) > 0:
            # Get heights of objects close to grasper
            heights = objs_pos[close_objs, 2]
            # Get rid of object in gripper's claws
            heights = [x for x in heights if x < .2]
            
            release_height = .12
            if len(heights) != 0:
                release_height += np.max(heights)
        release_height=0.3
        down_goal[2] = release_height
        #print("release height", release_height)
        #import pdb; pdb.set_trace()
        #down_goal += np.random.random(down_goal.shape)/10
        self.move_to_position(down_goal)
        self.closed_grasper = -1
        ctrl = np.random.random(len(self.sim.data.ctrl))*0#np.zeros(len(self.sim.data.ctrl))
        ctrl[7] = -1
        #ctrl[6] = 0
#         newstate = self.get_env_state()
#         newstate[0][0].qvel[6] = (np.random.random()-0.5)*10000
#         print("wvel", newstate[0][0].qvel[6] )
#         self.set_env_state(newstate)
#         for t in range(200):
#             self.sim.step()
#             #print("qpos", self.data.qpos[6])
#             self.render()
# #         print("post", newstate[0][0].qvel[6] )
        #print("ctrl",ctrl)
        self.do_simulation(ctrl)

#         newstate = self.get_env_state()
#         newstate[0][0].qvel[6] = 0
#         print("wvel", newstate[0][0].qvel[6] )
#         self.set_env_state(newstate)
        
        self.move_to_position(up_goal)

    def translate_action(self, action):
        '''
        Assume action space is box, [-1,1], two dims
        '''
        x, y = action
        x = (x + 1) / 2.
        y = (y + 1) / 2.
        xp = (1-x) * -.3 + x * .3
        yp = (1-y) * (0.6-0.3) + y * (0.6+0.3)
        return np.array([xp, yp])

    def idle_actions(self, num):
        ctrl = np.zeros(len(self.sim.data.ctrl))
        ctrl[:8] =self.closed_grasper
        for i in range(num):
            self.do_simulation(ctrl)
            self.render()
        return self._get_obs()

    def step(self, action):
        # tmp debug perfect policy
        '''
        obs = self._get_obs()
        lid_loc = obs[-3:-1]
        box_loc = obs[-6:-4]
        obj_idx = 0
        obj_loc = obs[3*obj_idx:3*obj_idx+2]
        obj_idx2 = 1
        obj_loc2 = obs[3*obj_idx2:3*obj_idx2+2]
        '''

        # box and lid policy
        '''
        if action[0] == 1:
            action = np.hstack((lid_loc, -lid_loc))
        elif action[0] == 2:
            action = np.hstack((obj_loc, box_loc))
        elif action[0] == 3:
            action = np.hstack((lid_loc, box_loc))
        '''

        # stacking policy
        '''
        if action[0] == 1:
            action = np.hstack((obj_loc, obj_loc2))
        '''

        # placing policy
        '''
        if action[0] == 1:
            action = np.hstack((obj_loc, -.6, .6))
        '''
        #print("start")
        action = action.astype(float)
        control = False
        # Add 0s for bin motors
        if control:
            self.set_xyz_action(action[:3])
            self.do_simulation(action[3:])
        else:
            # Do nothing action
            if np.linalg.norm(action) == 0:
                pass
            else:
                action[:2] = self.translate_action(action[:2])
                action[2:] = self.translate_action(action[2:])

                #obs     = self._get_obs()
                #print("before grasp")
                # Perform grasp if action and gripper is opened
                #if self.closed_grasper == -1:
                # Open gripper before grasping (might be redundant...)
                self.closed_grasper = -1
                self.move_to_position(np.append(action[:2], self.raised_height))
                self.execute_grasp()
                #print("middle grasp")
                # Perform placement if gripper is closed
                #elif self.closed_grasper == 1:
                # Close gripper before moving (might be redundant...)
                self.closed_grasper = 1
                self.move_to_position(np.append(action[2:], self.raised_height))
                #print("2middle grasp")
                # Set gripper to opened and take a step
                #self.closed_grasper = -1
                self.execute_release()
                #print("after grasp")
                # Let state settle
                self.idle_actions(10)

        self._set_goal_marker(self._state_goal)
        
#         if 0.6 > np.random.rand():
#             #print("EARTHQUAKE")
#             self._earthquake()
        obs     = self._get_obs()
        full_obs = self._get_fullstate()
        reward  = self.compute_reward(action, obs)
        #reward = int(self._get_done())
        #info    = self._get_info()
        obj0 = obs[3*0:3*0+3][:2]
        obj1 = obs[3:3+3][:2]

        boxloc = np.array( [0. , 0.27921843])
        info    = {'obj0_box': np.linalg.norm(obj0-boxloc),
                   'obj1_box': np.linalg.norm(obj1-boxloc),
                  }
        self.num_steps += 1
        done = self._get_done() or self._check_objects(obs)
        #obs     = self._get_obs()
        #reward += int(done) * 5
        #print(reward)
        
        return obs, reward, done, info

    def _check_objects(self, obs):
        reset = False
        for obji in range(self.object_num):
            pos = obs[3*obji:3*obji+3]
            if np.any(pos[:2]> 2.) or  np.any(pos[:2]< -2.) or pos[2] < -2:
                #import pdb; pdb.set_trace()
                reset= True
#                 new_pos = self._sample_object_positions(1)
#                 self._set_object_position(obji, new_pos[0*(self.object_dim):(0+1)*(self.object_dim)])

#                 self.sim.step()
#                 print("obj", obji, "pos", pos, "newpos", new_pos)
        return reset
    def set_task(self, task):
        # If putting in box, don't have object start in box

        self.task = task

    def _get_done(self):
        '''

        '''
        
        return self.num_steps > self.episode_length


    def trans_obs(self, obs):
        # Translate x and y differently
        # z is not translated
        obs_x = obs[::4]
        obs_y = obs[1::4]
        obs_z = obs[2::4]
        obs_t = obs[3::4]

        obs_x = obs_x / .3
        obs_y = (obs_y - .6) / .3
        obs_z = (obs_z - .0275) / (.0629-.0275)

        translated = np.zeros(int(len(obs)*3/4))
        translated[::3] = obs_x
        translated[1::3] = obs_y
        translated[2::3] = obs_z

        return translated

    def _get_obs(self):
        obj_pos = self.get_object_positions()
        obj_pos = self.trans_obs(obj_pos)
        return obj_pos 
    
    def _get_fullstate(self):
        obj_pos = self._get_obs()
        #obj_vels = self.get_object_velocities()
        return obj_pos

    def _get_info(self):
        return {'dist2goal': np.linalg.norm(self.get_object_positions()[:2] - self._state_goal[2:])}

    def get_object_positions(self):
        object_poses = [ np.hstack([self.data.get_body_xpos(_).copy(), \
                                   [ quat_to_zangle(self.sim.data.get_body_xquat(_).copy())] 
                            ]) for _ in self.object_names ]
        return np.array(object_poses).flatten()
    
    def get_object_velocities(self):
        object_poses = [self.data.get_body_xpos(_).copy() for _ in self.object_names ]
        return np.array(object_poses).flatten()


    def reset_model(self):
        self._reset_hand()
        self._reset_bin()
        self._set_object_positions(self._sample_object_positions(self.object_num))
        self._set_object_velocities()
        obs = self.get_object_positions()
        obs = np.append(obs[:2], [0,0])
        #goal = self.sample_goals(batch_size=1)
        goal = self.generate_goal(obs)
        self.set_goal({'state_desired_goal': goal})

        self.reset_counter += 1
        self.reset_mocap_welds()
        return self._get_obs()

        #return obs

    def reset(self):
        # Set task: stack 0 on 1
        task = [[1, [0,1]]]
        self.set_task(task)
        self.num_steps=0
        # Reset model
        ob = self.reset_model()

        # Setup viewer
        if self.viewer is not None:
            self.viewer_setup()

        # Let objects settle
        ob = self.idle_actions(10)

        # Render if `render_on=True`
        self.render()

        return ob

    @property
    def init_angles(self):
        return [1.2130368, -1.06266148, -0.0589842 ,  1.70264581,  3.17711987, -0.9330582 ,  0.3121358]
        return [1.78130368, -1.06266148, -0.0589842 ,  1.70264581,  3.17711987, -0.9330582 ,  0.3121358]
        return [1.7244448,  -0.92036369,         0.10234232,
                2.11178144,  2.97668632,        -0.38664629,
                0.54065733,  5.05442647e-04, 6.00496057e-01,
                3.06443862e-02, 1, 0, 0, 0]

    """
    Multitask functions
    """
    def get_goal(self):
        return {
            'desired_goal':       self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']
        self._set_goal_marker(self._state_goal)

    def set_to_goal(self, goal):
        hand_goal = goal['state_desired_goal'][:3]
        for _ in range(10):
            self.data.set_mocap_pos('mocap', hand_goal)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)
        object_goal = goal['state_desired_goal'][3:5]
        self._set_object_xy(object_goal)
        self.sim.forward()

    def generate_goal(self, goal):
        # One hot vector to index in to objects
        goal = np.zeros(8)
        goal[np.random.randint(8)] = 1
        return goal

    def sample_goal(self):
        # Goal position + hand position
        object_goal = self._sample_object_positions(self.object_num)

        # Re-config Z position
        goal = np.hstack([np.array([self.gripper_high]), 
                          np.array(self.init_hand_xyz), 
                          object_goal])

        return np.array(goal).flatten()

    def sample_goals(self, batch_size):
        goals = []
        for i in range(batch_size):
            goals.append(self.sample_goal())

        goals = np.array(goals)

        return  {'desired_goal': goals, 'state_desired_goal': goals}

    def compute_rewards(self, action, obs):
        subtask = self.task[0]
        task_idx = subtask[0]
        params = subtask[1]

        # Get the location of objects
        obj_pos = self._get_obs()

        # TODO: Set this as param
        dist_thresh = .1

        # Variable to track if we have accomplished this subtask
        subtask_success = False

        # Stack objects
        if task_idx == 1:
            # check if objects are "close"
            # no check for height right now

            # get relevant state
            objIdx1 = params[0]
            objIdx2 = params[1]
            objPos1 = obs[:, 3*objIdx1:3*objIdx1+2]
            objPos2 = obs[:, 3*objIdx2:3*objIdx2+2]

            # check closeness
            r = -1*np.linalg.norm(objPos1 - objPos2, axis=1)
        return r

#    def get_diagnostics(self, paths, prefix=''):
        
#         lastd2gs = []
#         for path in paths:
#             lastd2gs.append(path['env_infos'][-1]['dist2goal'])
#         return {'meanFinalDist': np.mean(lastd2gs), 'fracSuccess': np.sum(np.array(lastd2gs) < .05)/len(lastd2gs)}
        #return {'meanFinalDist': np.mean(lastd2gs), 'fracSuccess': np.sum(np.array(lastd2gs) < .05)/len(lastd2gs), 'minPathLength': np.min([len(x['terminals']) for x in paths])}

    def get_env_state(self):
        base_state = super().get_env_state()
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        super().set_env_state(base_state)
        self._state_goal = goal
        self._set_goal_marker(goal)


    ## Internal functions
    #     - All internal functions starts with underline
    def _sample_object_positions(self, object_num, margin=0.1):
        """ Get object locations based on Poisson Disc algorithm"""

        init_space_size = (self.hand_high - self.hand_low)
        random_locations = generate_points(margin,
                       init_space_size[0], init_space_size[1], 
                       object_num)
        #print(init_space_size)
        box_loc = np.array( [0.18796397 , 0.27921843])
        #print(box_loc)
        # Move box towards middle
        '''
        origin = np.array([.2, .2])
        box_loc = tuple((box_loc - origin) * .5 + origin)
        box_loc = np.array(box_loc)
        random_locations[-2] = box_loc
        '''

        # Remove objects from box
        for i in range(len(random_locations)):
            location = list(random_locations[i])
            while np.max(np.absolute(random_locations[i] - box_loc)) < .08:     # Hardcoded
                x = np.random.uniform(0,.4)
                y = np.random.uniform(0,.4)
                random_locations[i] = (x,y)

        object_poses = []
        for i, _location in enumerate(random_locations):
            # uniformly sample rotation [disabled for now]
            # NO ROTATION
            rotation = np.random.uniform(-np.pi / 2, np.pi / 2)
            rotation = 0

            # append (x,y) locations
            object_poses.append(( _location[0] + self.hand_low[0],
                                  _location[1] + self.hand_low[1],
                                  self.init_object_z,
                                  rotation             
                                ))

        # No rotation on last two objects (box and lid)
        # Put lid on top of box

        # Set box coordinates, no lid is present
        pose = list(object_poses[-1])
        pose[-1] = 0                        # No rotation
        object_poses[-1] = tuple(pose)

#         # Set lid coordinates
#         object_poses[-1] = object_poses[-2]
#         pose = list(object_poses[-1])
#         pose[-2] = .08
#         pose[-1] = 0
#         object_poses[-1] = tuple(pose)

        # Place random object in box
        object_poses = np.array(object_poses)

#         # Choose random obj to put in box, but not the `dont_put_in_box` obj
#         obj_in_box = np.random.randint(object_num - 2)
#         while obj_in_box == self.dont_put_in_box:
#             obj_in_box = np.random.randint(object_num - 2)
#         # Resset `dont_put_in_box` obj number
#         self.dont_put_in_box = -1
                
#         object_poses[obj_in_box][:2] = object_poses[-1][:2]
#         object_poses[obj_in_box][2] = .04

        return np.array(object_poses).flatten()

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        pass # not implemented right now

        # self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = (
        #     goal[:3] )
        # self.data.site_xpos[self.model.site_name2id('puck-goal-site')] = (
        #     goal[3:5]
        # )
        # if self.hide_goal_markers:
        #     self.data.site_xpos[self.model.site_name2id('hand-goal-site'), 2] = ( -1000 )
        #     self.data.site_xpos[self.model.site_name2id('puck-goal-site'), 2] = ( -1000 )

    def _set_object_positions(self, positions):
        assert positions.shape[0] == self.object_num * (self.object_dim)
#         # Set box first
#         i = self.object_num -1
#         self._set_object_position(i, positions[i*(self.object_dim):(i+1)*(self.object_dim)])
        for i in range(self.object_num ):
            self._set_object_position(i, positions[i*(self.object_dim):(i+1)*(self.object_dim)])


    def _set_object_position(self, i, position):
        #print('object{} position: {}'.format(i, position[:3]))
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        x = self.joint_num + i * 7
        y = self.joint_num + i * 7 + 3 # end index to xyz position
        z = self.joint_num + i * 7 + 7 # end index to rotation
        # setup x, y, z value
        qpos[x:y] = position[:3].copy()
        # setup rotation value quaternion
        qpos[y:z] = Quaternion(axis=[0, 0, -1], angle= position[3]).elements

        # not sure what following means
        x = 7  + i * 6                       
        y = 13 + i * 6
        qvel[x:y] = 0
        self.set_state(qpos, qvel)

    def _set_object_velocities(self):
        for i in range(self.object_num ):
            vel = (np.random.random(6)-0.5)
            vel = vel/np.linalg.norm(vel)*2.0
            #print("myvel", vel)
            self._set_object_velocity(i, vel)


    def _set_object_velocity(self, i, vel):
        #print('object{} position: {}'.format(i, position[:3]))
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        # not sure what following means
        x = 7  + i * 6                       
        y = 13 + i * 6
        #print("vel", vel, qvel[x:y])
        qvel[x:y] = vel
        self.set_state(qpos, qvel)
        
    def _reset_hand(self):
        velocities = self.data.qvel.copy()
        angles = self.data.qpos.copy()
        angles[:7] = self.init_angles[:7]
        self.set_state(angles.flatten(), velocities.flatten())
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.init_hand_xyz.copy())
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def _reset_bin(self):
        velocities = self.data.qvel.copy()
        angles = self.data.qpos.copy()
        angles[8:10] = np.zeros(2)
        velocities[8:10] = np.zeros(2)
        self.set_state(angles.flatten(), velocities.flatten())

    
    def _earthquake(self):
        #t1 = time.clock()
        def apply_force(force, joints):
            self.sim.data.ctrl[joints] = force
            #env.sim.data.ctrl[8:9] =0
            for i in range(100):
                if self.render_on:
                    self.render()
                self.sim.step()
            self.sim.data.ctrl[joints] = force*0
            for i in range(20):
                if self.render_on:
                    self.render()
                self.sim.step()
        joints = np.array([8,9,10])
        
        force1 = (np.random.rand(len(joints))-0.5)
        force1[2] = np.abs(force1[2])
        force1 = force1/np.linalg.norm(force1)*5000000#0000
        
        force2 = (np.random.rand(len(joints))-0.5)
        force2[2] = np.abs(force2[2])
        force2 = force2/np.linalg.norm(force2)*5000000#0000
        #print(np.linalg.norm(force1), np.linalg.norm(force2))
        #import pdb; pdb.set_trace()
        force = np.concatenate([force1,force2])
        joints = np.array([8,9,10,11,12,13])
        apply_force(force,joints)
        #t2 = time.clock()
        #print(t2-t1)
        #apply_force(-1*force, joints)
        
#         # Go back to middle if you are not
#         dt = 0.002
#         previous_error = np.zeros(2) - self.data.qpos[joints]
#         integral = 0
#         Kp =1000
#         Ki = 500
#         Kd = 100
#         while np.linalg.norm(self.data.qpos[8:11]) > 0.2:
#             error = np.zeros(2) - self.data.qpos[joints]
#             integral = integral + error * dt
#             derivative = (error - previous_error) / dt
#             output = Kp * error + Ki * integral + Kd * derivative
#             previous_error = error
#             self.sim.data.ctrl[joints] = output
#             self.sim.step()
#             self.render()
#             print(self.data.qpos[joints])
#         self._reset_bin()
        #self.sim.step()
        #self.render()
        
#     def _earthquake(self):
  
#         def apply_force(force, joints):
#             self.sim.data.qvel[joints] = force
# #             for i in range(5):
# #                 if self.render_on:
# #                     self.render()
# #                 self.sim.step()
# #             #env.sim.data.ctrl[8:9] =0
#             for i in range(30):
#                 self.sim.step()
#                 if self.render_on:
#                     self.render()
#             self.sim.data.qvel[joints] = force*0
#             self.sim.data.qvel[joints[::3]] = -1
#             for i in range(30):
#                 if self.render_on:
#                     self.render()
#                 self.sim.step()
#         joints = np.array([8,9,10])
        
#         force1 = (np.random.rand(len(joints))-0.5)
#         force1[2] = 0#np.abs(force1[2])
#         force1 = force1/np.linalg.norm(force1)*7
#         force2 = (np.random.rand(len(joints))-0.5)
#         force2 = force2/np.linalg.norm(force2)*7
#         force2[2] = 0#np.abs(force2[2])
#         force = np.concatenate([force1,force2])
#         joints = np.array([8,9,10,11+3,12+3,13+3])
#         print(self.sim.data.qvel.shape)
#         apply_force(force,joints)