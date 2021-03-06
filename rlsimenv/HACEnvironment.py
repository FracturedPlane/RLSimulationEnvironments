from tkinter import *
from tkinter import ttk
import time
import os
import random
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer

from rlsimenv.EnvWrapper import ActionSpace
from rlsimenv.Environment import Environment


class HACEnvironment(Environment):

    def __init__(self, settings):

        Environment.__init__(self, settings)

        model_name = settings["model_name"]
        goal_space_train = settings["goal_space_train"]
        goal_space_test = settings["goal_space_test"]

        def bound_angle(angle):

            bounded_angle = np.absolute(angle) % (2 * np.pi)
            if angle < 0:
                bounded_angle = -bounded_angle

            return bounded_angle

        project_state_to_end_goal = lambda sim, state: np.array(
            [bound_angle(sim.data.qpos[i]) for i in range(len(sim.data.qpos))])

        end_goal_thresholds = settings["end_goal_thresholds"]
        initial_state_space = settings["initial_state_space"]
        subgoal_bounds = settings["subgoal_bounds"]

        project_state_to_subgoal = lambda sim, state: np.concatenate((np.array(
            [bound_angle(sim.data.qpos[i]) for i in range(len(sim.data.qpos))]), np.array(
            [4 if sim.data.qvel[i] > 4 else -4 if sim.data.qvel[i] < -4 else sim.data.qvel[i] for i in
             range(len(sim.data.qvel))])))

        subgoal_thresholds = settings["subgoal_thresholds"]
        max_actions = settings["max_actions"]
        num_frames_skip = settings["num_frames_skip"]
        show = settings["show"]

        self.name = model_name

        # Create Mujoco Simulation
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_path, "mujoco_files/" + model_name)
        self.model = load_model_from_path(model_path)
        self.sim = MjSim(self.model)

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        if model_name == "pendulum.xml":
            self.state_dim = 2 * len(self.sim.data.qpos) + len(self.sim.data.qvel)
        else:
            self.state_dim = len(self.sim.data.qpos) + len(
                self.sim.data.qvel)  # State will include (i) joint angles and (ii) joint velocities
        self.action_dim = len(self.sim.model.actuator_ctrlrange)  # low-level action dim
        self.action_bounds = self.sim.model.actuator_ctrlrange[:, 1]  # low-level action bounds
        self.action_offset = np.zeros((len(self.action_bounds)))  # Assumes symmetric low-level action ranges
        self.end_goal_dim = len(goal_space_test)
        self.subgoal_dim = len(subgoal_bounds)
        self.subgoal_bounds = subgoal_bounds

        # Projection functions
        self.project_state_to_end_goal = project_state_to_end_goal
        self.project_state_to_subgoal = project_state_to_subgoal

        # Convert subgoal bounds to symmetric bounds and offset.
        # Need these to properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0]) / 2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        # End goal/subgoal thresholds
        self.end_goal_thresholds = end_goal_thresholds
        self.subgoal_thresholds = subgoal_thresholds

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space
        self.goal_space_train = goal_space_train
        self.goal_space_test = goal_space_test
        self.subgoal_colors = ["Magenta", "Green", "Red", "Blue", "Cyan", "Orange", "Maroon", "Gray", "White", "Black"]

        self.max_actions = max_actions

        # Implement visualization if necessary
        self.visualize = show  # Visualization boolean
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip

        self._action_space = ActionSpace(self._game_settings['action_bounds'])
        self._observation_space = ActionSpace(self._game_settings['state_bounds'])

        self.number_of_agents = self._game_settings['perform_multiagent_training'] if "perform_multiagent_training" in self._game_settings else 1
        self.__reward = [[0.0]] * self.number_of_agents

    # Get state, which concatenates joint positions and velocities
    def get_state(self):

        if self.name == "pendulum.xml":
            return np.concatenate([np.cos(self.sim.data.qpos), np.sin(self.sim.data.qpos),
                                   self.sim.data.qvel])
        else:
            return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    # Reset simulation to state within initial state specified by user
    def reset_sim(self):

        # Reset joint positions and velocities
        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0], self.initial_state_space[i][1])

        for i in range(len(self.sim.data.qvel)):
            self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],
                                                      self.initial_state_space[len(self.sim.data.qpos) + i][1])

        self.sim.step()

        self.end_goal = self.get_next_goal(False)

        # Return state
        return self.get_state()

    # Execute low-level action for number of frames specified by num_frames_skip
    def execute_action(self, action):

        self.sim.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        return self.get_state()

    # Visualize end goal.  This function may need to be adjusted for new environments.
    def display_end_goal(self, end_goal):

        # Goal can be visualized by changing the location of the relevant site object.
        if self.name == "pendulum.xml":
            self.sim.data.mocap_pos[0] = np.array([0.5 * np.sin(end_goal[0]), 0, 0.5 * np.cos(end_goal[0]) + 0.6])
        elif self.name == "ur5.xml":

            theta_1 = end_goal[0]
            theta_2 = end_goal[1]
            theta_3 = end_goal[2]

            # shoulder_pos_1 = np.array([0,0,0,1])
            upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
            forearm_pos_3 = np.array([0.425, 0, 0, 1])
            wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

            # Transformation matrix from shoulder to base reference frame
            T_1_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]])

            # Transformation matrix from upper arm to shoulder reference frame
            T_2_1 = np.array(
                [[np.cos(theta_1), -np.sin(theta_1), 0, 0], [np.sin(theta_1), np.cos(theta_1), 0, 0], [0, 0, 1, 0],
                 [0, 0, 0, 1]])

            # Transformation matrix from forearm to upper arm reference frame
            T_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0], [0, 1, 0, 0.13585],
                              [-np.sin(theta_2), 0, np.cos(theta_2), 0], [0, 0, 0, 1]])

            # Transformation matrix from wrist 1 to forearm reference frame
            T_4_3 = np.array(
                [[np.cos(theta_3), 0, np.sin(theta_3), 0.425], [0, 1, 0, 0], [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                 [0, 0, 0, 1]])

            # Determine joint position relative to original reference frame
            # shoulder_pos = T_1_0.dot(shoulder_pos_1)
            upper_arm_pos = T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3]
            forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
            wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

            joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

            """
            print("\nEnd Goal Joint Pos: ")
            print("Upper Arm Pos: ", joint_pos[0])
            print("Forearm Pos: ", joint_pos[1])
            print("Wrist Pos: ", joint_pos[2])
            """

            for i in range(3):
                self.sim.data.mocap_pos[i] = joint_pos[i]

        else:
            assert False, "Provide display end goal function in environment.py file"

    # Function returns an end goal
    def get_next_goal(self, test):

        end_goal = np.zeros((len(self.goal_space_test)))

        if self.name == "ur5.xml":

            goal_possible = False
            while not goal_possible:
                end_goal = np.zeros(shape=(self.end_goal_dim,))
                end_goal[0] = np.random.uniform(self.goal_space_test[0][0], self.goal_space_test[0][1])

                end_goal[1] = np.random.uniform(self.goal_space_test[1][0], self.goal_space_test[1][1])
                end_goal[2] = np.random.uniform(self.goal_space_test[2][0], self.goal_space_test[2][1])

                # Next need to ensure chosen joint angles result in achievable task (i.e., desired end effector position is above ground)

                theta_1 = end_goal[0]
                theta_2 = end_goal[1]
                theta_3 = end_goal[2]

                # shoulder_pos_1 = np.array([0,0,0,1])
                upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
                forearm_pos_3 = np.array([0.425, 0, 0, 1])
                wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

                # Transformation matrix from shoulder to base reference frame
                T_1_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]])

                # Transformation matrix from upper arm to shoulder reference frame
                T_2_1 = np.array(
                    [[np.cos(theta_1), -np.sin(theta_1), 0, 0], [np.sin(theta_1), np.cos(theta_1), 0, 0], [0, 0, 1, 0],
                     [0, 0, 0, 1]])

                # Transformation matrix from forearm to upper arm reference frame
                T_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0], [0, 1, 0, 0.13585],
                                  [-np.sin(theta_2), 0, np.cos(theta_2), 0], [0, 0, 0, 1]])

                # Transformation matrix from wrist 1 to forearm reference frame
                T_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3), 0.425], [0, 1, 0, 0],
                                  [-np.sin(theta_3), 0, np.cos(theta_3), 0], [0, 0, 0, 1]])

                forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
                wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

                # Make sure wrist 1 pos is above ground so can actually be reached
                if np.absolute(end_goal[0]) > np.pi / 4 and forearm_pos[2] > 0.05 and wrist_1_pos[2] > 0.15:
                    goal_possible = True

        elif not test and self.goal_space_train is not None:
            for i in range(len(self.goal_space_train)):
                end_goal[i] = np.random.uniform(self.goal_space_train[i][0], self.goal_space_train[i][1])
        else:
            assert self.goal_space_test is not None, "Need goal space for testing. Set goal_space_test variable in \"design_env.py\" file"

            for i in range(len(self.goal_space_test)):
                end_goal[i] = np.random.uniform(self.goal_space_test[i][0], self.goal_space_test[i][1])

        # Visualize End Goal
        self.display_end_goal(end_goal)

        return end_goal

    # Visualize all subgoals
    def display_subgoals(self, subgoals):

        # Display up to 10 subgoals and end goal
        if len(subgoals) <= 11:
            subgoal_ind = 0
        else:
            subgoal_ind = len(subgoals) - 11

        for i in range(1, min(len(subgoals), 11)):
            if self.name == "pendulum.xml":
                self.sim.data.mocap_pos[i] = np.array(
                    [0.5 * np.sin(subgoals[subgoal_ind][0]), 0, 0.5 * np.cos(subgoals[subgoal_ind][0]) + 0.6])
                # Visualize subgoal
                self.sim.model.site_rgba[i][3] = 1
                subgoal_ind += 1

            elif self.name == "ur5.xml":

                theta_1 = subgoals[subgoal_ind][0]
                theta_2 = subgoals[subgoal_ind][1]
                theta_3 = subgoals[subgoal_ind][2]

                # shoulder_pos_1 = np.array([0,0,0,1])
                upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
                forearm_pos_3 = np.array([0.425, 0, 0, 1])
                wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

                # Transformation matrix from shoulder to base reference frame
                T_1_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]])

                # Transformation matrix from upper arm to shoulder reference frame
                T_2_1 = np.array(
                    [[np.cos(theta_1), -np.sin(theta_1), 0, 0], [np.sin(theta_1), np.cos(theta_1), 0, 0], [0, 0, 1, 0],
                     [0, 0, 0, 1]])

                # Transformation matrix from forearm to upper arm reference frame
                T_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0], [0, 1, 0, 0.13585],
                                  [-np.sin(theta_2), 0, np.cos(theta_2), 0], [0, 0, 0, 1]])

                # Transformation matrix from wrist 1 to forearm reference frame
                T_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3), 0.425], [0, 1, 0, 0],
                                  [-np.sin(theta_3), 0, np.cos(theta_3), 0], [0, 0, 0, 1]])

                # Determine joint position relative to original reference frame
                # shoulder_pos = T_1_0.dot(shoulder_pos_1)
                upper_arm_pos = T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3]
                forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
                wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

                joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

                """
                print("\nSubgoal %d Joint Pos: " % i)
                print("Upper Arm Pos: ", joint_pos[0])
                print("Forearm Pos: ", joint_pos[1])
                print("Wrist Pos: ", joint_pos[2])
                """

                # Designate site position for upper arm, forearm and wrist
                for j in range(3):
                    self.sim.data.mocap_pos[3 + 3 * (i - 1) + j] = np.copy(joint_pos[j])
                    self.sim.model.site_rgba[3 + 3 * (i - 1) + j][3] = 1

                # print("\nLayer %d Predicted Pos: " % i, wrist_1_pos[:3])

                subgoal_ind += 1
            else:
                # Visualize desired gripper position, which is elements 18-21 in subgoal vector
                self.sim.data.mocap_pos[i] = subgoals[subgoal_ind]
                # Visualize subgoal
                self.sim.model.site_rgba[i][3] = 1
                subgoal_ind += 1

    def getActionSpaceSize(self):
        return self.action_dim

    def getObservationSpaceSize(self):
        return self.state_dim

    def setRandomSeed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def getNumAgents(self):
        return self.number_of_agents

    def updateAction(self, action):
        if isinstance(action, list) or isinstance(action, tuple):
            action = np.concatenate(action, -1)
        action, rest = action[:self.action_dim], action[self.action_dim:]
        if self.number_of_agents > 1:
            self.sub_goals = np.split(rest, self.number_of_agents, axis=(-1))
        else:
            self.sub_goals = []
        self.__action = action

        self.execute_action(self.__action)
        self.__reward = self.reward()

    def init(self):
        self.reset_sim()

    def initEpoch(self):
        self.reset_sim()

    def generateValidationEnvironmentSample(self, seed):
        self.initEpoch()

    def generateEnvironmentSample(self):
        self.initEpoch()

    """        
    def getAgentLocalState(self):
        st_ = self._map[(self._agent[0]-1)::2, (self._agent[1]-1)::2] # submatrix around agent location
        st_ = np.reshape(st_, (1,)) # Make vector
        return st_
    """

    """
    def act(self, action):
        print ("Trying discrete action: ", action)
        move = np.array(self.move(action))
        # loc = self._agent + (move * random.uniform(0.5,1.0))
        loc = self._agent + (move)

        if (((loc[0] < self._state_bounds[0][0]) or (loc[0] > self._state_bounds[1][0]) or 
            (loc[1] < self._state_bounds[0][1]) or (loc[1] > self._state_bounds[1][1])) or
            self.collision(loc) or
            self.fall(loc)):
            # Can't move out of map
            return self.reward() + -8

        # if self._map[loc[0]-1][loc[1]-1] == 1:
            # Can't walk onto obstacles
        #     return self.reward() +-5
        self._agent = loc
        return self.reward()
    """

    def actContinuous(self, action, bootstrapping):
        self.updateAction(action)
        return self.__reward

    def fall(self, loc):
        # Check to see if collision at loc with any obstacles
        # print (int(math.floor(loc[0])), int(math.floor(loc[1])))
        # if self._map[int(math.floor(loc[0]))][ int(math.floor(loc[1]))] < 0:
        #     return True
        return False

    def agentHasFallen(self):
        return False

    def collision(self, loc):
        # Check to see if collision at loc with any obstacles
        return False

    def reward(self):

        # WARNING: this is not exactly what the paper does, but as close as we can get
        state = self.get_state()

        return self.computeReward(state)

    def calcReward(self):
        return self.__reward

    def computeReward(self, state, next_state=None):

        distance_to_sub_goals = [-np.linalg.norm(g - state, ord=2) for g in self.sub_goals]
        distance_to_goal = -np.linalg.norm(self.end_goal - state[:self.end_goal_dim], ord=2)

        # Else goal is achieved
        return [[distance_to_goal]] + [[g] for g in distance_to_sub_goals]

    def getSimState(self):
        return self.get_state()

    def setSimState(self, state_):
        raise NotImplementedError

    def getState(self):
        return [np.concatenate((self.get_state(), self.end_goal), 0)] + [self.get_state()] * (self.number_of_agents - 1)

    def getStateForAgent(self, i):
        return self.get_state()

    def setState(self, st):
        raise NotImplementedError

    def getStateSamples(self):
        raise NotImplementedError

    def initRender(self, U, V, Q):
        pass

    def update(self):
        pass

    def display(self):
        pass

    def updatePolicy(self, U, V, Q):
        pass

    def updateMBAE(self, U, V, Q):
        pass

    def updateFD(self, U, V, Q):
        pass

    def reachedTarget(self):
        return self.reward() == 0

    def endOfEpoch(self):
        return self.reachedTarget()

    def saveVisual(self, fileName):
        pass

    def finish(self):
        pass

    def setLLC(self, llc):
        self._llc = llc

    def setHLP(self, hlp):
        self._hlp = hlp
