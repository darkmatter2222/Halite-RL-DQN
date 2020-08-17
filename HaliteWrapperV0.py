from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
import random
import scipy as sp
import cv2
import uuid
import matplotlib


tf.compat.v1.enable_v2_behavior()

class SusmanGameV3(py_environment.PyEnvironment):
    def __init__(self):
        # game parms
        self.board_size = 3
        self.frames = 400
        
        # initialize game
        self.environment = make("halite", configuration={"size": self.board_size, "startingHalite": 1000})
        self.agent_count = 1
        self.environment.reset(self.agent_count)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.frames, self.board_size, self.board_size, self.channels), dtype=np.int32, minimum=0,
            maximum=1, name='observation')
        self._state = np.zeros([self.board_size, self.board_size, self.channels])


    def action_spec(self):
        return_object = self._action_spec
        return return_object

    def observation_spec(self):
        return_object = self._observation_spec
        return return_object

    def _reset(self):
        self._state = np.zeros([self.board_height, self.board_width, self.channels])
        self._episode_ended = False
        self.total_reward = 0
        self.player_location = {'y': 0, 'x': 0}
        self.this_turn = 0
        self.set_goal()
        self.game_history = []
        self.state_history = [self._state] * self.frames
        self.render_image()
        return_object = ts.restart(np.array(self.state_history, dtype=np.int32))
        return return_object

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return_object = self.reset()
            return return_object

        reward = 0
        info = ''
        map_edge_exist = True
        continue_reward = -1
        win_reward = 10
        loose_reward = -10
        # 0=N 1=E 2=S 3=W
        if action == 0:  # Move North
            if map_edge_exist:
                self.player_location['y'] = self.player_location['y'] - 1
            else:
                if self.player_location['y'] == 0:
                    self.player_location['y'] = self.board_height - 1
                else:
                    self.player_location['y'] = self.player_location['y'] - 1
        elif action == 1:  # Move East
            if map_edge_exist:
                self.player_location['x'] = self.player_location['x'] + 1
            else:
                if self.player_location['x'] == self.board_width - 1:
                    self.player_location['x'] = 0
                else:
                    self.player_location['x'] = self.player_location['x'] + 1
        elif action == 2:  # Move South
            if map_edge_exist:
                self.player_location['y'] = self.player_location['y'] + 1
            else:
                if self.player_location['y'] == self.board_height - 1:
                    self.player_location['y'] = 0
                else:
                    self.player_location['y'] = self.player_location['y'] + 1
        elif action == 3:  # Move West
            if map_edge_exist:
                self.player_location['x'] = self.player_location['x'] - 1
            else:
                if self.player_location['x'] == 0:
                    self.player_location['x'] = self.board_width - 1
                else:
                    self.player_location['x'] = self.player_location['x'] - 1
        else:
            raise ValueError

        # Max Tries?
        if self.this_turn == self.max_turns - 1:
            info = 'Max Tries'
            self._episode_ended = True
            reward += loose_reward
        else:
            # Loose Fall Off Map?
            if self.player_location['y'] < 0 or self.player_location['x'] < 0 or \
                    self.player_location['x'] >= self.board_width or self.player_location['y'] >= self.board_height:
                info = 'Loose Fall Off Map'
                self._episode_ended = True
                reward += loose_reward
            elif self._state[self.player_location['y'], self.player_location['x']][0] == 1:
                info = 'Won Got the Goal'
                self._episode_ended = True
                reward += win_reward
            elif self._state[self.player_location['y'], self.player_location['x']][2] != 0:
                info = 'Continue w/ reward'
                self._episode_ended = False
                reward += continue_reward + self._state[self.player_location['y'], self.player_location['x']][0]
            else:
                info = 'Continue'
                self._episode_ended = False
                reward += continue_reward

        # preference
        if (action == 0 or action == 2) and self.this_turn < 2:
            reward += 0.1

        self.render_new_state()


        self.game_history.append(info)
        self.total_reward += reward
        self.this_turn += 1
        self.render_image()

        self.state_history.append(self._state)
        del self.state_history[:1]
        if self._episode_ended:
            return_object = ts.termination(np.array(self.state_history, dtype=np.int32), reward)
            return return_object
        else:
            return_object = ts.transition(np.array(self.state_history, dtype=np.int32), reward=reward, discount=1.0)
            return return_object