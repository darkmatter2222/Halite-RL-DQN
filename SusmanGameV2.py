from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

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


tf.compat.v1.enable_v2_behavior()

class CardGameEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._episode_ended = False

    def action_spec(self):
        return_object = self._action_spec
        return return_object

    def observation_spec(self):
        return_object = self._observation_spec
        return return_object

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        return_object = ts.restart(np.array([self._state], dtype=np.int32))
        return return_object

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return_object = self.reset()
            return return_object

        # Make sure episodes don't go on forever.
        if action == 1:
            self._episode_ended = True
        elif action == 0:
            new_card = np.random.randint(1, 11)
            self._state += new_card
        else:
            raise ValueError('`action` should be 0 or 1.')

        if self._episode_ended or self._state >= 21:
            reward = self._state - 21 if self._state <= 21 else -21
            return_object = ts.termination(np.array([self._state], dtype=np.int32), reward)
            return return_object
        else:
            return_object = ts.transition(np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
            return return_object


class SusmanGameV2(py_environment.PyEnvironment):
    def __init__(self):
        self.board_width = 4
        self.board_height = 4
        self.sigma_y = self.board_width / 2
        self.sigma_x = self.board_height / 2
        self.channels = 2
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1, self.board_height, self.board_width, self.channels), dtype=np.int32, minimum=0, maximum=1, name='observation')
        self._state = np.zeros([self.board_height, self.board_width, self.channels])
        # 0 = Reward Placement
        # 1 = Player Placement
        # 2 = Heat Map
        self._episode_ended = False
        self.this_turn = 0
        self.max_turns = 300
        self.total_reward = 0
        self.heatmap_reward = self.board_height * self.board_width
        self.player_location = {'y': 0, 'x': 0}
        self.set_player()
        self.set_goal()

    def set_goal(self):
        self.set_player()
        rand_y = 0
        rand_x = 0

        while True:
            if self.board_width > 1:
                rand_x = random.randrange(0, self.board_width - 1)
            if self.board_height > 1:
                rand_y = random.randrange(0, self.board_height - 1)
            if self.player_location['x'] != rand_x or self.player_location['y'] != rand_y:
                break

        self._state[rand_y, rand_x][0] = 1

        #self._state[rand_y, rand_x][2] = self.heatmap_reward



        # Apply gaussian filter
        #sigma = [self.sigma_y, self.sigma_x]
        #self._state[::2] = sp.ndimage.filters.gaussian_filter(self._state[::2], sigma, mode='constant')

        lol = 1

    def set_player(self):
        rand_y = 0
        rand_x = 0

        if self.board_width > 1:
            rand_x = random.randrange(0, self.board_width - 1)
        if self.board_height > 1:
            rand_y = random.randrange(0, self.board_height - 1)
        self.player_location['x'] = rand_x
        self.player_location['y'] = rand_y
        self._state[rand_y, rand_x][1] = 1

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
        self.set_player()
        self.set_goal()
        return_object = ts.restart(np.array([self._state], dtype=np.int32))
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
        win_reward = 1
        loose_reward = -1
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
                self.player_location['x'] = self.player_location['y'] + 1
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
        else:
            # Loose Fall Off Map?
            if self.player_location['y'] < 0 or self.player_location['x'] < 0 or \
                    self.player_location['x'] >= self.board_width or self.player_location['y'] >= self.board_height:
                info = 'Loose Fall Off Map'
                self._episode_ended = True
                reward = loose_reward
            elif self._state[self.player_location['y'], self.player_location['x']][0] == 1:
                info = 'Won Got the Goal'
                self._episode_ended = True
                reward = win_reward
            #elif self._state[self.player_location['y'], self.player_location['x']][2] != 0:
                #info = 'Continue w/ reward'
                #self._episode_ended = False
                #reward = continue_reward + self._state[self.player_location['y'], self.player_location['x']][0]
            else:
                info = 'Continue'
                self._episode_ended = False
                reward = continue_reward

        self.total_reward += reward
        self.this_turn += 1

        if self._episode_ended:
            return_object = ts.termination(np.array([self._state], dtype=np.int32), reward)
            return return_object
        else:
            return_object = ts.transition(np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
            return return_object