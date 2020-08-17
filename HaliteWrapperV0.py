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

class HaliteWrapperV0(py_environment.PyEnvironment):
    def __init__(self):
        # game parms
        self._board_size = 3
        self._frames = 400
        self._max_turns = self._frames
        self._agent_count = 1
        self._channels = 2
        self._action_def = {0: ShipAction.EAST,
                           1: ShipAction.NORTH,
                           2: "NOTHING",
                           3: ShipAction.SOUTH,
                           4: ShipAction.WEST}


        # runtime_parms
        self.turns_counter = 0
        self.episode_ended = False

        # initialize game
        self.environment = make("halite", configuration={"size": self._board_size, "startingHalite": 1000})
        self.environment.reset(self._agent_count)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(self._action_def), name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._frames, self._board_size, self._board_size, self._channels), dtype=np.int32, minimum=0,
            maximum=1, name='observation')

        self.state = np.zeros([self._board_size, self._board_size, self._channels])
        # 0 = Halite 0-1
        # 1 = Friendly Ships (This One Hot, rest are .75)

        self.state_history = [self.state] * self._frames




    def action_spec(self):
        return_object = self._action_spec
        return return_object

    def observation_spec(self):
        return_object = self._observation_spec
        return return_object

    def _reset(self):
        self.turns_counter = 0
        self.episode_ended = False
        return_object = ts.restart(np.array(self.state_history, dtype=np.int32))
        return return_object

    def _step(self, action):
        if self.episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return_object = self.reset()
            return return_object

        if self.turns_counter == self._max_turns:
            self.episode_ended = True



        # final wrap up
        self.turns_counter += 1
        # final
        if self.episode_ended:
            return_object = ts.termination(np.array(self.state_history, dtype=np.int32), 0.0)
            return return_object
        else:
            return_object = ts.transition(np.array(self.state_history, dtype=np.int32), reward=0.0, discount=1.0)
            return return_object