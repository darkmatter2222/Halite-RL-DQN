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
import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
from helpers.data import data


class SusmanHalite(py_environment.PyEnvironment):
    def __init__(self):
        self.board_size = 10
        self.environment = make("halite", configuration={"size": self.board_size, "startingHalite": 1000})
        self.agent_count = 2
        self.environment.reset(self.agent_count)
        #self.state = self.environment.state[0]
        self.environment.reset()

        board = data.env_to_board(self.environment)
        observation = data.board_to_state(board)
        obs_shape = observation.shape
        state = np.array([observation])
        state_shape = state.shape
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=4, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=state.shape, dtype=np.int32, minimum=0, name='observation')

        self._state = state
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.environment.reset(self.agent_count)
        board = data.env_to_board(self.environment)
        observation = data.board_to_state(board)
        obs_shape = observation.shape
        state = np.array([observation])
        state_shape = state.shape
        self._state = state
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32)) #

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        if action == 1:
            self._episode_ended = True
        elif action == 0:
            new_card = np.random.randint(1, 11)
            self._state[0] += new_card
        else:
            new_card = np.random.randint(1, 11)
            self._state[0] += new_card

        if self._episode_ended or self._state[0] >= 21:
            reward = self._state[0] - 21 if self._state[0] <= 21 else -21
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)