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
        self.action_def = {0: ShipAction.EAST,
                           1: ShipAction.NORTH,
                           2: "NOTHING",
                           3: ShipAction.SOUTH,
                           4: ShipAction.WEST}

        self.board_size = 4
        self.environment = make("halite", configuration={"size": self.board_size, "startingHalite": 1000})
        self.agent_count = 1
        self.environment.reset(self.agent_count)
        #self.state = self.environment.state[0]
        self.environment.reset()

        self.board = data.env_to_board(self.environment)
        observation = data.board_to_state(self.board)
        obs_shape = observation.shape
        state = np.array([observation])
        state_shape = state.shape
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, maximum=4, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=state.shape, dtype=np.int32, minimum=0, maximum=500, name='observation')

        self._state = state
        self._episode_ended = False
        self.total_reward = 0
        self.historical_action = []

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.environment.reset(self.agent_count)
        self.board = data.env_to_board(self.environment)
        observation = data.board_to_state(self.board)
        obs_shape = observation.shape
        state = np.array([observation])
        state_shape = state.shape
        self._state = state
        self._episode_ended = False
        self.total_reward = 0
        print(f'history:{self.historical_action}')
        self.historical_action = []
        return ts.restart(np.array(self._state, dtype=np.int32)) #

    def _step(self, action):
        self.historical_action.append(action[0])
        if self.board.step >= self.environment.configuration.episodeSteps:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        cargo = 0
        cargo_delta = 0
        current_player = self.board.current_player
        if current_player.id == 0:
            for ship in current_player.ships:
                cargo = ship.halite
                if self.action_def[action[0]] != "NOTHING":
                    ship.next_action = self.action_def[action[0]]
                if self.action_def[action[0]] == "NOTHING":
                    lol = 1
                break
        self.board = self.board.next()
        current_player = self.board.current_player
        if current_player.id == 0:
            for ship in current_player.ships:
                cargo_delta = ship.halite - cargo
                if cargo_delta > 0:
                    lol = 1
                else:
                    if self.action_def[action[0]] == "NOTHING":
                        cargo_delta += -2
                    else:
                        cargo_delta += -1

                break

        reward = cargo_delta
        self.total_reward += reward
        if reward > 0:
            lol = 1

        observation = data.board_to_state(self.board)
        self._state = np.array([observation])
        if self.board.step >= self.environment.configuration.episodeSteps:
            return ts.termination(np.array(self._state, dtype=np.int32), 0.0)
        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=reward, discount=0.5)



