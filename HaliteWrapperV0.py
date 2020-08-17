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
        # game parameters
        self._board_size = 3
        self._frames = 10
        self._max_turns = self._frames
        self._agent_count = 1
        self._channels = 3
        self._action_def = {0: ShipAction.EAST,
                            1: ShipAction.NORTH,
                            2: "NOTHING",
                            3: ShipAction.SOUTH,
                            4: ShipAction.WEST}

        # runtime parameters
        self.turns_counter = 0
        self.episode_ended = False
        self.total_reward = 0

        # initialize game
        self.environment = make("halite", configuration={"size": self._board_size, "startingHalite": 1000,
                                                         "episodeSteps": self._frames})
        self.environment.reset(self._agent_count)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(self._action_def)-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._frames, self._board_size, self._board_size, self._channels), dtype=np.int32, minimum=0,
            maximum=1, name='observation')

        self.state = np.zeros([self._board_size, self._board_size, self._channels])
        # 0 = Halite 0-1
        # 1 = Friendly Ships (This One Hot, rest are .75)

        self.state_history = [self.state] * self._frames

        # get board
        self.board = self.get_board()

    def action_spec(self):
        return_object = self._action_spec
        return return_object

    def observation_spec(self):
        return_object = self._observation_spec
        return return_object

    def _reset(self):
        self.turns_counter = 0
        self.episode_ended = False
        self.total_reward = 0
        # initialize game
        self.environment = make("halite", configuration={"size": self._board_size, "startingHalite": 1000,
                                                         "episodeSteps": self._frames})
        self.environment.reset(self._agent_count)
        # get board
        self.board = self.get_board()
        return_object = ts.restart(np.array(self.state_history, dtype=np.int32))
        return return_object

    def _step(self, action):
        if self.episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return_object = self.reset()
            return return_object

        # max turns?
        if self.turns_counter == self._max_turns:
            self.episode_ended = True

        # take action
        current_player = self.board.current_player
        if current_player.id == 0:
            for ship in current_player.ships:
                cargo = ship.halite
                int_action = int(action)
                if self._action_def[int_action] != "NOTHING":
                    ship.next_action = self._action_def[int_action]
                elif self._action_def[int_action] == "NOTHING":
                    lol = 1
                else:
                    raise Exception('invalid action received')
                break
                # TODO just because we only have 1 ship
        reward = 0

        # commit
        self.board = self.board.next()

        # calculate reward
        if current_player.id == 0:
            for ship in current_player.ships:
                cargo_delta = ship.halite - cargo
                reward = cargo_delta
                break

        self.total_reward += reward
        # get new state
        self.state = self.get_state()

        # final wrap up
        self.turns_counter += 1
        self.state_history.append(self.state)
        del self.state_history[:1]
        # final
        if self.episode_ended:
            return_object = ts.termination(np.array(self.state_history, dtype=np.int32), reward)
            return return_object
        else:
            return_object = ts.transition(np.array(self.state_history, dtype=np.int32), reward=reward, discount=1.0)
            return return_object

    def get_board(self):
        obs = self.environment.state[0].observation
        config = self.environment.configuration
        actions = [agent.action for agent in self.environment.state]
        return Board(obs, config, actions)

    def get_state(self):
        size = self.board.configuration.size
        pixels = []
        for x in range(0, size):
            row = []
            for y in range(0, size):
                cell = self.board[(x, size - y - 1)]
                # cell_halite = int(9.0 * cell.halite / float(board.configuration.max_cell_halite))
                cell_halite = 1.0 * cell.halite / float(self.board.configuration.max_cell_halite)

                pixel = [0, 0, 0]
                if cell.ship is not None:
                    pixel[1] = 1
                if cell.shipyard is not None:
                    pixel[2] = 1
                pixel[0] = cell_halite
                # 0 = Halite (normalized)
                # 1 = Ship Presence
                # 2 = Shipyard Presence

                row.append(np.array(pixel))
            pixels.append(np.array(row))
        return np.array(pixels)