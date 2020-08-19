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
from helpers.HaliteImageRender import HaliteImageRender


tf.compat.v1.enable_v2_behavior()

class HaliteWrapperV0(py_environment.PyEnvironment):
    def __init__(self):
        # game parameters
        self._board_size = 5
        self._max_turns = 400
        if self._max_turns > 20:
            self._frames = 20
        else:
            self._frames = self._max_turns
        self._agent_count = 2
        self._channels = 3
        self._action_def = {0: ShipAction.EAST,
                            1: ShipAction.NORTH,
                            2: "NOTHING",
                            3: ShipAction.SOUTH,
                            4: ShipAction.WEST,
                            5: ShipAction.CONVERT,
                            6: ShipyardAction.SPAWN}

        # runtime parameters
        self.turns_counter = 0
        self.episode_ended = False
        self.total_reward = 0
        self.ships_idle = []
        self.shipyards_idle = []

        # initialize game
        self.environment = make("halite", configuration={"size": self._board_size, "startingHalite": 1000,
                                                         "episodeSteps": self._max_turns })
        self.environment.reset(self._agent_count)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(self._action_def)-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._frames, self._board_size, self._board_size, self._channels), dtype=np.int32, minimum=0,
            maximum=1, name='observation')

        self.state = np.zeros([self._board_size, self._board_size, self._channels])
        # 0 = Halite 0-1
        # 1 = Ships (This One Hot, rest are .75)

        self.state_history = [self.state] * self._frames

        # get board
        self.board = self.get_board()
        self.halite_image_render = HaliteImageRender(self._board_size)
        self.previous_ship_count = 0

    def action_spec(self):
        return_object = self._action_spec
        return return_object

    def observation_spec(self):
        return_object = self._observation_spec
        return return_object

    def _reset(self):
        self.turns_counter = 0
        self.previous_ship_count = 0
        self.episode_ended = False
        self.total_reward = 0
        # initialize game
        self.environment = make("halite", configuration={"size": self._board_size, "startingHalite": 1000,
                                                         "episodeSteps": self._max_turns })
        self.environment.reset(self._agent_count)
        # get board
        self.board = self.get_board()
        self.state_history = [self.state] * self._frames
        return_object = ts.restart(np.array(self.state_history, dtype=np.int32))
        return return_object

    def _step(self, action):
        if self.episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return_object = self.reset()
            return return_object

        reward = 0
        self.state, actionable_object_id, actionable_type = self.get_state()

        if actionable_type == 'UNKNOWN':
            self.episode_ended = True

        # global rules
        if self.turns_counter == self._max_turns:
            self.episode_ended = True

        if self.episode_ended == False:
            int_action = int(action)
            if actionable_type == 'ship':
                if action == 6:
                    reward += -1000
                    self.episode_ended = True
            else:
                if action != 6 and action != 2:
                    reward += -1000
                    self.episode_ended = True

        if self.episode_ended == False:
            if actionable_type == 'ship':
                if self._action_def[int_action] != "NOTHING":
                    self.board.ships[actionable_object_id].next_action = self._action_def[int_action]
                else:
                    self.ships_idle.append(actionable_object_id)
            else:
                if self._action_def[int_action] != "NOTHING":
                    self.board.shipyards[actionable_object_id].next_action = self._action_def[int_action]
                else:
                    self.shipyards_idle.append(actionable_object_id)

        self.state, actionable_object_id, actionable_type = self.get_state()

        self.halite_image_render.render_board(self.board)

        # final rules
        if len(self.board._players[0].ships) < self.previous_ship_count:
            reward -= 1000

        if len(self.board._players[0].ships) < 1:
            self.episode_ended = True

        self.previous_ship_count = len(self.board._players[0].ships)

        self.total_reward += reward
        # get new state
        one_hot_ship_id = '99'
        for ship in self.board.ships:
            if self.board.ships[ship]._player_id == 0 and self.board.ships[ship].next_action != None:
                one_hot_ship_id = ship
                break

        self.state, actionable_ship_id = self.get_state(one_hot_ship_id)

        # final wrap up
        self.turns_counter += 1
        self.state_history.append(self.state)
        del self.state_history[:1]

        #self.renderer()

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
        actionable_object_id = None
        actionable_type = 'UNKNOWN'
        loop_counter = 0

        while True:
            # ship check first
            for ship_id in self.board.ships:
                if self.board.ships[ship_id]._player_id == 0 and self.board.ships[ship_id].next_action == None \
                        and ship_id not in self.ships_idle:
                    actionable_object_id = ship_id
                    actionable_type = 'ship'
                    break

            if actionable_type == 'UNKNOWN':
                for shipyard_id in self.board.shipyards:
                    if self.board.shipyards[shipyard_id]._player_id == 0 and self.board.shipyards[shipyard_id].next_action == None \
                        and shipyard_id not in self.shipyards_idle:
                        actionable_object_id = shipyard_id
                        actionable_type = 'shipyard'
                        break

            if actionable_type == 'UNKNOWN':
                self.board = self.board.next()
                self.ships_idle = []
                self.shipyards_idle = []
                if loop_counter == 1:
                    break
                loop_counter += 1
            else:
                break

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
                    if cell.ship.id == ship_id:
                        pixel[1] = 1
                    else:
                        pixel[1] = 0.5
                if cell.shipyard is not None:
                    pixel[2] = 1
                pixel[0] = cell_halite
                # 0 = Halite
                # 1 = Ship Presence (One Hot 'ship_id', rest 0.5)
                # 2 = Shipyard Presence

                row.append(np.array(pixel))
            pixels.append(np.array(row))
        return np.array(pixels), actionable_object_id, actionable_type

    def renderer(self, highlight=None):
        size = self.board.configuration.size
        sudo_board = []
        shift_delta = Point(0, 0)
        if highlight != None:
            shift_delta = Point(4, 4) - highlight

        for y in range(size):
            sudo_board.append(['   '] * 10)

        for y in range(size):
            for x in range(size):
                board_cell = self.board[(x, size - y - 1)]
                bord_cell_halite = int(9.0 * board_cell.halite / float(self.board.configuration.max_cell_halite))
                precolor = ''
                postcolor = ''

                if bord_cell_halite > 0:
                    precolor = '\x1b[32m'
                    postcolor = '\x1b[0m'

                if board_cell.ship is not None:
                    precolor = '\x1b[31m'
                    postcolor = '\x1b[0m'
                if board_cell.shipyard is not None:
                    precolor = '\x1b[31m'
                    postcolor = '\x1b[0m'

                if highlight != None:
                    if (x, size - y - 1) == highlight:
                        precolor = '\x1b[36m'
                        postcolor = '\x1b[0m'

                sudo_cell = ''
                sudo_cell += precolor

                if board_cell.ship is not None:
                    sudo_cell += chr(ord('a') + board_cell.ship.player_id)
                else:
                    sudo_cell += ' '

                sudo_cell += str(bord_cell_halite)

                if board_cell.shipyard is not None:
                    sudo_cell += chr(ord('A') + board_cell.shipyard.player_id)
                else:
                    sudo_cell += ' '

                sudo_cell += postcolor
                sudo_board[y][x] = str(sudo_cell)

        shifted_result = ''
        for y in range(size):
            for x in range(size):
                shifted_result += '|'
                shifted_result += sudo_board[y][x]
            shifted_result += '|\n'

        # os.system('cls' if os.name == 'nt' else 'clear')
        print(shifted_result)