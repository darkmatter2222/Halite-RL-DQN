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
from .helpers.image_render_v2 import image_render_v2
from .helpers.stopwatch import stopwatch
from .helpers.random_agent import random_agent

# NOTE: This class is only to train a single bot to navigate, collect halite and return it to base.
# In later envs, we will introduce other bots

tf.compat.v1.enable_v2_behavior()

class halite_ship_navigation(py_environment.PyEnvironment):
    def __init__(self, window_name):
        self._this_stopwatch = stopwatch()
        print('Initializing Env')
        # game parameters
        self._board_size = 25
        self._max_turns = 400
        if self._max_turns > 20:
            self._frames = 20
        else:
            self._frames = self._max_turns
        self._agent_count = 2
        self._channels = 4
        self._action_def = {0: ShipAction.EAST,
                            1: ShipAction.NORTH,
                            2: "NOTHING",
                            3: ShipAction.SOUTH,
                            4: ShipAction.WEST}
        self.render_step = True

        # runtime parameters
        self.turns_counter = 0
        self.episode_ended = False
        self.total_reward = 0
        self.ships_idle = []
        self.shipyards_idle = []
        self.last_reward = 0

        # initialize game
        self.environment = make("halite", configuration={"size": self._board_size, "startingHalite": 1000,
                                                         "episodeSteps": self._max_turns })
        self.environment.reset(self._agent_count)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(self._action_def)-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._frames, self._channels, self._board_size, self._board_size), dtype=np.int32, minimum=0,
            maximum=1, name='observation')

        self.state = np.zeros([self._channels, self._board_size, self._board_size])
        # 0 = Halite 0-1
        # 1 = Ships (This One Hot, rest are .5)
        # 2 = Shipyards (This One Hot, rest are .5)
        # 3 = Halite Heat Map

        self.state_history = [self.state] * self._frames

        # get board
        self.board = self.get_board()
        self.prime_board()
        self.halite_image_render = image_render_v2(self._board_size)
        self.previous_ship_count = 0
        print(f'Initialized at {self._this_stopwatch.elapsed()}')

    def action_spec(self):
        return_object = self._action_spec
        return return_object

    def observation_spec(self):
        return_object = self._observation_spec
        return return_object

    def _reset(self):
        self.last_reward = 0
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
        self.state = np.zeros([self._channels, self._board_size, self._board_size])
        self.state_history = [self.state] * self._frames

        self.prime_board()
        return_object = ts.restart(np.array(self.state_history, dtype=np.int32))
        return return_object

    def _step(self, action):
        if self.episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return_object = self.reset()
            return return_object

        reward = 0
        self.state = self.get_state_v2()

        # global rules
        if self.turns_counter == self._max_turns:
            self.episode_ended = True

        int_action = int(action)

        if not self._action_def[int_action] == 'NOTHING' and '2-1' in self.board.ships:
            self.board.ships['2-1'].next_action = self._action_def[int_action]

        random_agent(self.board, self.board.players[1])

        self.board = self.board.next()

        self.state = self.get_state_v2()

        if len(self.board.players[0].ships) == 0:
            self.episode_ended = True
        if len(self.board.players[0].shipyards) == 0:
            self.episode_ended = True

        if not self.episode_ended:
            distance = self.board.players[0].ships[0].position - self.board.players[0].shipyards[0].position
            if abs(distance)[0] > 5 or abs(distance)[1] > 5:
                reward += -10000
                self.episode_ended = True

        ship_cargo = 0
        for ship in self.board.players[0].ships:
            ship_cargo += ship.halite

        reward += (self.board.players[0].halite * 2) + ship_cargo

        temp_reward = reward
        reward = reward - self.last_reward
        self.last_reward = temp_reward
        self.total_reward += reward

        if self.render_step:
            self.halite_image_render.render_board(self.board, self.state,
                                                  total_reward=self.total_reward, this_step_reward=reward)

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

    def set_rendering(self, enabled=True):
        self.render_step = enabled

    def prime_board(self):
        self.board.players[0].ships[0].next_action = ShipAction.CONVERT
        self.board = self.board.next()
        self.state = self.get_state_v2()
        self.state_history.append(self.state)
        del self.state_history[:1]
        self.turns_counter += 1
        self.board.players[0].shipyards[0].next_action = ShipyardAction.SPAWN
        self.board = self.board.next()
        self.state = self.get_state_v2()
        self.state_history.append(self.state)
        del self.state_history[:1]
        self.turns_counter += 1


    def get_board(self):
        obs = self.environment.state[0].observation
        config = self.environment.configuration
        actions = [agent.action for agent in self.environment.state]
        return Board(obs, config, actions)

    def get_state_v2(self):
        # this method, we are constructing both the board to be rendered and what is provided to the neural network.
        reward_heatmap = np.zeros([self._board_size, self._board_size])
        # First Render the Heatmap
        for x in range(0, self._board_size):
            for y in range(0, self._board_size):
                cell = self.board[(x, self._board_size - y - 1)]
                cell_halite = 255 * cell.halite / float(self.board.configuration.max_cell_halite)
                reward_heatmap[y, x] = cell_halite
                if cell.ship_id == '2-1':
                    reward_heatmap[y, x] += cell.ship.halite # make it tasty to return home
        sigma = [self._board_size / 2, self._board_size / 2]
        reward_heatmap = sp.ndimage.filters.gaussian_filter(reward_heatmap, sigma, mode='constant')
        # Next, render the state
        state_pixels = np.zeros([self._channels, self._board_size, self._board_size])
        for x in range(0, self._board_size):
            for y in range(0, self._board_size):
                cell = self.board[(x, self._board_size - y - 1)]
                # cell_halite = int(9.0 * cell.halite / float(board.configuration.max_cell_halite))
                cell_halite = 1.0 * cell.halite / float(self.board.configuration.max_cell_halite)

                # 0 = Halite
                # 1 = Ship Presence (One Hot 'ship_id', rest 0.5)
                # 2 = Shipyard Presence (One Hot 'ship_id', rest 0.5)
                # 3 = Halite Heat Map
                state_pixels[0, y, x] = cell_halite
                if cell.ship is not None:
                    state_pixels[1, y, x] = 1
                elif cell.shipyard is not None:
                    state_pixels[2, y, x] = 1
                state_pixels[3, y, x] = reward_heatmap[y, x]
        return state_pixels

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