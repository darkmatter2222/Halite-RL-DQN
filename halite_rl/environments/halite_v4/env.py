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
from halite_rl.environments.halite_v4.helpers.image_render_v2 import image_render_v2
from halite_rl.environments.halite_v4.helpers.stopwatch import stopwatch
from halite_rl.environments.halite_v4.helpers.random_agent import random_agent

# NOTE: This class is only to train a single bot to navigate, collect halite and return it to base.
# In later envs, we will introduce other bots

tf.compat.v1.enable_v2_behavior()

class halite_ship_navigation(py_environment.PyEnvironment):
    def __init__(self, env_name, render_me=True):
        self._this_stopwatch = stopwatch()
        print('Initializing Env')
        # game parameters
        self._board_size = 5
        self._max_turns = 400
        if self._max_turns > 5:
            self._frames = 5
        else:
            self._frames = self._max_turns
        self._agent_count = 1
        self._channels = 3
        self._action_def = {0: ShipAction.EAST,
                            1: ShipAction.NORTH,
                            2: "NOTHING",
                            3: ShipAction.SOUTH,
                            4: ShipAction.WEST}
        self.render_step = render_me
        self.window_name = f''
        self._env_name = env_name

        # runtime parameters
        self.turns_counter = 0
        self.episode_ended = False
        self.total_reward = 0
        self.ships_idle = []
        self.shipyards_idle = []
        self.last_reward = 0

        self.turns_not_moved = 0
        self.last_action = 'NOTHING'
        self.max_turns_not_moved = 10

        self.action_history = []

        # initialize game
        self.environment = make("halite", configuration={"size": self._board_size, "startingHalite": 1000,
                                                         "episodeSteps": self._max_turns })
        self.environment.reset(self._agent_count)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(self._action_def)-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._frames, self._channels, self._board_size, self._board_size), dtype=np.float, minimum=0.0,
            maximum=1.0, name='observation')

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
        self.turns_not_moved = 0
        self.action_history = []
        # initialize game
        self.environment = make("halite", configuration={"size": self._board_size, "startingHalite": 1000,
                                                         "episodeSteps": self._max_turns })
        self.environment.reset(self._agent_count)
        # get board
        self.board = self.get_board()
        self.state = np.zeros([self._channels, self._board_size, self._board_size])
        self.state_history = [self.state] * self._frames

        self.prime_board()
        return_object = ts.restart(np.array(self.state_history, dtype=np.float))
        return return_object

    def _step(self, action):
        # ===initialize variables===
        int_action = int(action)
        self.action_history.append(self._action_def[int_action])
        reward = 0
        halite_after_turn = 0
        cargo_after_turn = 0
        halite_after_turn = 0
        cargo_after_turn = 0

        grant_cargo_delta_reward = False

        # ===get cargo and halite before turn===
        if '2-1' in self.board.ships:
            cargo_before_turn = self.board.ships['2-1'].halite
        halite_before_turn = self.board.players[0].halite

        # ===calculate reward and perform move===
        # if move = NOTHING and if adjacent cells are > current cell then punishment (what was max delta)
        # if move = NOTHING and if adjacent cells are < current cell then reward (what was collected)
        if self._action_def[int_action] == 'NOTHING':
            this_ship = self.board.ships['2-1']
            this_cell_halite = this_ship.cell.halite

            adj_halite = [this_ship.cell.north.halite,
                          this_ship.cell.south.halite,
                          this_ship.cell.east.halite,
                          this_ship.cell.west.halite]

            if this_cell_halite > np.max(adj_halite):
                reward += this_cell_halite - np.max(adj_halite)
            else:
                grant_cargo_delta_reward = True
        # if move = (N,S,E,W) and target cells is > current cell then reward (target cell halite value)
        # if move = (N,S,E,W) and target cells is < current cell then punishment (target cell halite delta)
        elif not self._action_def[int_action] == 'NOTHING':
            this_ship = self.board.ships['2-1']
            this_cell_halite = this_ship.cell.halite
            delta = 0

            if self._action_def[int_action] == ShipAction.NORTH:
                delta = this_ship.cell.north.halite - this_cell_halite
            elif self._action_def[int_action] == ShipAction.SOUTH:
                delta = this_ship.cell.south.halite - this_cell_halite
            elif self._action_def[int_action] == ShipAction.EAST:
                delta = this_ship.cell.east.halite - this_cell_halite
            elif self._action_def[int_action] == ShipAction.WEST:
                delta = this_ship.cell.west.halite - this_cell_halite

            reward += delta

            self.board.ships['2-1'].next_action = self._action_def[int_action]

        # ===move random bots===
        #random_agent(self.board, self.board.players[1])
        #random_agent(self.board, self.board.players[2])
        #random_agent(self.board, self.board.players[3])

        # ===perform move===
        self.board = self.board.next()
        self.state, heat_map = self.get_state_v2()

        # ===get cargo and halite after turn===
        if '2-1' in self.board.ships:
            cargo_after_turn = self.board.ships['2-1'].halite
        halite_after_turn = self.board.players[0].halite

        # ===additional grants===
        if grant_cargo_delta_reward:
            reward += cargo_after_turn - cargo_before_turn

        # ===determine if game over=== (no punishment)
        # no ship
        if len(self.board.players[0].ships) == 0:
            self.episode_ended = True
        # no shipyard
        if len(self.board.players[0].shipyards) == 0:
            self.episode_ended = True
        # max turns
        if self.turns_counter == self._max_turns:
            self.episode_ended = True
        # distance qualifier
        #if not self.episode_ended:
            #distance = self.board.ships['2-1'].position - self.board.shipyards['1-1'].position
            #if abs(distance)[0] > 5 or abs(distance)[1] > 5:
                #reward -= 1000
                #self.episode_ended = True

        # ===discouragements===
        last_five_history = self.action_history[-5:]
        if len(last_five_history) >= 5:
            if last_five_history == ['NOTHING', 'NOTHING', 'NOTHING', 'NOTHING', 'NOTHING']:
                reward -= 1000
                self.episode_ended = True

        # ===append to state history===
        self.turns_counter += 1
        self.state_history.append(self.state)
        del self.state_history[:1]

        # ===totals===
        self.total_reward += reward

        # ===render image===
        if self.render_step:
            self.halite_image_render.render_board(self.board, self.state, heat_map=heat_map,
                                                  total_reward=self.total_reward, this_step_reward=reward,
                                                  window_name=self.window_name,
                                                  last_action=self._action_def[int_action],
                                                  player_halite=halite_after_turn,
                                                  total_ship_cargo=cargo_after_turn,
                                                  action_history=self.action_history,
                                                  env_name=self._env_name)

        # ===return to engine===
        if self.episode_ended:
            return_object = ts.termination(np.array(self.state_history, dtype=np.float), reward)
            return return_object
        else:
            return_object = ts.transition(np.array(self.state_history, dtype=np.float), reward=reward, discount=1.0)
            return return_object


    def set_rendering(self, enabled=True):
        self.render_step = enabled

    def prime_board(self):
        self.board.players[0].ships[0].next_action = ShipAction.CONVERT
        self.board = self.board.next()
        self.state, heat_map = self.get_state_v2()
        self.state_history.append(self.state)
        del self.state_history[:1]
        self.turns_counter += 1
        self.board.players[0].shipyards[0].next_action = ShipyardAction.SPAWN
        self.board = self.board.next()
        self.state, heat_map = self.get_state_v2()
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
        state_pixels = np.zeros([self._channels, self._board_size, self._board_size])
        for x in range(0, self._board_size):
            for y in range(0, self._board_size):
                cell = self.board[(x, self._board_size - y - 1)]
                cell_halite = 1.0 * cell.halite / float(self.board.configuration.max_cell_halite)
                cell_halite_heat = 255 * cell.halite / float(self.board.configuration.max_cell_halite)
                reward_heatmap[y, x] = cell_halite_heat
                # 0 = Halite
                # 1 = Ship Presence (One Hot 'ship_id', rest 0.5)
                # 2 = Shipyard Presence (One Hot 'ship_id', rest 0.5)
                state_pixels[0, y, x] = cell_halite
                if cell.ship is not None:
                    if cell.ship.player_id == 0:
                        state_pixels[1, y, x] = 1.0
                    else:
                        state_pixels[1, y, x] = 0.5
                elif cell.shipyard is not None:
                    if cell.shipyard.player_id == 0:
                        state_pixels[2, y, x] = 1.0
                    else:
                        state_pixels[2, y, x] = 0.5

        sigma = [0.7, 0.7]
        reward_heatmap = sp.ndimage.filters.gaussian_filter(reward_heatmap, sigma, mode='constant')

        return state_pixels, reward_heatmap