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
import sklearn
import cv2
import uuid
import matplotlib
from halite_rl.environments.halite_v4.helpers.image_render_v3 import image_render_v3
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
        self._board_size = 25
        self._max_turns = 400
        self._network_frame_depth = 1

        if self._max_turns > self._network_frame_depth:
            self._frames = self._network_frame_depth
        else:
            self._frames = self._max_turns

        self._agent_count = 2
        self._channels = 2
        # attract Target /w heatmap - avoid Target w/ heatmap
        # self

        self._action_def = {0: ShipAction.EAST,
                            1: ShipAction.NORTH,
                            2: "NOTHING",
                            3: ShipAction.SOUTH,
                            4: ShipAction.WEST}

        self.render_step = render_me
        self._env_name = env_name
        self._max_groth_step = 1000000

        self.station_to_ship = {}

        # runtime parameters
        self.turns_counter = 0
        self.episode_ended = False
        self.total_reward = 0
        self.ship_directive = {}  # ship id: {target:T, Action At Target:A}
        self.action_history = []
        self.env_step_count = 0

        # initialize game
        self.environment = make("halite", configuration={"size": self._board_size, "startingHalite": 1000,
                                                         "episodeSteps": self._max_turns})
        self.environment.reset(self._agent_count)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(self._action_def) - 1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._frames, self._channels, self._board_size, self._board_size), dtype=np.float, minimum=0.0,
            maximum=1.0, name='observation')

        self.state = np.zeros([self._channels, self._board_size, self._board_size])

        self.state_history = [self.state] * self._frames

        # get board
        self.board = self.get_board()
        self.prime_board()
        self.halite_image_render = image_render_v3(self._board_size)
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
        self.ship_directive = {}
        # initialize game
        self.environment = make("halite", configuration={"size": self._board_size, "startingHalite": 1000,
                                                         "episodeSteps": self._max_turns})
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
        reward = 0
        self.env_step_count += 1

        # ===pick targets===
        # modes
        # -parking
        # -parked

        if not '2-1' in self.ship_directive:
            if self.env_step_count >= self._max_groth_step:
                max_range = self._board_size - 1
            else:
                max_range = int((self._board_size - 1 * self.env_step_count) / self._max_groth_step) + 1

            if max_range >= self._board_size:
                max_range = self._board_size - 1

            rand_y = random.randrange(0, max_range)
            rand_x = random.randrange(0, max_range)

            target_y = self.board.players[0].shipyards[0].position.y + rand_y
            target_x = self.board.players[0].shipyards[0].position.x + rand_x

            if target_y >= self._board_size:
                target_y = target_y - self._board_size

            if target_x >= self._board_size:
                target_x = target_x - self._board_size

            self.ship_directive['2-1'] = {
                'mode': 'parking',
                'target': (target_y, target_x),  # south
                'action_at_target': 'park'
            }

        # ===take action===
        if not self._action_def[int_action] == 'NOTHING':
            self.board.ships['2-1'].next_action = self._action_def[int_action]

        # ===move random bots===
        random_agent(self.board, self.board.players[1])

        # ===perform move===
        self.board = self.board.next()
        self.state = self.get_state_v2()

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

        # ===calculate reward===
        if not self.episode_ended:
            pos = self.board.ships['2-1'].position
            reward += self.state[0, self._board_size - pos.y - 1, pos.x]

        # ===append to state history===
        self.turns_counter += 1
        self.state_history.append(self.state)
        del self.state_history[:1]

        # ===render image===
        if self.render_step:
            self.halite_image_render.render_board(self.board, self.state)

        # ===return to engine===
        if self.episode_ended:
            return_object = ts.termination(np.array(self.state_history, dtype=np.float), reward)
            return return_object
        else:
            return_object = ts.transition(np.array(self.state_history, dtype=np.float), reward=reward, discount=1.0)
            return return_object

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

        self.station_to_ship[self.board.players[0].shipyards[0].id] = self.board.players[0].ships[0].id

        self.board.players[0].ships[0].next_action = ShipAction.NORTH
        self.board = self.board.next()
        self.state = self.get_state_v2()
        self.state_history.append(self.state)
        del self.state_history[:1]
        self.turns_counter += 1
        self.board.players[0].ships[0].next_action = ShipAction.NORTH
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

    def scale(self, X, x_min, x_max):
        nom = (X - X.min(axis=0)) * (x_max - x_min)
        denom = X.max(axis=0) - X.min(axis=0)
        denom[denom == 0] = 1
        return x_min + nom / denom

    def get_state_v2(self, ship_in_question_id='2-1'):
        # this method, we are constructing both the board to be rendered and what is provided to the neural network.
        attract_heatmap = np.zeros([self._board_size, self._board_size])
        detract_heatmap = np.zeros([self._board_size, self._board_size])
        self_location = np.zeros([self._board_size, self._board_size])
        state = np.zeros([self._channels, self._board_size, self._board_size])

        attract_heatmap_topoff_location = None
        detract_heatmap_topoff_location = []
        player_shipyard_temp_topoff = []

        for ship_id in self.board.ships:
            ship = self.board.ships[ship_id]
            if not ship.player_id == 0:
                detract_heatmap[self._board_size - ship.position.y - 1, ship.position.x] = self._board_size * 10
                detract_heatmap_topoff_location.append((self._board_size - ship.position.y - 1, ship.position.x))
            if ship.id == ship_in_question_id:
                self_location[self._board_size - ship.position.y - 1, ship.position.x] = 1.0
            if ship_id == ship_in_question_id and ship_in_question_id in self.ship_directive:
                if self.ship_directive[ship_in_question_id]['mode'][-3:] == 'ing':
                    target = (self._board_size - self.ship_directive[ship_in_question_id]['target'][0] - 1,
                                     self.ship_directive[ship_in_question_id]['target'][1])
                    attract_heatmap[target] = self._board_size * 15
                    attract_heatmap_topoff_location = target
        for shipyard_id in self.board.shipyards:
            shipyard = self.board.shipyards[shipyard_id]
            if shipyard.player_id == 0:
                dont_care_for_now = 1  # TODO do something with this, at this point, meaningless
                player_shipyard_temp_topoff.append((self._board_size - shipyard.position.y - 1, shipyard.position.x))
            else:
                detract_heatmap[self._board_size - shipyard.position.y - 1, shipyard.position.x] = self._board_size * 10

        attract_sigma = [self._board_size/10, self._board_size/10]
        attract_heatmap = sp.ndimage.filters.gaussian_filter(attract_heatmap, attract_sigma, mode='constant')
        detract_sigma = [self._board_size/20, self._board_size/20]
        detract_heatmap = sp.ndimage.filters.gaussian_filter(detract_heatmap, detract_sigma, mode='constant')
        #if not attract_heatmap_topoff_location == None:
            #attract_heatmap[attract_heatmap_topoff_location] = self._board_size
        #for _ in detract_heatmap_topoff_location:
            #detract_heatmap[_] = self._board_size
        #for _ in player_shipyard_temp_topoff:
            #attract_heatmap[_] = self._board_size
        navigation_map = attract_heatmap - detract_heatmap

        min = np.min(navigation_map)
        delta_to_zero = abs(0 - min)
        navigation_map = navigation_map + delta_to_zero
        max = np.max(navigation_map)

        if max > 0:
            navigation_map = ((navigation_map * 1) / max)

        state[0] = navigation_map
        state[1] = self_location

        return state
