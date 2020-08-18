import os
import math
import random
import numpy as np
import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.environments import gym_wrapper
from tf_agents.environments.wrappers import ActionDiscretizeWrapper
from collections import OrderedDict

from gym import Env, spaces
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from kaggle_environments import make, evaluate
from kaggle_environments.envs.halite.helpers import Board, Point

from tqdm.notebook import tqdm

from matplotlib import pyplot as plt

N_CPU = os.cpu_count()
print('CPU Cores =', N_CPU)

LOG_DIR = './log/'
AGENT_DIR = '../input/swarm-intelligence-with-sdk/'

os.makedirs(LOG_DIR, exist_ok=True)

agents = {}

for infile, outfile in agents.items():
    with open(os.path.join(AGENT_DIR, infile), 'rt') as f:
        agent_src = f.read()

    with open(outfile, 'wt') as f:
        f.write(agent_src)

# agents = ['random', 'idle.py', 'beetle.py', 'duo.py' 'swarm.py', 'attack.py']

# GAME_CONFIG = {'episodeSteps': 400, 'size': 21, 'num_agents': 4}
GAME_CONFIG = {'episodeSteps': 50, 'size': 10, 'num_agents': 2}

# num_agents = GAME_CONFIG['num_agents']
# position = random.randint(0, num_agents-1)
# GAME_AGENTS = random.sample(agents, num_agents-1)
# GAME_AGENTS.insert(position, None)
GAME_AGENTS = ['idle.py', None]


def sort_cells(cells):
    ordered_cells = OrderedDict()
    size = int(len(cells) ** 0.5)

    for x in range(size):
        for y in range(size):
            point = Point(x, y)
            ordered_cells[point] = cells[point]
    return ordered_cells


SHIP_ACTIONS = [None, 'CONVERT', 'NORTH', 'EAST', 'SOUTH', 'WEST']
YARD_ACTIONS = [None, 'SPAWN']

N_SHIP_ACTIONS = len(SHIP_ACTIONS)
N_YARD_ACTIONS = len(YARD_ACTIONS)

MAX_SHIPS = 5
MAX_YARDS = 5


def transform_actions(actions, obs, config):
    next_actions = dict()

    board = Board(obs, config)
    me = board.current_player

    board_cells = sort_cells(board.cells)

    si = 0
    yi = MAX_SHIPS

    for _, c in board_cells.items():
        if c.ship in me.ships and si < MAX_SHIPS:
            i = actions[si]
            ship_action = SHIP_ACTIONS[i]
            si += 1

            if ship_action is not None:
                next_actions[c.ship.id] = ship_action

        if c.shipyard in me.shipyards and yi < MAX_SHIPS + MAX_YARDS:
            i = actions[yi]
            yard_action = YARD_ACTIONS[i]
            yi += 1

            if yard_action is not None:
                next_actions[c.shipyard.id] = yard_action

    return next_actions


N_FEATURES = 8
MAX_SHIP_HALITE = 1000


def transform_observation(obs, config):
    board = Board(obs, config)
    me = board.current_player

    board_cells = sort_cells(board.cells)

    step = []
    cell_yield = []
    me_yard = []
    me_ship = []
    me_ship_cargo = []
    opp_yard = []
    opp_ship = []
    opp_ship_cargo = []

    for _, c in board_cells.items():
        step.append(obs['step'] / config.episodeSteps)

        cell_yield.append(c.halite / config.maxCellHalite)

        if c.ship is None:
            me_ship.append(0)
            me_ship_cargo.append(0)
            opp_ship.append(0)
            opp_ship_cargo.append(0)

        elif c.ship in me.ships:
            me_ship.append(1)
            me_ship_cargo.append(c.ship.halite / MAX_SHIP_HALITE)
            opp_ship.append(0)
            opp_ship_cargo.append(0)

        else:
            me_ship.append(0)
            me_ship_cargo.append(0)
            opp_ship.append(1)
            opp_ship_cargo.append(c.ship.halite / MAX_SHIP_HALITE)

        if c.shipyard is None:
            me_yard.append(0)
            opp_yard.append(0)

        elif c.shipyard in me.shipyards:
            me_yard.append(1)
            opp_yard.append(0)

        else:
            me_yard.append(0)
            opp_yard.append(1)

    x_obs = np.vstack((step,
                       cell_yield,
                       me_yard,
                       me_ship,
                       me_ship_cargo,
                       opp_yard,
                       opp_ship,
                       opp_ship_cargo))

    x_obs = x_obs.reshape(config.size, config.size, N_FEATURES)
    x_obs = x_obs.astype(np.float32).clip(0, 1)

    return x_obs


REWARD_WON = GAME_CONFIG['episodeSteps']
REWARD_LOST = -REWARD_WON

MAX_DELTA = 1000


def transform_reward(done, last_obs, obs, config):
    board = Board(obs, config)
    me = board.current_player

    nships = len(me.ships)
    nyards = len(me.shipyards)
    halite = me.halite
    cargo = sum(s.halite for s in me.ships)

    if nships == 0:
        if nyards == 0:
            return REWARD_LOST

        if halite < config.spawnCost:
            return REWARD_LOST

    if done:
        scores = [p.halite for p in board.players.values() if
                  len(p.ships) > 0 or
                  (len(p.shipyards) > 0 and p.halite >= config.spawnCost)]

        if halite == max(scores):
            if scores.count(halite) == 1:
                return REWARD_WON
        return REWARD_LOST

    delta = 0

    if last_obs is not None:
        last_board = Board(last_obs, config)
        last_me = last_board.current_player

        last_nships = len(last_me.ships)
        last_nyards = len(last_me.shipyards)
        last_halite = last_me.halite
        last_cargo = sum(s.halite for s in last_me.ships)

        delta_ships = (nships - last_nships) * config.spawnCost
        delta_yards = (nyards - last_nyards) * (config.convertCost + config.spawnCost)
        delta_halite = halite - last_halite
        delta_cargo = cargo - last_cargo

        delta = delta_ships + delta_yards + delta_halite + delta_cargo

        if nyards == 0:
            delta -= config.convertCost

        if nships == 0:
            delta -= config.spawnCost

        delta = float(np.clip(delta / MAX_DELTA, -1, 1))

    reward = delta + 0.01
    return reward

def get_actions(model, obs, config, deterministic=False):
    x_obs = transform_observation(obs, config)
    actions, state = model.predict(x_obs, deterministic=deterministic)
    next_actions = transform_actions(actions, obs, config)
    return next_actions


class HaliteGym(py_environment.PyEnvironment):
    def __init__(self):
        halite_env = make('halite', configuration=GAME_CONFIG, debug=True)
        self.env = halite_env.train(GAME_AGENTS)

        self.config = halite_env.configuration



        self.action_space = spaces.MultiDiscrete([N_SHIP_ACTIONS] * MAX_SHIPS +
                                                 [N_YARD_ACTIONS] * MAX_YARDS)
        self.action_space = gym_wrapper.spec_from_gym_space(space=self.action_space, name='action')

        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.config.size,
                                                   self.config.size,
                                                   N_FEATURES),
                                            dtype=np.float32)
        self.observation_space = gym_wrapper.spec_from_gym_space(space=self.observation_space, name='observation')

        self.observation_space = array_spec.BoundedArraySpec(
            shape=(self.config.size, self.config.size, N_FEATURES), dtype=np.int32, minimum=0,
            maximum=1, name='observation')


        self.reward_range = (REWARD_LOST, REWARD_WON)

        self.obs = None
        self.last_obs = None

        self.spec = None
        self.metadata = None

    def action_spec(self):
        return_object = self.action_space
        return return_object

    def observation_spec(self):
        return_object = self.observation_space
        return return_object

    def _reset(self):
        self.last_obs = None
        self.obs = self.env.reset()
        x_obs = transform_observation(self.obs, self.config)
        x_obs = ts.restart(np.array(x_obs, dtype=np.int32))
        return x_obs

    def _step(self, actions):
        next_actions = transform_actions(actions, self.obs, self.config)

        self.last_obs = self.obs
        self.obs, reward, done, info = self.env.step(next_actions)

        x_obs = transform_observation(self.obs, self.config)
        x_reward = transform_reward(done, self.last_obs, self.obs, self.config)


        # final
        if x_reward <= REWARD_LOST:
            done, info = True, {}
            return_object = ts.termination(np.array(x_obs, dtype=np.int32), x_reward)
            return return_object
        else:
            return_object = ts.transition(np.array(x_obs, dtype=np.int32), reward=x_reward, discount=1.0)
            return return_object
