import copy
import json
import math
import numpy as np
from os import path
from random import choice, randint, randrange, sample, seed
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from kaggle_environments import utils

def random_agent(board, player):
    me = player
    remaining_halite = me.halite
    ships = me.ships
    # randomize ship order
    ships = sample(ships, len(ships))
    for ship in ships:
        if ship.cell.halite > ship.halite and randint(0, 1) == 0:
            # 50% chance to mine
            continue
        if ship.cell.shipyard is None and remaining_halite > board.configuration.convert_cost:
            # 5% chance to convert at any time
            if randint(0, 19) == 0:
                remaining_halite -= board.configuration.convert_cost
                ship.next_action = ShipAction.CONVERT
                continue
            # 50% chance to convert if there are no shipyards
            if randint(0, 1) == 0 and len(me.shipyards) == 0:
                remaining_halite -= board.configuration.convert_cost
                ship.next_action = ShipAction.CONVERT
                continue
        # None represents the chance to do nothing
        ship.next_action = choice(ShipAction.moves())
    shipyards = me.shipyards
    # randomize shipyard order
    shipyards = sample(shipyards, len(shipyards))
    ship_count = len(player.ships)
    for shipyard in shipyards:
        # If there are no ships, always spawn if possible
        if ship_count == 0 and remaining_halite > board.configuration.spawn_cost:
            remaining_halite -= board.configuration.spawn_cost
            shipyard.next_action = ShipyardAction.SPAWN
        # 20% chance to spawn if no ships
        elif randint(0, 4) == 0 and remaining_halite > board.configuration.spawn_cost:
            remaining_halite -= board.configuration.spawn_cost
            shipyard.next_action = ShipyardAction.SPAWN