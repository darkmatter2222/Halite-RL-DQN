from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from helpers import render
from helpers import data
from helpers import image
import numpy as np
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import uuid
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import TextBox
import json


root_image_directory = 'N:\\Halite'
board_size = 10
environment = make("halite", configuration={"size": board_size, "startingHalite": 1000})
agent_count = 2
environment.reset(agent_count)
state = environment.state[0]
board = Board(state.observation, environment.configuration)
SHIP_DIRECTIVES = {'w': ShipAction.NORTH,
                   'd': ShipAction.EAST,
                   's': ShipAction.SOUTH,
                   'a': ShipAction.WEST,
                   '7': ShipAction.CONVERT,
                   '': 'NOTHING'}
SHIPYARD_DIRECTIVES = {'9': ShipyardAction.SPAWN}
LABELS = {'8': 'NORTH', '6': 'EAST', '5': 'SOUTH', '4': 'WEST', '7': 'CONVERT', '9': 'SPAWN', '': 'NOTHING'}

def human_action(observation, configuration):
    #this_step += 1
    try:
        board = Board(observation, configuration)
        current_player = board.current_player

        for ship in current_player.ships:
            render.renderer(board, ship.position)
            ship_directive = input(f"What Direction to Move Ship w/ cargo {ship.halite} for this step:{observation.step}?")
            if ship_directive != '':
                ship.next_action = SHIP_DIRECTIVES[ship_directive]
                # clear_output(wait=False)
            img = data.get_training_data(board,
                                    ship,
                                    board.ships,
                                    board.shipyards,
                                    board_size,
                                    current_player.id,
                                    board.observation["players"][0][0],
                                    ship.halite,
                                    'ship',
                                    board.observation['halite'])
            image.save_image(img, SHIP_DIRECTIVES[ship_directive])
        return current_player.next_actions
    except Exception as e:
        return current_player.next_actions


environment.reset(agent_count)
environment.configuration.actTimeout += 10000
environment.run([human_action, "random"])
environment.render(mode="ipython", width=800, height=600)