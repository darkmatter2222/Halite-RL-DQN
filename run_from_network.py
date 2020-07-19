import numpy as np
import tensorflow as tf
import time
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from helpers import render
from helpers import data


def human_action(observation, configuration):
    try:
        board = Board(observation, configuration)
        current_player = board.current_player
        for ship in current_player.ships:
            render.renderer(board, ship.position)
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
            n_image = np.asarray([np.asarray(img)])
            y_prob = model.predict(n_image)
            y_class = label_map_actions[np.argmax(y_prob)]
            if y_class != 'NOTHING':
                ship.next_action = y_class
            print(f'Moving:{y_class} w/ cargo:{ship.halite}')
            time.sleep(0.25)

        return current_player.next_actions
    except Exception as e:
        return current_player.next_actions


model = tf.keras.models.load_model('N:\\Halite\\Models\\v2.h5')
root_image_directory = 'N:\\Halite'
board_size = 10
environment = make("halite", configuration={"size": board_size, "startingHalite": 1000})
agent_count = 2
environment.reset(agent_count)

label_map = {0: "EAST", 1: "NORTH", 2: "NOTHING", 3: "SOUTH", 4: "WEST"}
label_map_actions = {0: ShipAction.EAST, 1: ShipAction.NORTH, 2: "NOTHING", 3: ShipAction.SOUTH, 4: ShipAction.WEST}

environment.reset(agent_count)
environment.configuration.actTimeout += 10000
environment.run([human_action, "random"])
# environment.render(mode="ipython", width=800, height=600)

