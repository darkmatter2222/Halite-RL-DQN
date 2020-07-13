from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import numpy as np
import database_interface
import json
import database_interface
import numpy as np
import json
import tensorflow as tf
import logging
import time
import os

# Load Model
model = tf.keras.models.load_model('T1.h5')

database = database_interface.database


# Render Override
def renderer(board, highlight=None):
    """
    The board is printed in a grid with the following rules:
    Capital letters are shipyards
    Lower case letters are ships
    Digits are cell halite and scale from 0-9 directly proportional to a value between 0 and self.configuration.max_cell_halite
    Player 1 is letter a/A
    Player 2 is letter b/B
    etc.
    """
    size = board.configuration.size
    result = ''
    for y in range(size):
        for x in range(size):
            cell = board[(x, size - y - 1)]
            result += '|'
            precolor = ''
            postcolor = ''

            if highlight != None:
                if (x, size - y - 1) == highlight:
                    precolor = '\x1b[31m'
                    postcolor = '\x1b[0m'

            result += precolor

            result += (
                chr(ord('a') + cell.ship.player_id)
                if cell.ship is not None
                else ' '
            )
            # This normalizes a value from 0 to max_cell halite to a value from 0 to 9
            normalized_halite = int(9.0 * cell.halite / float(board.configuration.max_cell_halite))
            result += str(normalized_halite)
            result += (
                chr(ord('A') + cell.shipyard.player_id)
                if cell.shipyard is not None
                else ' '
            )
            result += postcolor
        result += '|\n'
    print(result)


# calculate_deltas
def calculate_delta(source_ship, all_ships, all_shipyards, board_size, player_id=0):
    x_deltas_ships = np.zeros([board_size, board_size])
    y_deltas_ships = np.zeros([board_size, board_size])
    x_deltas_friend_shipyards = np.zeros([board_size, board_size])
    y_deltas_friend_shipyards = np.zeros([board_size, board_size])
    x_deltas_foe_shipyards = np.zeros([board_size, board_size])
    y_deltas_foe_shipyards = np.zeros([board_size, board_size])

    # get distance from this ship to all other ships (Friend and Foe)
    for ship in all_ships:
        if ship != source_ship.id:
            distance_to_ship = abs(source_ship.position - all_ships[ship].position)
            this_ship = all_ships[ship]
            x_deltas_ships[board_size - 1 - this_ship.position.y][this_ship.position.x] = distance_to_ship.x
            y_deltas_ships[board_size - 1 - this_ship.position.y][this_ship.position.x] = distance_to_ship.y

    # get distance from this ship to all shipyards (Friend and Foe)
    for shipyard in all_shipyards:
        distance_to_shipyard = abs(source_ship.position - all_shipyards[shipyard].position)
        this_shipyard = all_shipyards[shipyard]
        if all_shipyards[shipyard].player_id == player_id:
            # Do Friend
            x_deltas_friend_shipyards[board_size - 1 - this_shipyard.position.y][this_shipyard.position.x] = distance_to_shipyard.x
            y_deltas_friend_shipyards[board_size - 1 - this_shipyard.position.y][this_shipyard.position.x] = distance_to_shipyard.y
        else:
            # Do foe
            x_deltas_foe_shipyards[board_size - 1 - this_shipyard.position.y][this_shipyard.position.x] = distance_to_shipyard.x
            y_deltas_foe_shipyards[board_size - 1 - this_shipyard.position.y][this_shipyard.position.x] = distance_to_shipyard.y

    return x_deltas_ships, y_deltas_ships, x_deltas_friend_shipyards, y_deltas_friend_shipyards, x_deltas_foe_shipyards, y_deltas_foe_shipyards


def get_training_data(source_ship, all_ships, all_shipyards, board_size, player_id , total_halite, cargo, object_type, halite_on_field):
    relitive_attributes = np.zeros([board_size, board_size])
    relitive_attributes[0][1] = total_halite
    relitive_attributes[0][2] = cargo

    if object_type == 'ship':
        x_deltas_ships, y_deltas_ships, x_deltas_friend_shipyards, y_deltas_friend_shipyards, x_deltas_foe_shipyards, y_deltas_foe_shipyards = calculate_delta(
            source_ship, all_ships, all_shipyards, board_size, player_id)
        relitive_attributes[0][0] = 0 # is a ship
    else:
        x_deltas_ships = np.zeros([board_size, board_size])
        y_deltas_ships = np.zeros([board_size, board_size])
        x_deltas_friend_shipyards = np.zeros([board_size, board_size])
        y_deltas_friend_shipyards = np.zeros([board_size, board_size])
        x_deltas_foe_shipyards = np.zeros([board_size, board_size])
        y_deltas_foe_shipyards = np.zeros([board_size, board_size])
        relitive_attributes[0][0] = 1 # is a shipyard

    field = np.zeros([board_size, board_size])
    field = np.array(halite_on_field).reshape((board_size, board_size))

    return [x_deltas_ships.tolist(), y_deltas_ships.tolist(), x_deltas_friend_shipyards.tolist(), y_deltas_friend_shipyards.tolist(), x_deltas_foe_shipyards.tolist(), y_deltas_foe_shipyards.tolist(), field.tolist(), relitive_attributes.tolist()]


board_size = 10
environment = make("halite", configuration={"size": board_size, "startingHalite": 1000})
agent_count = 2
environment.reset(agent_count)
state = environment.state[0]
board = Board(state.observation, environment.configuration)

SHIP_DIRECTIVES = {'8':ShipAction.NORTH,'6':ShipAction.EAST,'2':ShipAction.SOUTH,'4':ShipAction.WEST, '7':ShipAction.CONVERT}
SHIPYARD_DIRECTIVES = {'9':ShipyardAction.SPAWN}

LABELS = {'8': 'NORTH', '6': 'EAST', '2': 'SOUTH', '4': 'WEST', '7': 'CONVERT', '9': 'SPAWN', '5': 'NOTHING'}
LABELS = {"NORTH": 0, "EAST": 1, "SOUTH": 2, "WEST": 3, "NOTHING": 4}
ACTIONS = {0: ShipAction.NORTH, 1: ShipAction.EAST, 2: ShipAction.SOUTH, 3: ShipAction.WEST }

def human_action(observation, configuration):
    time.sleep(0.1)
    try:
        board = Board(observation, configuration)
        current_player = board.current_player

        for ship in current_player.ships:
            renderer(board, ship.position)
            data = get_training_data(ship, board.ships, board.shipyards, board_size, current_player.id,
                                     board.observation["players"][0][0], ship.halite, 'ship',
                                     board.observation['halite'])
            normalized_data = tf.keras.utils.normalize(data)

            testing_data_array = np.empty([1, 8, 10, 10])
            testing_data_array[0] = normalized_data
            prediction = model.predict(np.array(testing_data_array))

            argmax = np.argmax(prediction)

            if argmax != 4:
                ship.next_action = ACTIONS[argmax]

        return current_player.next_actions
    except Exception as e:
        return current_player.next_actions


environment.reset(agent_count)
environment.configuration.actTimeout += 10000
environment.run([human_action, "random"])
environment.render(mode="ansi", width=800, height=600)
