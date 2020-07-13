from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import numpy as np
import database_interface
import tensorflow as tf
import json

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
    self_ship = np.zeros([board_size, board_size])
    other_ships = np.zeros([board_size, board_size])
    friend_shipyards = np.zeros([board_size, board_size])
    foe_shipyards = np.zeros([board_size, board_size])

    # get distance from this ship to all other ships (Friend and Foe)
    for ship in all_ships:
        this_ship = all_ships[ship]
        if ship != source_ship.id:
            other_ships[board_size - 1 - this_ship.position.y][this_ship.position.x] = 1
        else:
            self_ship[board_size - 1 - this_ship.position.y][this_ship.position.x] = 1

    # get distance from this ship to all shipyards (Friend and Foe)
    for shipyard in all_shipyards:
        this_shipyard = all_shipyards[shipyard]
        if all_shipyards[shipyard].player_id == player_id:
            # Do Friend
            friend_shipyards[board_size - 1 - this_shipyard.position.y][this_shipyard.position.x] = 1
        else:
            # Do foe
            foe_shipyards[board_size - 1 - this_shipyard.position.y][this_shipyard.position.x] = 1

    return self_ship, other_ships, friend_shipyards, friend_shipyards, foe_shipyards


def get_training_data(source_ship, all_ships, all_shipyards, board_size, player_id , total_halite, cargo, object_type, halite_on_field):
    if object_type == 'ship':
        self_ship, other_ships, friend_shipyards, foe_shipyards, foe_shipyards = calculate_delta(source_ship, all_ships, all_shipyards, board_size, player_id)

    else:
        self_ship = np.zeros([board_size, board_size])
        other_ships = np.zeros([board_size, board_size])
        friend_shipyards = np.zeros([board_size, board_size])
        foe_shipyards = np.zeros([board_size, board_size])

    field = np.zeros([board_size, board_size])
    field = np.array(halite_on_field).reshape((board_size, board_size))
    field = tf.keras.utils.normalize(field)

    return [self_ship.tolist(), other_ships.tolist(), friend_shipyards.tolist(), foe_shipyards.tolist(), field.tolist()]


board_size = 10
environment = make("halite", configuration={"size": board_size, "startingHalite": 1000})
agent_count = 2
environment.reset(agent_count)
state = environment.state[0]
board = Board(state.observation, environment.configuration)

SHIP_DIRECTIVES = {'8': ShipAction.NORTH, '6': ShipAction.EAST, '5': ShipAction.SOUTH, '4': ShipAction.WEST, '7': ShipAction.CONVERT}
SHIPYARD_DIRECTIVES = {'9': ShipyardAction.SPAWN}

LABELS = {'8': 'NORTH', '6': 'EAST', '5': 'SOUTH', '4': 'WEST', '7': 'CONVERT', '9': 'SPAWN', '': 'NOTHING'}


def human_action(observation, configuration):
    try:
        board = Board(observation, configuration)
        current_player = board.current_player

        for ship in current_player.ships:
            renderer(board, ship.position)
            ship_directive = input(f"What Direction to Move Ship w/ cargo {ship.halite}?")
            if ship_directive != '':
                ship.next_action = SHIP_DIRECTIVES[ship_directive]
                # clear_output(wait=False)
            data = get_training_data(ship, board.ships, board.shipyards, board_size, current_player.id, board.observation["players"][0][0], ship.halite, 'ship', board.observation['halite'])
            database.post(database, json.dumps(data), LABELS[ship_directive])

        return current_player.next_actions
    except Exception as e:
        return current_player.next_actions


environment.reset(agent_count)
environment.configuration.actTimeout += 10000
environment.run([human_action, "random"])
environment.render(mode="ipython", width=800, height=600)