from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import numpy as np
import database_interface
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import uuid
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import TextBox
import json

database = database_interface.database
rolling_reserve = []
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


def renderer(board, highlight=None):
    size = board.configuration.size
    sudo_board = []
    shift_delta = Point(0, 0)
    if highlight != None:
        shift_delta = Point(4, 4) - highlight

    for y in range(size):
        sudo_board.append(['   '] * 10)

    for y in range(size):
        for x in range(size):
            board_cell = board[(x, size - y - 1)]
            bord_cell_halite = int(9.0 * board_cell.halite / float(board.configuration.max_cell_halite))
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

    for x in range(size):
        sudo_board[x] = shift(sudo_board[x], shift_delta[0])
    sudo_board = shift(sudo_board, shift_delta[1] * -1)

    shifted_result = ''
    for y in range(size):
        for x in range(size):
            shifted_result += '|'
            shifted_result += sudo_board[y][x]
        shifted_result += '|\n'


    """
    The board is printed in a grid with the following rules:
    Capital letters are shipyards
    Lower case letters are ships
    Digits are cell halite and scale from 0-9 directly proportional to a value between 0 and self.configuration.max_cell_halite
    Player 1 is letter a/A
    Player 2 is letter b/B
    etc.
    """
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
    print(shifted_result)


def shift(seq, n=0):
    a = n % len(seq)
    return seq[-a:] + seq[:-a]


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


def get_training_data(source_ship, all_ships, all_shipyards, board_size, player_id , total_halite, cargo, object_type,
                      halite_on_field):
    if object_type == 'ship':
        self_ship, other_ships, friend_shipyards, foe_shipyards, foe_shipyards = calculate_delta(source_ship, all_ships, all_shipyards, board_size, player_id)

    else:
        self_ship = np.zeros([board_size, board_size])
        other_ships = np.zeros([board_size, board_size])
        friend_shipyards = np.zeros([board_size, board_size])
        foe_shipyards = np.zeros([board_size, board_size])

    field = np.zeros([board_size, board_size])
    field = np.array(halite_on_field).reshape((board_size, board_size))
    #field = tf.keras.utils.normalize(field)

    shift_delta = Point(4, 4) - source_ship.position

    pixels = []
    for x in range(0, board_size):
        row = []
        for y in range(0, board_size):
            pixel = [0, 0, 0] #ThisShip, Attract, Avoide
            pixel[1] = int(200.0 * field[x][y] / float(board.configuration.max_cell_halite)) # Attract (Halite) (Max 200 of 255)
            if self_ship[x][y] == 1:
                pixel[0] = 255 # This Ship
            if other_ships[x][y] == 1:
                pixel[2] = 255 # Avoide
                pixel[1] = 0 # Set Attract to NONE
            if foe_shipyards[x][y]:
                pixel[2] = 255 # Avoide
                pixel[1] = 0 # Set Attract to NONE
            row.append(tuple(pixel))
        row = shift(row, shift_delta[0])
        pixels.append(row)
    pixels = shift(pixels, (shift_delta[1] * -1))

    # Convert the pixels into an array using numpy
    array = np.array(pixels, dtype=np.uint8)

    # Use PIL to create an image from the new array of pixels
    new_image = Image.fromarray(array)

    #plt.imshow(new_image)
    #plt.show(block=False)

    #new_image.save('new.png')

    return new_image


def save_image(img, name):
    directory = f'{root_image_directory}\\TRAIN\\{name}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    img.save(f'{directory}\\{uuid.uuid1()}.png')


def human_action(observation, configuration):
    #this_step += 1
    try:
        board = Board(observation, configuration)
        current_player = board.current_player

        for ship in current_player.ships:
            renderer(board, ship.position)
            ship_directive = input(f"What Direction to Move Ship w/ cargo {ship.halite} for this step:{observation.step}?")
            if ship_directive != '':
                ship.next_action = SHIP_DIRECTIVES[ship_directive]
                # clear_output(wait=False)
            img = get_training_data(ship,
                                    board.ships,
                                    board.shipyards,
                                    board_size,
                                    current_player.id,
                                    board.observation["players"][0][0],
                                    ship.halite,
                                    'ship',
                                    board.observation['halite'])
            save_image(img, SHIP_DIRECTIVES[ship_directive])
        return current_player.next_actions
    except Exception as e:
        return current_player.next_actions


environment.reset(agent_count)
environment.configuration.actTimeout += 10000
environment.run([human_action, "random"])
environment.render(mode="ipython", width=800, height=600)