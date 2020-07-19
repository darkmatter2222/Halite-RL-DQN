import numpy as np
from kaggle_environments.envs.halite.helpers import *
from helpers import data
from PIL import Image


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


def get_training_data(board, source_ship, all_ships, all_shipyards, board_size, player_id , total_halite, cargo, object_type,
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
        row = data.shift(row, shift_delta[0])
        pixels.append(row)
    pixels = data.shift(pixels, (shift_delta[1] * -1))

    # Convert the pixels into an array using numpy
    array = np.array(pixels, dtype=np.uint8)

    # Use PIL to create an image from the new array of pixels
    new_image = Image.fromarray(array)

    #plt.imshow(new_image)
    #plt.show(block=False)

    #new_image.save('new.png')

    return new_image