import numpy as np
from kaggle_environments.envs.halite.helpers import *
import math



def origional(board):
    c = 0
    size = board.configuration.size
    result = ''
    for y in range(size):
        for x in range(size):
            cell = board[(x, size - y - 1)]
            result += '|'
            result += (
                chr(ord('a') + cell.ship.player_id)
                if cell.ship is not None
                else ' '
            )
            # This normalizes a value from 0 to max_cell halite to a value from 0 to 9
            normalized_halite = c
            result += str(normalized_halite)
            result += (
                chr(ord('A') + cell.shipyard.player_id)
                if cell.shipyard is not None
                else ' '
            )
            c+=1
        result += '|\n'
    return result



def shift(seq, n=0):
    a = n % len(seq)
    return seq[-a:] + seq[:-a]


def render_field_matrix(board):
    """
    y0,x0________ Y+
        |
        |
        |
        X+
    """
    field_matrix = []
    size = board.configuration.size
    for y in range(size):
        row = []
        for x in range(size):
            cell = board[(x, size - y - 1)]
            row.append(cell)
        field_matrix.append(row)
    return field_matrix


def center_field_matrix(board, field_matrix, ship_id):
    size = board.configuration.size
    mid_point = math.floor(size / 2)

    ship_point = None
    for y in range(size):
        for x in range(size):
            if field_matrix[x][y].ship_id is not None:
                if field_matrix[x][y].ship_id == ship_id:
                    ship_point = (x, y)
                    break
        if ship_point is not None:
            break

    shift_delta = Point(mid_point, mid_point) - Point(ship_point[0], ship_point[1])

    for x in range(size):
        field_matrix[x] = shift(field_matrix[x], shift_delta[1])
    field_matrix = shift(field_matrix, shift_delta[0] * -1)

    return field_matrix


def render_field_ASCII(board, field_matrix, highlight_center = True):
    size = board.configuration.size
    result = ''
    for y in range(size):
        for x in range(size):
            result += '|'
            cell = field_matrix[y][x]
            cell_halite = int(9.0 * cell.halite / float(board.configuration.max_cell_halite))
            precolor = ''
            postcolor = ''

            if cell_halite > 0:
                precolor = '\x1b[32m'
                postcolor = '\x1b[0m'

            if cell.ship is not None:
                precolor = '\x1b[31m'
                postcolor = '\x1b[0m'
            if cell.shipyard is not None:
                precolor = '\x1b[31m'
                postcolor = '\x1b[0m'

            if highlight_center != None:
                if (x, y ) == (math.floor(size / 2), math.floor(size / 2)):
                    precolor = '\x1b[36m'
                    postcolor = '\x1b[0m'

            sudo_cell = ''
            sudo_cell += precolor

            if cell.ship is not None:
                sudo_cell += chr(ord('a') + cell.ship.player_id)
            else:
                sudo_cell += ' '

            sudo_cell += str(cell_halite)

            if cell.shipyard is not None:
                sudo_cell += chr(ord('A') + cell.shipyard.player_id)
            else:
                sudo_cell += ' '

            sudo_cell += postcolor

            result += str(sudo_cell)
        result += '|\n'
    return result