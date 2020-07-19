from kaggle_environments.envs.halite.helpers import *
from helpers import data
import os


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
        sudo_board[x] = data.shift(sudo_board[x], shift_delta[0])
    sudo_board = data.shift(sudo_board, shift_delta[1] * -1)

    shifted_result = ''
    for y in range(size):
        for x in range(size):
            shifted_result += '|'
            shifted_result += sudo_board[y][x]
        shifted_result += '|\n'

    os.system('cls' if os.name == 'nt' else 'clear')
    print(shifted_result)