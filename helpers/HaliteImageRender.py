import cv2
import uuid
import numpy as np
import math
from skimage import draw
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *


class HaliteImageRender():
    def __init__(self, this_board_size):
        self._board_size = this_board_size
        self.image_name = 'Default'
        self._final_image_dimension = 200
        self.sprite_models = {}
        self.BGR_colors = {
            'blue': (255, 0, 0),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'black': (0, 0, 0)
        }
        self.player_colors = {
            0: 'blue',
            1: 'red',
            2: 'yellow',
            3: 'green'
        }
        self.premade_rendered_sprites = {}

        self.initialize_sprite_models()

    def initialize_sprite_models(self):
        # calculate sprite size
        sprite_size = math.floor(self._final_image_dimension / self._board_size)

        # render ship and shipyard model
        ship_sprite_model = np.zeros([sprite_size, sprite_size])
        shipyard_sprite_model = np.zeros([sprite_size, sprite_size])

        for h in range(sprite_size):
            for w in range(sprite_size):
                if h > ((sprite_size / 3) * 1) and h < ((sprite_size / 3) * 2) and \
                    w > ((sprite_size / 3) * 1) and w < ((sprite_size / 3) * 2):
                    ship_sprite_model[h, w] = 1
                else:
                    shipyard_sprite_model[h, w] = 1

        self.sprite_models['ship_sprite'] = ship_sprite_model
        self.sprite_models['shipyard_sprite'] = shipyard_sprite_model

        BGR_color = self.BGR_colors['blue']
        render_ship_sprite = np.zeros([sprite_size, sprite_size, 3])
        for h in range(sprite_size):
            for w in range(sprite_size):
                if self.sprite_models['ship_sprite'][h, w] == 1:
                    render_ship_sprite[h, w] = BGR_color
                else:
                    render_ship_sprite[h, w] = self.BGR_colors['black']

        self.premade_rendered_sprites[f'ship_sprite_player_{0}'] = render_ship_sprite

        circle_center = math.floor(sprite_size / 2)
        for s in range(10):
            circle_sprite_model = np.zeros([sprite_size, sprite_size])
            radius = ((sprite_size * (s * 3)) / 100)
            ri, ci = draw.circle(circle_center, circle_center, radius=radius, shape=circle_sprite_model.shape)
            circle_sprite_model[ri, ci] = 1
            self.sprite_models[f'circle_sprite_{s}'] = circle_sprite_model

            render_halite_sprite = np.zeros([sprite_size, sprite_size, 3])
            BGR_color = self.BGR_colors['cyan']
            for h in range(sprite_size):
                for w in range(sprite_size):
                    if self.sprite_models[f'circle_sprite_{s}'][h, w] == 1:
                        render_halite_sprite[h, w] = BGR_color
                    else:
                        render_halite_sprite[h, w] = self.BGR_colors['black']
            self.premade_rendered_sprites[f'circle_sprite_{s}'] = render_halite_sprite

    def render_board(self, board):
        # calculate sprite size
        sprite_size = math.floor(self._final_image_dimension / self._board_size)
        master_image = np.zeros([self._final_image_dimension, self._final_image_dimension, 3])

        for board_h in range(self._board_size):
            for board_w in range(self._board_size):
                board_y = (self._board_size - board_h - 1)
                board_x = board_w
                board_cell = board[board_x, board_y]
                bord_cell_halite = int(9.0 * board_cell.halite / float(board.configuration.max_cell_halite))

                master_image[board_h * sprite_size:board_h * sprite_size + sprite_size,
                board_x * sprite_size:board_x * sprite_size + sprite_size] = \
                    self.premade_rendered_sprites[f'circle_sprite_{bord_cell_halite}']

                if board_cell.ship is not None:
                    master_image[board_h * sprite_size:board_h * sprite_size + sprite_size,
                    board_x * sprite_size:board_x * sprite_size + sprite_size] =\
                        self.premade_rendered_sprites[f'ship_sprite_player_{0}']
        # TODO Code Shipyards
        cv2.imshow('Real Time Play', master_image)
        cv2.waitKey(1)
        # ship
        lol = 1
