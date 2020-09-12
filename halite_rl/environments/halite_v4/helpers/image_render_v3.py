import cv2
import uuid
import numpy as np
import math
from skimage import draw
from collections import Counter
import math
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import scipy as sp


class image_render_v3():
    def __init__(self, this_board_size):
        self._board_size = this_board_size
        self.image_name = 'Default'
        self._final_image_dimension = 400
        self._final_stats_dimension = 400
        self._sprite_models = {}
        self._BGR_colors = {
            'blue': (255, 0, 0),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'black': (0, 0, 0)
        }
        self._player_colors = {
            0: 'blue',
            1: 'red',
            2: 'yellow',
            3: 'green'
        }
        self._premade_rendered_sprites = {}

        self.initialize_sprite_models()

    def initialize_sprite_models(self):
        # calculate sprite size
        sprite_size = math.floor(self._final_image_dimension / self._board_size)

        # render ship and shipyard model
        ship_sprite_model = np.zeros([sprite_size, sprite_size])
        shipyard_sprite_model = np.zeros([sprite_size, sprite_size])
        ship_and_shipyard_sprite_model = np.zeros([sprite_size, sprite_size])

        for h in range(sprite_size):
            for w in range(sprite_size):
                if h > ((sprite_size / 3) * 1) and h < ((sprite_size / 3) * 2) and \
                    w > ((sprite_size / 3) * 1) and w < ((sprite_size / 3) * 2):
                    ship_sprite_model[h, w] = 1
                else:
                    shipyard_sprite_model[h, w] = 1
                ship_and_shipyard_sprite_model[h, w] = 1

        self._sprite_models['ship_sprite'] = ship_sprite_model
        self._sprite_models['shipyard_sprite'] = shipyard_sprite_model
        self._sprite_models['ship_and_shipyard_sprite'] = ship_and_shipyard_sprite_model

        # paint the models
        for player_id in range(4):
            # get this player color
            BGR_color = self._BGR_colors[self._player_colors[player_id]]

            # paint ships
            render_ship_sprite = np.zeros([sprite_size, sprite_size, 3])
            for h in range(sprite_size):
                for w in range(sprite_size):
                    if self._sprite_models['ship_sprite'][h, w] == 1:
                        render_ship_sprite[h, w] = BGR_color
                    else:
                        render_ship_sprite[h, w] = [0, 0, 0] # heatmap background

            self._premade_rendered_sprites[f'ship_sprite_player_{player_id}'] = render_ship_sprite

            # paint shipyards
            render_shipyard_sprite = np.zeros([sprite_size, sprite_size, 3])
            for h in range(sprite_size):
                for w in range(sprite_size):
                    if self._sprite_models['shipyard_sprite'][h, w] == 1:
                        render_shipyard_sprite[h, w] = BGR_color
                    else:
                        render_shipyard_sprite[h, w] = [0, 0, 0] # heatmap background

            self._premade_rendered_sprites[f'shipyard_sprite_player_{player_id}'] = render_shipyard_sprite

            # paint shipyard and ship
            render_ship_and_shipyard_sprite = np.zeros([sprite_size, sprite_size, 3])
            for h in range(sprite_size):
                for w in range(sprite_size):
                    if self._sprite_models['ship_and_shipyard_sprite'][h, w] == 1:
                        render_ship_and_shipyard_sprite[h, w] = BGR_color
                    else:
                        render_ship_and_shipyard_sprite[h, w] = [0, 0, 0] # heatmap background

            self._premade_rendered_sprites[f'ship_and_shipyard_sprite_player_{player_id}'] = render_ship_and_shipyard_sprite


        # paint halite
        circle_center = math.floor(sprite_size / 2)
        for s in range(10):
            circle_sprite_model = np.zeros([sprite_size, sprite_size])
            radius = ((sprite_size * (s * 3)) / 100)
            ri, ci = draw.circle(circle_center, circle_center, radius=radius, shape=circle_sprite_model.shape)
            circle_sprite_model[ri, ci] = 1
            self._sprite_models[f'circle_sprite_{s}'] = circle_sprite_model

            render_halite_sprite = np.zeros([sprite_size, sprite_size, 3])
            BGR_color = self._BGR_colors['cyan']
            for h in range(sprite_size):
                for w in range(sprite_size):
                    if self._sprite_models[f'circle_sprite_{s}'][h, w] == 1:
                        render_halite_sprite[h, w] = BGR_color
                    else:
                        render_halite_sprite[h, w] = [0, 0, 0] # heatmap background
            self._premade_rendered_sprites[f'circle_sprite_{s}'] = render_halite_sprite

    def render_board(self, board, state):
        # calculate sprite size
        sprite_size = math.floor(self._final_image_dimension / self._board_size)
        master_image = np.zeros([self._final_image_dimension, self._final_image_dimension, 3], dtype='uint8')
        state_image = np.zeros([self._final_image_dimension, self._final_image_dimension, 3], dtype='uint8')
        stats_image = np.zeros([self._final_stats_dimension, self._final_stats_dimension, 3], dtype='uint8')

        reward_heatmap = np.zeros([self._board_size, self._board_size])

        for height in range(self._board_size):
            for width in range(self._board_size):
                board_cell = board[width, height]
                bord_cell_halite = int(9.0 * board_cell.halite / float(board.configuration.max_cell_halite))
                reward_heatmap[height, width] = bord_cell_halite

        # Apply gaussian filter
        sigma = [0.7, 0.7]
        reward_heatmap = sp.ndimage.filters.gaussian_filter(reward_heatmap, sigma, mode='constant')


        for board_h in range(self._board_size):
            for board_w in range(self._board_size):
                board_y = (self._board_size - board_h - 1)
                board_x = board_w
                board_cell = board[board_x, board_y]
                bord_cell_halite = math.floor(reward_heatmap[board_h, board_w])

                half_sprite_size = math.floor(sprite_size / 2)

                sudo_sprite = np.ndarray([half_sprite_size, half_sprite_size, 3])
                sudo_sprite.fill(state[0, board_h, board_x] * 200)
                state_image[board_h * half_sprite_size:board_h * half_sprite_size + half_sprite_size,
                            board_x * half_sprite_size:board_x * half_sprite_size + half_sprite_size] = \
                            sudo_sprite
                sudo_sprite.fill(state[1, board_h, board_x] * 255)
                state_image[board_h * half_sprite_size + 200:(board_h * half_sprite_size + 200)+ half_sprite_size,
                            board_x * half_sprite_size:board_x * half_sprite_size + half_sprite_size] = \
                            sudo_sprite


                master_image[board_h * sprite_size:board_h * sprite_size + sprite_size,
                board_x * sprite_size:board_x * sprite_size + sprite_size] = \
                    self._premade_rendered_sprites[f'circle_sprite_{bord_cell_halite}']

                if board_cell.shipyard is not None and board_cell.ship is not None:
                    master_image[board_h * sprite_size:board_h * sprite_size + sprite_size,
                    board_x * sprite_size:board_x * sprite_size + sprite_size] =\
                        self._premade_rendered_sprites[f'ship_and_shipyard_sprite_player_{board_cell.shipyard._player_id}']
                elif board_cell.ship is not None:
                    master_image[board_h * sprite_size:board_h * sprite_size + sprite_size,
                    board_x * sprite_size:board_x * sprite_size + sprite_size] =\
                        self._premade_rendered_sprites[f'ship_sprite_player_{board_cell.ship._player_id}']
                elif board_cell.shipyard is not None:
                    master_image[board_h * sprite_size:board_h * sprite_size + sprite_size,
                    board_x * sprite_size:board_x * sprite_size + sprite_size] =\
                        self._premade_rendered_sprites[f'shipyard_sprite_player_{board_cell.shipyard._player_id}']
                else:
                    # nothing
                    lol = 1



        cv2.imshow(f'Real Time Play ', master_image)
        cv2.imshow(f'State ', state_image)

        cv2.waitKey(1)
        # ship
        lol = 1