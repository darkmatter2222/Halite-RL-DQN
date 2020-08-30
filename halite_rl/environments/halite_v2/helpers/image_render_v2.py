import cv2
import uuid
import numpy as np
import math
from skimage import draw
import math
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *

# NOTE, From here on, will be:
# Color (BGR), thank you CV2 for that...
# Height
# Width

class image_render_v2():
    def __init__(self, this_board_size):
        self._board_size = this_board_size
        self.image_name = 'Default'
        self._final_image_dimension = 400
        self._final_image_dimension_extension = 200
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
            for halite_level in range(9):
                render_ship_sprite = np.zeros([3, sprite_size, sprite_size])
                for h in range(sprite_size):
                    for w in range(sprite_size):
                        if self._sprite_models['ship_sprite'][h, w] == 1:
                            render_ship_sprite[:, h, w] = BGR_color
                        else:
                            white = ((halite_level * 255) / 10)
                            render_ship_sprite[:, h, w] = [white, white, white] # heatmap background

                self._premade_rendered_sprites[f'ship_sprite_player_{player_id}_h_{halite_level}'] = render_ship_sprite

            # paint shipyards
            for halite_level in range(9):
                render_shipyard_sprite = np.zeros([3, sprite_size, sprite_size])
                for h in range(sprite_size):
                    for w in range(sprite_size):
                        if self._sprite_models['shipyard_sprite'][h, w] == 1:
                            render_shipyard_sprite[:, h, w] = BGR_color
                        else:
                            white = ((halite_level * 255) / 10)
                            render_shipyard_sprite[:, h, w] = [white, white, white] # heatmap background

                self._premade_rendered_sprites[f'shipyard_sprite_player_{player_id}_h_{halite_level}'] = render_shipyard_sprite

            # paint shipyard and ship
            for halite_level in range(9):
                render_ship_and_shipyard_sprite = np.zeros([3, sprite_size, sprite_size])
                for h in range(sprite_size):
                    for w in range(sprite_size):
                        if self._sprite_models['ship_and_shipyard_sprite'][h, w] == 1:
                            render_ship_and_shipyard_sprite[:, h, w] = BGR_color
                        else:
                            white = ((halite_level * 255) / 10)
                            render_ship_and_shipyard_sprite[:, h, w] = [white, white, white] # heatmap background

                self._premade_rendered_sprites[f'ship_and_shipyard_sprite_player_{player_id}_h_{halite_level}'] = render_ship_and_shipyard_sprite


        # paint halite
        for halite_level in range(9):
            circle_center = math.floor(sprite_size / 2)
            for s in range(10):
                circle_sprite_model = np.zeros([sprite_size, sprite_size])
                radius = ((sprite_size * (s * 3)) / 100)
                ri, ci = draw.circle(circle_center, circle_center, radius=radius, shape=circle_sprite_model.shape)
                circle_sprite_model[ri, ci] = 1
                self._sprite_models[f'circle_sprite_{s}'] = circle_sprite_model

                render_halite_sprite = np.zeros([3, sprite_size, sprite_size])
                BGR_color = self._BGR_colors['cyan']
                for h in range(sprite_size):
                    for w in range(sprite_size):
                        if self._sprite_models[f'circle_sprite_{s}'][h, w] == 1:
                            render_halite_sprite[:, h, w] = BGR_color
                        else:
                            white = ((halite_level * 255) / 10)
                            render_halite_sprite[:, h, w] = [white, white, white] # heatmap background
                self._premade_rendered_sprites[f'circle_sprite_{s}_h_{halite_level}'] = render_halite_sprite

    def render_board(self, board, state, total_reward, this_step_reward):
        # calculate sprite size
        sprite_size = math.floor(self._final_image_dimension / self._board_size)
        master_image = np.zeros([self._final_image_dimension, self._final_image_dimension, 3])
        master_image_extension = np.zeros([self._final_image_dimension_extension, self._final_image_dimension,  3])

        for board_h in range(self._board_size):
            for board_w in range(self._board_size):
                board_y = (self._board_size - board_h - 1)
                board_x = board_w
                board_cell = board[board_x, board_y]
                bord_cell_halite = int(9.0 * board_cell.halite / float(board.configuration.max_cell_halite))
                heat = state[board_h, board_w, 3]
                premade_halite_cell = self._premade_rendered_sprites[f'circle_sprite_{bord_cell_halite}']
                lol = premade_halite_cell[:, :, 1]
                premade_halite_cell[premade_halite_cell[:, :, 1] == 0] = heat
                halite_heat_rendered_cell = premade_halite_cell



                master_image[board_h * sprite_size:board_h * sprite_size + sprite_size,
                board_x * sprite_size:board_x * sprite_size + sprite_size] = \
                    halite_heat_rendered_cell

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

        master_image = np.append(master_image, master_image_extension, axis=0)
        cv2.putText(master_image, f'Total Reward: {total_reward}', (10, self._final_image_dimension + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 1), 2)
        cv2.putText(master_image, f'This Step Reward: {this_step_reward}', (10, self._final_image_dimension + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 1), 2)
        cv2.imshow('Real Time Play', master_image)
        cv2.waitKey(1)
        # ship
        lol = 1