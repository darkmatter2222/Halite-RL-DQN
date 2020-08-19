import cv2
import uuid
import numpy as np
import math
from skimage import draw


class HaliteImageRender():
    def __init__(self, this_board_size):
        self._board_size = this_board_size
        self.image_name = 'Default'
        self._final_image_dimension = 1000
        self.sprite_models = {}
        self.BGR_colors = {
            'blue': (255, 0, 0),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255)
        }
        self.initialize_sprite_models()

    def initialize_sprite_models(self):
        # calculate sprite size
        sprite_size = math.floor(self._final_image_dimension / self._board_size)

        # render ship and shipyard model
        ship_sprite_model = np.zeros([sprite_size, sprite_size])
        shipyard_sprite_model = np.zeros([sprite_size, sprite_size])
        circle_sprite_model = np.zeros([sprite_size, sprite_size])

        for h in range(sprite_size):
            for w in range(sprite_size):
                if h > ((sprite_size / 3) * 1) and h < ((sprite_size / 3) * 2) and \
                    w > ((sprite_size / 3) * 1) and w < ((sprite_size / 3) * 2):
                    ship_sprite_model[h, w] = 1
                else:
                    shipyard_sprite_model[h, w] = 1

        self.sprite_models['ship_sprite'] = ship_sprite_model
        self.sprite_models['shipyard_sprite'] = shipyard_sprite_model

        circle_center = math.floor(sprite_size / 2)
        for s in range(math.floor(sprite_size/10)):
            radius = math.floor(s * 2)
            ri, ci = draw.circle(circle_center, circle_center, radius=radius, shape=circle_sprite_model.shape)
            circle_sprite_model[ri, ci] = 1
            self.sprite_models[f'circle_sprite_{s}'] = circle_sprite_model

    def render_board(self):
        



        # ship
        lol = 1
