import cv2
import uuid

class HaliteImageRender(bord_size = 10):
    def __init__(self):
        self.image_name = 'Default'
        self.final_image_dimension = 1000
        self.sprites = {}
        self.BGR_colors = {
            'blue':(255, 0, 0),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255)
        }

    def initialize_image_shapes(self):
        #
