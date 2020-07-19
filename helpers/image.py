import uuid
import os


def save_image(img, name, root_image_directory):
    directory = f'{root_image_directory}\\TRAIN\\{name}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    img.save(f'{directory}\\{uuid.uuid1()}.png')
