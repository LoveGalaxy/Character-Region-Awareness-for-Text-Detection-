import os
from PIL import Image
import numpy as np


def load_imgs_dir(dir_path):
    images = os.listdir(dir_path)
    r_images = []
    for image in images:
        r_images.append(os.path.join(dir_path, image))
    return r_images

def read_image(img_path, image_size=(320, 320)):
    image = np.array(Image.open(img_path).resize((image_size), Image.ANTIALIAS)).transpose((2, 0, 1))
    c, h, w = image.shape
    return image.reshape(1, c, h, w)