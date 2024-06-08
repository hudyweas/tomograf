import cv2
import numpy as np

def pad_image(image, padding_size, border_color=(0, 0, 0)):
    padded_image = cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size,
                                       cv2.BORDER_CONSTANT, value=border_color)
    return padded_image

def image_center(image):
    return np.floor(np.array(image.shape) / 2).astype(int)

def normalize_image(image):
    normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return normalized_image

def resize_to_square(image):
    width, height = image.shape
    size = max(width, height)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
