import numpy as np
import cv2
import time
import os

from Extensions import api

image_path = os.path.join('test_data', 'traffic-sign-to-test.jpg')

img, spatial_dimension = api.reading_image(image_path)

