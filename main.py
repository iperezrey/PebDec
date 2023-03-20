import os
import cv2

# Import splitter
from splitter import splitter

image_names = os.listdir('images_original')
for image_name in image_names:
    
    splitter(image_name=image_name)
