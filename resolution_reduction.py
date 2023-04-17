import cv2
import os
from os import listdir

# Function for rescale
def rescaleFrame(frame, scale):
  width = int(frame.shape[1] * scale)
  height = int(frame.shape[0] * scale)
  dimensions = (width, height)

  return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


image_names = os.listdir('images_original')

for image_name in image_names:
    print(image_name)

    # Read image
    img = cv2.imread(f'images_original/{image_name}')
    print(img.shape[:2])

    img_resized = rescaleFrame(frame=img, scale=0.5)
    print(img_resized.shape[:2])
    
    # Save the resized images
    cv2.imwrite(f'images_lowres/{image_name[:-5]}_res.jpeg', img_resized)