import os
import cv2

# Get the names of all images in the directory
image_names = os.listdir('images_original')

for image_name in image_names:
    print(image_name)

    # Read image
    img = cv2.imread(f'images_original/{image_name}')

    # Get image size
    (h, w) = img.shape[:2]

    if (h == 1536) and (w == 2048):

        # Rotate the image 90 degrees clockwise
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # Save the image
        cv2.imwrite(f'images_original/rotated_{image_name}', img)