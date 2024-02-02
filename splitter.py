# Import libraries (os - Operating System; cv2 - OpenCV)
import os
import cv2

def splitter(image_name):

    # Read image from folder
    image = cv2.imread(f'images_original/{image_name}')
  
    # Extract height and width of the image
    print(image.shape[:2])
    (h,w) = image.shape[:2]

    # Loop reading the full image to crop into smaller pieces
    x = 0
    y = 0
    x_end = 256
    y_end = 256

    for i in range(48):
        cropped_image = image[y:y_end,x:x_end]
        # cv2.imshow('test_img',cropped_image)
        # cv2.waitKey(100)
        cv2.imwrite(f'images/{image_name[:-5]}_{i}.jpeg', cropped_image)
        
        # Rotates cropped image and saves it
        rot_cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(f'images/{image_name[:-5]}_{i}_rot.jpeg', rot_cropped_image)

        print(i)
        print(x, y, x_end, y_end)

        x += 256
        x_end += 256

        if x == 1536:
            x = 0
            x_end = 256
            y += 256
            y_end += 256

if __name__ == '__main__':
    
    image_names = os.listdir('images_original')
    for image_name in image_names:
        
        splitter(image_name=image_name)
    