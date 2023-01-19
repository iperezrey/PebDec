# Import libraries
import cv2

# Read image from folder
image = cv2.imread('images_original/IMG_6646.jpeg')

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
    cv2.imwrite(f'images/{i}.jpeg', cropped_image)
    
    print(i)
    print(x, y, x_end, y_end)

    x += 256
    x_end += 256

    if x == 1536:
        x = 0
        x_end = 256
        y += 256
        y_end += 256
    
  