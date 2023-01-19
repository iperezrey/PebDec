# Import libraries
import cv2
import numpy as np

# Reading the image and converting it to b&w
image = cv2.imread('images/IMG_6701.jpeg')

image_graysc = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image_graysc = np.float32(image_graysc)


# Applying Harris function
dst = cv2.cornerHarris(image_graysc,blockSize=2,ksize=3,k=0.04)


# Dilate to mark the corners
dst = cv2.dilate(dst,None)
image[dst > 0.01 * dst.max()] = [0, 255, 0]

# Displaying image
cv2.imshow('image_gs_test',image)

# Set visualization time (0 means forever)
cv2.waitKey()