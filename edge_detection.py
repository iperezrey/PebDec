import cv2
import numpy as np

"""This file includes the testing of several
edge detection methods"""

"""FAST Method"""
# # Reading the image and converting it to b&w
# image = cv2.imread('images_original/IMG_6747.jpeg')
# image_graysc = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# # Application of the FAST function
# fast = cv2.FastFeatureDetector_create()
# fast.setNonmaxSuppression(False)

# # Drawing the keypoints
# kp = fast.detect(image_graysc,None)
# kp_image = cv2.drawKeypoints(image,kp,None,color=(0,255,0))

# # Displaying image
# cv2.imshow('FAST',kp_image)

# # Set visualization time (0 means forever)
# cv2.waitKey()

"""ORB Method"""
# # Read the image and convert to grayscale
# image = cv2.imread('images_original/test_1.jpg')
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Applying Oriented FAST and Rotated Brief
# orb = cv2.ORB_create(nfeatures=2000)
# kp, des = orb.detectAndCompute(gray_image, None)

# # Drawing the keypoints
# kp_image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)

# # Show the results
# cv2.imshow('ORB', kp_image)
# cv2.waitKey()

"""Harris corner detector"""
# # Reading the image and converting it to b&w
# image = cv2.imread('images_original/IMG_6701.jpeg')

# image_graysc = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# image_graysc = np.float32(image_graysc)

# # Applying Harris function
# dst = cv2.cornerHarris(image_graysc,blockSize=2,ksize=3,k=0.04)

# # Dilate to mark the corners
# dst = cv2.dilate(dst,None)
# image[dst > 0.01 * dst.max()] = [0, 255, 0]

# # Displaying image
# cv2.imshow('image_gs_test',image)

# # Set visualization time (0 means forever)
# cv2.waitKey()

"""Sobel/Canny/Laplacian + contour detection"""
# Read the image and convert to grayscale
img = cv2.imread('images_original/IMG_6701.jpeg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
img_blur = cv2.GaussianBlur(img_gray, (13, 13), 0)

# Apply different filters: Sobel, Canny or Laplace
img_laplace = cv2.Laplacian(src=img_blur, ddepth=cv2.CV_64F)

sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
img_sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

# Canny Edge Detection
img_canny = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) 

# Apply binary thresholding
ret, thresh = cv2.threshold(img_blur, 150, 255, cv2.THRESH_BINARY) # here we can pass the blur image or canny filtered

# Contour detection
img_contour = img.copy()
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:
        cv2.drawContours(img_contour, contour, -1, (255, 10, 0), 2)
        perimeter = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
        x, y, width, height = cv2.boundingRect(corners)
        cv2.rectangle(img_contour, (x, y), (x+width, y+height), (0, 255, 0), 2) # Maybe change for a circle

# Displaying image
cv2.imshow('image test', img_contour)

# cv2.imshow('laplace', img_laplace)
# cv2.imshow('soble', img_sobelxy)
# cv2.imshow('canny', img_canny)

# Set visualization time (0 means forever)
cv2.waitKey()
