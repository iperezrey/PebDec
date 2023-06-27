import matplotlib.pyplot as plt
import cv2
import numpy as np

figsize = (10, 7)
# rows = 1
# columns = 2

"""FAST Method"""
# # Reading the image and converting it to b&w 
# image1 = cv2.imread('images_original/IMG_6762.jpeg')
# image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
# image_graysc1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)

# # Application of the FAST function
# fast = cv2.FastFeatureDetector_create()
# fast.setNonmaxSuppression(False)

# # Drawing the keypoints
# kp = fast.detect(image_graysc1,None)
# kp_image = cv2.drawKeypoints(image1,kp,None,color=(0,255,0))

# #Create the figure
# figFast = plt.figure(figsize=figsize)

# #Add firts image
# figFast.add_subplot(rows, columns, 2)

# plt.imshow(kp_image)
# plt.axis('off')
# plt.title('FAST method')

# #Add second image
# figFast.add_subplot(rows, columns, 1)

# plt.imshow(image1)
# plt.axis('off')
# plt.title('Original image')

# #Save the figure
# plt.savefig('new/FAST mehtod figure.jpg', dpi=300)
# plt.show()


"""ORB Method"""
# # Read the image and convert to grayscale
# image2 = cv2.imread('images_original/IMG_6762.jpeg')
# image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
# gray_image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# # Applying Oriented FAST and Rotated Brief
# orb = cv2.ORB_create(nfeatures=2000)
# kp, des = orb.detectAndCompute(gray_image, None)

# # Drawing the keypoints
# kp_image2 = cv2.drawKeypoints(image2, kp, None, color=(0, 255, 0), flags=0)

# #Create the figure
# figORB = plt.figure(figsize=figsize)

# #Add first image
# figORB.add_subplot(rows, columns, 2)

# plt.imshow(kp_image2)
# plt.axis('off')
# plt.title('ORB method')

# #Add second image
# figORB.add_subplot(rows, columns, 1)

# plt.imshow(image2)
# plt.axis('off')
# plt.title('Original image')

# # #Save the figure
# plt.savefig('new/ORB mehtod figure.jpg', dpi=300)
# plt.show()

"""Harris corner detector"""
# # Reading the image and converting it to b&w
# image3 = cv2.imread('images_original/IMG_6762.jpeg')
# image31 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

# image_graysc = cv2.cvtColor(image31,cv2.COLOR_BGR2GRAY)
# image_graysc = np.float32(image_graysc)

# # Applying Harris function
# dst = cv2.cornerHarris(image_graysc,blockSize=2,ksize=3,k=0.04)

# # Dilate to mark the corners
# dst = cv2.dilate(dst,None)
# image3[dst > 0.01 * dst.max()] = [0, 255, 0]
# image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

# # #Create the figure
# figHarris = plt.figure(figsize=figsize)

# #Add first image
# figHarris.add_subplot(rows, columns, 2)

# plt.imshow(image3)
# plt.axis('off')
# plt.title('Harris corner detector method')

# #Add second image
# figHarris.add_subplot(rows, columns, 1)

# plt.imshow(image31)
# plt.axis('off')
# plt.title('Original image')

# #Save the figure
# plt.savefig('new/Harris corner detector figure.jpg', dpi=300)
# plt.show()

"""Sobel/Canny/Laplacian + contour detection"""
# Read the image and convert to grayscale
# img = cv2.imread('images_original/IMG_6762.jpeg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply Gaussian blur
# img_blur = cv2.GaussianBlur(img_gray, (13, 13), 0)

# # Apply different filters: Sobel, Canny or Laplace
# img_laplace = cv2.Laplacian(src=img_blur, ddepth=cv2.CV_64F)

# sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
# sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
# img_sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

# # Canny Edge Detection
# img_canny = cv2.Canny(image=img_blur, threshold1=50, threshold2=100) 

# # Apply binary thresholding
# ret, thresh = cv2.threshold(img_canny, 150, 255, cv2.THRESH_BINARY) # here we can pass the blur image or canny filtered

# # Convert binary image to CV_8UC1 format (in Laplace and Sobel)
# thresh = cv2.convertScaleAbs(thresh)

# # Contour detection
# img_contour = img.copy()
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for contour in contours:
#     area = cv2.contourArea(contour)
#     if area > 100:
#         cv2.drawContours(img_contour, contour, -1, (255, 10, 0), 2)
#         perimeter = cv2.arcLength(contour, True)
#         corners = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
#         x, y, width, height = cv2.boundingRect(corners)
#         cv2.rectangle(img_contour, (x, y), (x+width, y+height), (0, 255, 0), 2) # Maybe change for a circle

# # Displaying image
# cv2.imshow('image test', img_contour)
# # cv2.imwrite('results/Methods/Sobel_Canny_Laplacian_contour detection/Canny contours.jpg', img_contour)


# cv2.imshow('laplace', img_laplace)
# # cv2.imwrite('results/Methods/Sobel_Canny_Laplacian_contour detection/Laplacian.jpg', img_laplace)
# cv2.imshow('soble', img_sobelxy)
# # cv2.imwrite('results/Methods/Sobel_Canny_Laplacian_contour detection/Sobel.jpg', img_sobelxy)
# cv2.imshow('canny', img_canny)
# # cv2.imwrite('results/Methods/Sobel_Canny_Laplacian_contour detection/Canny.jpg', img_canny)
 
# # # Set visualization time (0 means forever)
# # cv2.waitKey()


"""Plot theese three methods"""
ImgOri = cv2.imread('results/Methods/Sobel_Canny_Laplacian_contour detection/IMG_6762.jpeg')
ImgOri = cv2.cvtColor(ImgOri, cv2.COLOR_BGR2RGB)

Sobel = cv2.imread('results/Methods/Sobel_Canny_Laplacian_contour detection/Sobel.jpg')
SobelContour = cv2.imread('results/Methods/Sobel_Canny_Laplacian_contour detection/Sobel contours.jpg')
SobelContour = cv2.cvtColor(SobelContour, cv2.COLOR_BGR2RGB)

Canny = cv2.imread('results/Methods/Sobel_Canny_Laplacian_contour detection/Canny.jpg')
CannyContour = cv2.imread('results/Methods/Sobel_Canny_Laplacian_contour detection/Canny contours.jpg')
CannyContour = cv2.cvtColor(CannyContour, cv2.COLOR_BGR2RGB)

Laplacian = cv2.imread('results/Methods/Sobel_Canny_Laplacian_contour detection/Laplacian.jpg')
LaplacianContour = cv2.imread('results/Methods/Sobel_Canny_Laplacian_contour detection/Laplacian contours.jpg')
LaplacianContour = cv2.cvtColor(LaplacianContour, cv2.COLOR_BGR2RGB)


#Create the figure
fig = plt.figure(figsize=figsize)


#Add first image
fig.add_subplot(2, 3, 1)

plt.imshow(ImgOri)
plt.axis('off')
plt.title('Original image')

#Add second image
fig.add_subplot(2, 3, 2)

plt.imshow(Canny)
plt.axis('off')
plt.title('Canny')

#Add third image
fig.add_subplot(2, 3, 3)

plt.imshow(CannyContour)
plt.axis('off')
plt.title('Canny + contour detector')


plt.savefig('new/Canny contour detection.jpg', dpi=300)
plt.show()