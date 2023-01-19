# Import libraries
import cv2

# Reading the image and converting it to b&w
image = cv2.imread('images/IMG_6747.jpeg')
image_graysc = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Application of the FAST function
fast = cv2.FastFeatureDetector_create()
fast.setNonmaxSuppression(False)

# Drawing the keypoints
kp = fast.detect(image_graysc,None)
kp_image = cv2.drawKeypoints(image,kp,None,color=(0,255,0))

# Displaying image
cv2.imshow('FAST',kp_image)

# Set visualization time (0 means forever)
cv2.waitKey()
