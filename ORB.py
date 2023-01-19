import cv2

# Read the image and convert to grayscale
image = cv2.imread('images/test_1.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying Oriented FAST and Rotated Brief
orb = cv2.ORB_create(nfeatures=2000)
kp, des = orb.detectAndCompute(gray_image, None)

# Drawing the keypoints
kp_image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)

# Show the results
cv2.imshow('ORB', kp_image)
cv2.waitKey()
