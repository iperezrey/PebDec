import matplotlib.pyplot as plt
import cv2 as cv

figsize = (10, 7)
rows = 1
columns = 2


""" FAST METHOD RESULT """
# FastMethod = cv.imread('results/Methods/FAST Method/FAST_method.jpg')
# FastMethod = cv.cvtColor(FastMethod, cv.COLOR_BGR2RGB)
# ImgOrigFAST = cv.imread('results/Methods/FAST Method/IMG_6747.jpeg')
# ImgOrigFAST = cv.cvtColor(ImgOrigFAST, cv.COLOR_BGR2RGB)

# #Create the figure
# figFast = plt.figure(figsize=figsize)

# #Add firts image
# figFast.add_subplot(rows, columns, 2)

# plt.imshow(FastMethod)
# plt.axis('off')
# plt.title('FAST method')

# #Add second image
# figFast.add_subplot(rows, columns, 1)

# plt.imshow(ImgOrigFAST)
# plt.axis('off')
# plt.title('Original image')

# #Save the figure
# plt.savefig('results/Methods/FAST Method/FAST mehtod figure.jpg', dpi=300)
# plt.show()


""" HARRIS CORNER DETECTOR RESULTS """
# HarrisMethod = cv.imread('results/Methods/Harris corner detector/Harris_method.jpg')
# HarrisMethod = cv.cvtColor(HarrisMethod, cv.COLOR_BGR2RGB)
# ImgOrigHarris = cv.imread('results/Methods/Harris corner detector/IMG_6701.jpeg')
# ImgOrigHarris = cv.cvtColor(ImgOrigHarris, cv.COLOR_BGR2RGB)

# #Create the figure
# figHarris = plt.figure(figsize=figsize)

# #Add first image
# figHarris.add_subplot(rows, columns, 2)

# plt.imshow(HarrisMethod)
# plt.axis('off')
# plt.title('Harris corner detector method')

# #Add second image
# figHarris.add_subplot(rows, columns, 1)

# plt.imshow(ImgOrigHarris)
# plt.axis('off')
# plt.title('Original image')

# #Save the figure
# plt.savefig('results/Methods/Harris corner detector/Harris corner detector figure.jpg', dpi=300)
# plt.show()


""" ORB METHOD RESULTS """
# ORBMethod = cv.imread('results/Methods/ORB Method/ORB_method.jpg')
# ORBMethod = cv.cvtColor(ORBMethod, cv.COLOR_BGR2RGB)
# ImgOrigORB = cv.imread('results/Methods/ORB Method/IMG_6701.jpeg')
# ImgOrigORB = cv.cvtColor(ImgOrigORB, cv.COLOR_BGR2RGB)

# #Create the figure
# figORB = plt.figure(figsize=figsize)

# #Add first image
# figORB.add_subplot(rows, columns, 2)

# plt.imshow(ORBMethod)
# plt.axis('off')
# plt.title('ORB method')

# #Add second image
# figORB.add_subplot(rows, columns, 1)

# plt.imshow(ImgOrigORB)
# plt.axis('off')
# plt.title('Original image')

# #Save the figure
# plt.savefig('results/Methods/ORB Method/ORB method figure.jpg', dpi=300)
# plt.show()



""" SOBEL, CANNY AND LAPLACIAN PLUS CONTOUR DETECTION """
ImgOri = cv.imread('results/Methods/Sobel_Canny_Laplacian_contour detection/IMG_6701.jpeg')
ImgOri = cv.cvtColor(ImgOri, cv.COLOR_BGR2RGB)

Sobel = cv.imread('results/Methods/Sobel_Canny_Laplacian_contour detection/Sobel.jpg')
SobelContour = cv.imread('results/Methods/Sobel_Canny_Laplacian_contour detection/Sobel contours.jpg')
SobelContour = cv.cvtColor(SobelContour, cv.COLOR_BGR2RGB)

Canny = cv.imread('results/Methods/Sobel_Canny_Laplacian_contour detection/Canny.jpg')
CannyContour = cv.imread('results/Methods/Sobel_Canny_Laplacian_contour detection/Canny contours.jpg')
CannyContour = cv.cvtColor(CannyContour, cv.COLOR_BGR2RGB)

Laplacian = cv.imread('results/Methods/Sobel_Canny_Laplacian_contour detection/Laplacian.jpg')
LaplacianContour = cv.imread('results/Methods/Sobel_Canny_Laplacian_contour detection/Laplacian contours.jpg')
LaplacianContour = cv.cvtColor(LaplacianContour, cv.COLOR_BGR2RGB)


#Create the figure
fig = plt.figure(figsize=(10, 10))

#Add first image
fig.add_subplot(3, 3, 1)

plt.imshow(ImgOri)
plt.axis('off')
plt.title('Original image')

#Add second image
fig.add_subplot(3, 3, 2)

plt.imshow(Sobel)
plt.axis('off')
plt.title('Sobel')

#Add third image
fig.add_subplot(3, 3, 3)

plt.imshow(SobelContour)
plt.axis('off')
plt.title('Sobel + contour detector')

#Add fourth image
fig.add_subplot(3, 3, 4)

plt.imshow(ImgOri)
plt.axis('off')
plt.title('Original image')

#Add fifth image
fig.add_subplot(3, 3, 5)

plt.imshow(Canny)
plt.axis('off')
plt.title('Canny')

#Add sixth image
fig.add_subplot(3, 3, 6)

plt.imshow(CannyContour)
plt.axis('off')
plt.title('Canny + contour detector')

#Add seventh image
fig.add_subplot(3, 3, 7)

plt.imshow(ImgOri)
plt.axis('off')
plt.title('Original image')

#Add eighth image
fig.add_subplot(3, 3, 8)

plt.imshow(Laplacian)
plt.axis('off')
plt.title('Laplacian')

#Add ninth image
fig.add_subplot(3, 3, 9)

plt.imshow(LaplacianContour)
plt.axis('off')
plt.title('Laplacian + contour detector')


plt.savefig('results/Methods/Sobel_Canny_Laplacian_contour detection/Figure.jpg', dpi=300)
plt.show()

