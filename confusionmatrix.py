from sklearn.metrics import confusion_matrix
import cv2 as cv
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Route to images
routes_manual = ['metrics/matrixtest/manual/Lstudio_6648_2.png','metrics/matrixtest/manual/Lstudio_6662_2.png','metrics/matrixtest/manual/Lstudio_6666_2.png','metrics/matrixtest/manual/Lstudio_6673_2.png','metrics/matrixtest/manual/Lstudio_6701_2.png','metrics/matrixtest/manual/Lstudio_6710_2.png','metrics/matrixtest/manual/Lstudio_6716_2.png','metrics/matrixtest/manual/Lstudio_6720_2.png','metrics/matrixtest/manual/Lstudio_6750_2.png','metrics/matrixtest/manual/Lstudio_6762_2.png']
routes_SAM = ['metrics/matrixtest/SAM/IMG_6648_masks.jpeg','metrics/matrixtest/SAM/IMG_6662_masks.jpeg','metrics/matrixtest/SAM/IMG_6666_masks.jpeg','metrics/matrixtest/SAM/IMG_6673_masks.jpeg','metrics/matrixtest/SAM/IMG_6701_masks.jpeg','metrics/matrixtest/SAM/IMG_6710_masks.jpeg','metrics/matrixtest/SAM/IMG_6716_masks.jpeg','metrics/matrixtest/SAM/IMG_6720_masks.jpeg','metrics/matrixtest/SAM/IMG_6750_masks.jpeg','metrics/matrixtest/SAM/IMG_6762_masks.jpeg']

# Initialize matrices for TP, FN, FP, TN
TP_total = 0
FN_total = 0
FP_total = 0
TN_total = 0

for route_manual, soute_SAM in zip (routes_manual, routes_SAM):
    # Load manual (LabelStudio) masked image
    image_manual = cv.imread(route_manual)

    # Load SAM masked image
    image_SAM = cv.imread(route_manual)

    # Obtain heigth and width from labelStudio masked images
    (h,w) = image_manual.shape[:2]
    
    # Resize SAM masked images
    image_SAM_r = cv.resize(image_SAM, (768,1024))

    # Convert to grayscale
    image_manual_g = cv.cvtColor(image_manual, cv.COLOR_BGR2GRAY)
    image_SAM_g = cv.cvtColor(image_SAM_r, cv.COLOR_BGR2GRAY)

    # Binarize the images
    _, image_manual_b = cv.threshold(image_manual_g, 1, 255, cv.THRESH_BINARY)
    _, image_SAM_b = cv.threshold(image_SAM_g, 1, 255, cv.THRESH_BINARY)

    # Flattening binary images to one-dimensional arrays
    image_manual_flat = image_manual_b.flatten()
    image_SAM_flat = image_SAM_b.flatten()

    # Calculate elements of the confusion matrix for the current image
    TP = np.sum((image_manual_flat == 1) & (image_SAM_flat == 1))
    FN = np.sum((image_manual_flat == 1) & (image_SAM_flat == 0))
    FP = np.sum((image_manual_flat == 0) & (image_SAM_flat == 1))
    TN = np.sum((image_manual_flat == 0) & (image_SAM_flat == 0))

    # Accumulate values for total matrix
    TP_total += TP
    FN_total += FN
    FP_total += FP
    TN_total += TN

   

# Build the total confusion matrix
confusion_matrix_total = np.array([[TP_total, FN_total], [FP_total, TN_total]])

# Print the results
print("Total confusion matrix:")
print(confusion_matrix_total)