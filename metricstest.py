from sklearn.metrics import precision_score, recall_score, f1_score
import cv2 as cv

# Number of the IMG
num_image = '6762'

# Routes to the manual and predicted mask by SAM
route_manual = 'metrics/Lstudio_'+ num_image +'_2.png'
route_SAM = 'metrics/IMG_'+ num_image +'_masks.jpeg'

# Open both masked images
image_manual = cv.imread(route_manual)

image_SAM = cv.imread(route_SAM)

# Obtain heigth and width from labelStudio masked images
(h,w) = image_manual.shape[:2]
print (h,w)

# Resize SAM masked images
image_SAM_r = cv.resize(image_SAM, (768,1024))
print (image_SAM_r.shape[:2])

# Convert to grayscale
image_manual_g = cv.cvtColor(image_manual, cv.COLOR_BGR2GRAY)
image_SAM_g = cv.cvtColor(image_SAM_r, cv.COLOR_BGR2GRAY)

# Binarize the images
_, image_manual_b = cv.threshold(image_manual_g, 1, 255, cv.THRESH_BINARY)
_, image_SAM_b = cv.threshold(image_SAM_g, 1, 255, cv.THRESH_BINARY)

# Flattening binary images to one-dimensional arrays
image_manual_flat = image_manual_b.flatten()
image_SAM_flat = image_SAM_b.flatten()

# Calculate metrics
precision = precision_score(image_manual_flat, image_SAM_flat, pos_label=255)
recall = recall_score(image_manual_flat, image_SAM_flat, pos_label=255)
f1 = f1_score(image_manual_flat, image_SAM_flat, pos_label=255)

# Print the results
print(f'Precisión: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
