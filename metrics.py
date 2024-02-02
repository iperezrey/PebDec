from sklearn.metrics import precision_score, recall_score, f1_score
import cv2

#Number of the Image
num_image = '6648'

# Routes to the manual and predicted masks by  SAM
route_manual = 'metrics/Lstudio_'+ num_image +'_2.png'
route_SAM = 'metrics/IMG_'+ num_image +'_masks.jpeg'

# Size of LabelStudio masked image (manual)
imagen_manual = cv2.imread(route_manual)
height, width = imagen_manual.shape[:2]
print(height,width)

# Resize SAm masked image
mask_SAM = cv2.imread(route_SAM)
mask_SAM = cv2.resize(mask_SAM, (height, width))

# Loading grayscale masks
mask_manual = cv2.imread(route_manual, cv2.IMREAD_GRAYSCALE)
mask_SAM = cv2.imread(route_SAM, cv2.IMREAD_GRAYSCALE)


# Apply threshold
_, mask_manual = cv2.threshold(mask_manual, 128, 255, cv2.THRESH_BINARY)
_, mask_SAM = cv2.threshold(mask_SAM, 128, 255, cv2.THRESH_BINARY)


# Flattening binary images to one-dimensional arrays
mask_manual_flat = mask_manual.flatten()
mask_SAM_flat = mask_SAM.flatten()

# Calculate metrics
precision = precision_score(mask_manual_flat, mask_SAM_flat)
recall = recall_score(mask_manual_flat, mask_SAM_flat)
f1 = f1_score(mask_manual_flat, mask_SAM_flat)

print(f'Precisión: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
