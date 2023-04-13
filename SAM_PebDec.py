
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys
sys.path.append('..')
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

image = cv.imread('D:/software/PebDec/images_original/IMG_6710.JPEG')
# cv.imshow('ImgPrueba', img)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# cv.imshow('ImgPruebaRGB', img)
# cv.waitKey(0)

# Extract height and width of the image
print(image.shape[:2])
(h,w) = image.shape[:2]

plt.figure(figsize=(7,7))
plt.imshow(image)
plt.axis('off')
plt.show()

sam_checkpoint = 'sam_vit_h_4b8939.pth'
model_type = 'vit_h'
device = 'cuda'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
# print(sam)

mask_generator_ = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.96,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100
)

masks = mask_generator_.generate(image)

print(len(masks))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    total_area = 0.0  # initialize total area to zero
    for ann in sorted_anns:
        m = ann['segmentation']
        mask_area = np.count_nonzero(m) * 1.0
        total_area += mask_area  # add current mask area to total area
        print(f"Mask area: {mask_area:.2f} sq. pixels")
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
    print(f"Total mask area: {total_area:.2f} sq. pixels")  # print total area
    
    return total_area

plt.figure(figsize=(7, 7))
plt.imshow(image)
total_area = show_anns(masks)
plt.axis('off')
plt.show()

# Percentage of pixels (pebbles)
percentage = total_area / (h * w)
print(percentage)