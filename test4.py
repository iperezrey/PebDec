import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.cm as cm
import cv2 as cv
import sys
sys.path.append('..')
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

image = cv.imread('images_lowres/IMG_6674_res.JPEG')
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)


# Extract height and width of the image
print(image.shape[:2])
(h,w) = image.shape[:2]

blank = np.zeros((image.shape), dtype='uint8') # create an empty image to represent later all the masks


# Show the img
plt.figure(figsize=(7,7))
plt.imshow(image)
plt.axis('off')
plt.show()

sam_checkpoint = 'sam_vit_h_4b8939.pth'
model_type = 'vit_h'
device = 'cpu'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


# Create the masks using SamAutomaticMaskGenerator from Segment_anything
mask_generator_ = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    points_per_batch=64,
    pred_iou_thresh=0.9, # The lower the threshold the more objects will pick up
    stability_score_thresh=0.96,
    stability_score_offset=1.0, # The amount to shift the cutoff when calculated the stability score.
    box_nms_thresh=0.7,
    crop_n_layers=0, # If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.
    crop_nms_thresh=0.7,
    crop_overlap_ratio=512 / 1500,
    crop_n_points_downscale_factor=2,
    point_grids=None,
    min_mask_region_area=0,
    output_mode='binary_mask'
    )

masks = mask_generator_.generate(image)

# print(masks)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    total_area = 0.0  # initialize total area to zero
    for  ann in sorted_anns:
        m = ann['segmentation']
        mask_area = np.count_nonzero(m) * 1.0
        total_area += mask_area # add current mask area to total area
        # print(f"Mask area: {mask_area:.2f} sq. pixels")
        color = cm.rainbow(mask_area/np.max([ann['area'] for ann in anns]))
        color_mask = np.zeros_like(image)
        color_mask[:, :, 0] = color[0] * 255
        color_mask[:, :, 1] = color[1] * 255
        color_mask[:, :, 2] = color[2] * 255
        alpha_mask = (m*0.35).astype(np.uint8)*255
        if alpha_mask.max() > 0:  # only show mask if there are any non-zero alpha values
            color_mask = cv.cvtColor(color_mask, cv.COLOR_RGB2RGBA)
            color_mask[:, :, 3] = alpha_mask
            ax.imshow(color_mask)
    print(f"Total mask area: {total_area:.2f} sq. pixels")  # print total area
    return total_area, color_mask


# Plot the image with the masks
plt.figure(1, figsize=(7, 7))
plt.imshow(image)
total_area, color_mask = show_anns(masks)
plt.axis('off')

# Plot the mask over a blank image
plt.figure(2, figsize=(7, 7))
plt.imshow(blank)
total_area, color_mask = show_anns(masks)
plt.axis('off')

# Generate an histogram of mask areas
list_mask_areas = []
for mask in masks:
    area = np.count_nonzero(mask['segmentation']) * 1.0
    list_mask_areas.append(area)

# Plot the histogram of mask areas
plt.figure(3, figsize=(7,7))
plt.hist(list_mask_areas, bins=20, rwidth=0.7)
plt.title('Mask histogram')
plt.xlabel('Mask area (sq. pixels)')
plt.ylabel('Frequency')
plt.show()

# Percentage of pixels (pebbles)
percentage = total_area / (h * w)
print(f'Percentage of pebbles : {percentage:.4f}')