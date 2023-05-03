import sys
sys.path.append('..')
import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.style.use('ggplot')

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def map_to_rainbow(lst):
    """
    Maps each integer in a sorted list of integers to a color in the "rainbow" color map from matplotlib.

    Parameters:
    lst (list[int]): The sorted list of integers to be mapped to colors.

    Returns:
    list[str]: The list of colors mapped from the sorted list of integers.
    """
    # Get the minimum and maximum values in the list
    vmin = lst[0]
    vmax = lst[-1]

    # Create a color map using the "rainbow" colormap from matplotlib
    cmap = matplotlib.colormaps['rainbow']

    # Map each integer in the list to a color in the color map
    colors = [cmap((x - vmin) / (vmax - vmin)) for x in lst]

    # Convert the colors to hexadecimal format and return as a list of strings
    return [list(mcolors.hex2color(mcolors.rgb2hex(color))) for color in colors]

def show_anns(anns, color_by_size):
    """
    Represent the masks over the original image and over a blank image. It also calculates 
    the total area of the masks (sq. pixels)
    -----------
    Arguments:
    anns (dict) -- dictionary containing 7 elements: 'segmentation', 'area', 'bbox', 'predicted_iou', 
        'point_cords', stability_score', 'crop_box'
    color_by_size (boolean) -- if True the colors of the masks are colored by size, from smallest to biggest, using 
        'rainbow' color map from matplotlib 
    Returns:
    total area (float) -- summ of all the areas corresponding to each of the pebbles
        detected in the image (sq. pixels)
    """

    if len(anns) == 0:
        return
        
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=False)
    # np.save('sorted_anns.npy', sorted_anns)
    
    # Extract and sort the areas
    sorted_areas = []      
    for ann in sorted_anns:
        sorted_areas.append(ann['area'])
    sorted_areas = sorted(sorted_areas) 

    # Get the color for each area
    colors = map_to_rainbow(lst=sorted_areas)
    
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    total_area = 0.0  # initialize total area to zero
    if color_by_size == True: 
        for i, ann in enumerate(sorted_anns):
            # Get the area and add it to get total
            mask_area = ann['area']
            total_area += mask_area
            # print(f"Mask area: {mask_area:.2f} sq. pixels")
            
            # Plotting
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))

            color = colors[i]
            
            img[:, :, 0] = color[0]
            img[:, :, 1] = color[1]
            img[:, :, 2] = color[2]
            ax.imshow(np.dstack((img, m * 0.35)))
    else:
        for ann in sorted_anns:
            m = ann['segmentation']
            mask_area = np.count_nonzero(m) * 1.0
            total_area += mask_area # add current mask area to total area
            # print(f"Mask area: {mask_area:.2f} sq. pixels")
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:,:,i] = color_mask[i]
            ax.imshow(np.dstack((img, m*0.35)))
            
    
    print(f"Total mask area: {total_area:.2f} sq. pixels")  # print total area
    return total_area


sam_checkpoint = 'sam_vit_h_4b8939.pth'
model_type = 'vit_h'
device = 'cpu'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

list_images = ['IMG_6648_res.JPEG', 'IMG_6662_res.JPEG', 'IMG_6666_res.JPEG', 'IMG_6682_res.JPEG', 'IMG_6701_res.JPEG', 'IMG_6710_res.JPEG', 'IMG_6716_res.JPEG', 'IMG_6720_res.JPEG', 'IMG_6750_res.JPEG', 'IMG_6762_res.JPEG']

for name in list_images:

    for points_per_side in range(16, 44, 2):

        print(name, points_per_side)

        image = cv.imread(f'images_selected/{name}')
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)


        # Extract height and width of the image
        print(image.shape[:2])
        (h,w) = image.shape[:2]
        

        # Create the masks using SamAutomaticMaskGenerator from Segment_anything
        mask_generator_ = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
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



        # Show the original img
        plt.figure(figsize=(7,7))
        plt.imshow(image)
        plt.axis('off')
        # plt.show()
        plt.savefig(f'results/{name[4:8]}/IMG_{name[4:8]}_original.JPEG')
        plt.close()

        # Plot the image with the masks
        plt.figure(1, figsize=(7, 7))
        plt.imshow(image)
        total_area = show_anns(masks, color_by_size=True)
        plt.axis('off')
        plt.savefig(f'results/{name[4:8]}/IMG_{name[4:8]}_{points_per_side}_masked.JPEG')
        plt.close()

        # Plot the mask over a blank image
        blank = np.zeros((image.shape), dtype='uint8')
        plt.figure(2, figsize=(7, 7))
        plt.imshow(blank)
        total_area = show_anns(masks, color_by_size=True)
        plt.axis('off')
        plt.savefig(f'results/{name[4:8]}/IMG_{name[4:8]}_{points_per_side}_masks.JPEG')
        plt.close()

        # Generate a histogram of mask areas
        list_mask_areas = []
        for mask in masks:
            area = np.count_nonzero(mask['segmentation']) * 1.0
            list_mask_areas.append(area)

        # Plot the histogram of mask areas
        plt.figure(3, figsize=(7,7))
        plt.hist(list_mask_areas, bins=20, rwidth=0.7)
        plt.title('Mask histogram IMG ' + name[4:8])
        plt.xlabel('Mask area (sq. pixels)')
        plt.ylabel('Frequency')
        # plt.show()
        plt.savefig(f'results/{name[4:8]}/IMG_{name[4:8]}_{points_per_side}_histogram.JPEG')
        plt.close()

        # Percentage of pixels (pebbles)
        percentage = (total_area / (h * w)) * 100
        print(f'Percentage of pebbles: {percentage:.2f}%')
        with open(f'results/{name[4:8]}/percentage_pebbles_{name[4:8]}.txt', 'a') as text_file:
            text_file.write(f'Percentage of pebbles for points_per_side={points_per_side:.2f}: {percentage:.2f}%\n')