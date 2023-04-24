import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load sorted_anns
sorted_anns = np.load('sorted_anns.npy', allow_pickle=True)

# Extract and sort the areas
sorted_areas = []
for ann in sorted_anns:
    sorted_areas.append(ann['area'])
sorted_areas = sorted(sorted_areas)

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

colors = map_to_rainbow(lst=sorted_areas)

print(colors)

# Build the loop
total_area = 0
for i, ann in enumerate(sorted_anns):
    mask_area = ann['area']
    total_area += mask_area
    
    m = ann['segmentation'] # Segmentation contains a boolean that tells us whether a pixel has a mask or not. 
    # That is why in the previous 
    img = np.ones((m.shape[0], m.shape[1], 3))