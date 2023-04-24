import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cv2 as cv
import sys
sys.path.append('..')
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# image = cv.imread('images_lowres/IMG_6674_res.JPEG')
# image = cv.cvtColor(image, cv.COLOR_BGR2RGB)


rainbow = mpl.colormaps['rainbow'].resampled(8)

print('rainbow.colors', rainbow.color)
print('rainbow(range(8))', rainbow(range(8)))
print('rainbow(np.linspace(0, 1, 8))', rainbow(np.linspace(0, 1, 8)))