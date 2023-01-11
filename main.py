from PIL import Image, ImageFilter

filename = 'images/IMG_6669.jpeg'

# Open image
im = Image.open(filename)
# im.show()

# Converts to grayscale
im_graysc = im.convert("L")

# Detect edges with FIND_EDGES
# im_find = im.filter(ImageFilter.FIND_EDGES)
# im_find.show()

# Detect edges with Laplacian Kernel
im_lKernel = im.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), 1, 0))
im_lKernel.show()

# Enhance edges
im_enhance = im.filter(ImageFilter.EDGE_ENHANCE)
im_find_enhance = im_enhance.filter(ImageFilter.FIND_EDGES)
im_find_enhance.show()

# Binarize image
#   Define theshold
threshold = 150

im_threshold = im_graysc.point(lambda p: 255 if p > threshold else 0)
im_threshold.show()

#   Convert to mono
im_mono = im_threshold.convert('1')
im_mono.show()