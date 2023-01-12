from PIL import Image, ImageFilter

class Picture():
    
    def __init__(self, name) -> None:
        self.name = name
        self.image = Image.open(self.name)
        self.image.show()
    
    def convert2gray(self):
        self.img_gray = self.image.convert("L") # converts the image to grayscale
        self.img_gray.show()
    
    def blur_picture(self):
        self.img_blur = self.img_gray.filter(ImageFilter.GaussianBlur(5))
        self.img_blur.show()
    
    def erode(self):
        # First convert the image to its binary version
        threshold = 155
        img_bin = self.img_gray.point(lambda x: 255 if x > threshold else 0)
        img_bin = img_bin.convert("1")
        
        # Erode the image three times with a 3x3 kernel
        for _ in range(3):
            self.img_erode = img_bin.filter(ImageFilter.MaxFilter(3))
        
        self.img_erode.show()
    
    def find_edges(self):
        img_smooth = self.img_erode.filter(ImageFilter.SMOOTH) # First smooth the image
        self.img_edges = img_smooth.filter(ImageFilter.FIND_EDGES) # Second, find the edges
        self.img_edges.show()

if __name__== '__main__':

    img = Picture('images/IMG_6646.JPEG')

    img.convert2gray()

    img.blur_picture()
    
    img.erode()

    img.find_edges()