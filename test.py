import cv2

class Image:
    
    def __init__(self, name) -> None:
        self.name = name
        self.image = cv2.imread(self.name, cv2.IMREAD_COLOR)
    
    # Make method to reduce the size of the image. See cvgranite

    def convert2gray(self):
        # img_gray = cv2.cvtColor(self.read_image(), cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray image', img_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img_gray

if __name__ == '__main__':

    img = Image('images/IMG_6669.jpeg')

    img.convert2gray()