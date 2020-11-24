# performs augmentations, resizes and merges
#from flip_horizontal import flip_horizontal
from PIL import Image, ImageEnhance
import os


# final_size
final_size = (128,128)

# source
datapath = "../Data/pictures_resized/"
source = os.listdir(datapath)

# destination
destination = os.path.join("../Data/Train/",
                           str(final_size[0]) + str(final_size[1])+"/")
# "/home/andreas/Documents/Seminar/appliedml_using_gan/Data/pictures_all/"
os.system("rm -rf {}".format(destination))

os.mkdir(destination)

# initial size
width, height = Image.open(datapath + source[0]).size


# methods
# flip_horizontal.__name__
def flip_horizontal():
    for file in source:
        image = Image.open(datapath + file)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = image.resize(final_size)
        image.save(os.path.join(destination + file + '_flipped.png'))


def rotate(degree):
    for file in source:
        image = Image.open(datapath + file)
        image = image.rotate(degree)
        image = image.crop((35, 35, width - 35, height - 35))
        image = image.resize(final_size)
        image.save(os.path.join(destination + file + str(degree) + '_rotate.png'))


def crop(left, top, right, bottom, description):
    for file in source:
        image = Image.open(datapath + file)
        image = image.crop((left, top, right, bottom))
        image = image.resize(final_size)
        image.save(destination + file + description + '_cropped.png')

def sharpen(factor):
    for file in source:
        image = Image.open(datapath + file)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)
        image = image.resize(final_size)
        image.save(destination + file + str(factor) + '_enhanced.png')


if __name__ == "__main__":
    flip_horizontal()
    rotate(10)
    rotate(350)
    #lefttop
    crop(0, 0, 3 * width / 4.0, 3 * height / 4.0, "lefttop")
    #righttop, überarbeiten!
    crop(width / 4.0, 0,  width , 3 * height / 4.0, "righttop")
    #rightbottom
    crop( width / 4.0, height / 4.0, width, height, "rightbottom")
    #leftbottom
    crop(0, height / 4, 3 * width / 4, height, "leftbottom")
    sharpen(0.5)
    sharpen(1.5)
    sharpen(0.75)
    sharpen(1.25)
