from PIL import Image
import os

# path to list
datapath = "/home/andreas/Documents/Seminar/appliedml_using_gan/Data/pictures_original/"
dire = os.listdir(datapath)

savedir = "/home/andreas/Documents/Seminar/appliedml_using_gan/Data/pictures_resized/"

#newSize
newSize=300

if newSize > 0:

    elementsInDirectory = len(os.listdir(savedir))
    if elementsInDirectory != 0:
        os.chdir(savedir)
        os.system("rm *")

    for file in dire:
        image = Image.open(datapath + file)
        image = image.resize((newSize, newSize))
        image.save(savedir + file)

    print("Successfully filled directory with resized images")
