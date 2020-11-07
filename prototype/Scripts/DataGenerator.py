from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image
img = load_img('/home/andreas/Documents/Seminar/appliedml_using_gan/Test/prediction.png')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(horizontal_flip=True, zoom_range=[0.5,1.0],shear_range=90)#horizontal_flip=True, zoom_range=[0.5,1.0],shear_range=90
# prepare iterator
#it = datagen.flow(samples, batch_size=1)
it = datagen.flow_from_directory(
    directory=r'/home/andreas/Documents/Seminar/appliedml_using_gan/Test/result2500/',
    class_mode='categorical',
    target_size=(300,300),
    shuffle=True,
    batch_size=10
)
# generate samples and plot
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
pyplot.savefig("Test/Aug.jpg")
